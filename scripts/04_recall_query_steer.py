"""Query-space steering vs residual steering, standard protocol: extract on pairs, steer held-out prompts.

extract (fit needles, filler A):  pos = ctx + " Quick reminder, the secret word is"   (model recalls the needle)
                                  neg = ctx + " Anyway, the weather today is"           (it does not)
  q*[ℓ,h] = mean(q_pos − q_neg) at the last token, after q_norm, before RoPE     (full-attn layers)
  r*[ℓ]   = mean(h_pos − h_neg) at the last token, residual input to layer ℓ   (baseline)
steer (test needles, filler B, 4 unrelated endings), at the last token of every step:
  q-steer:  q_last += α·q*   then the model's own RoPE, softmax over this prompt's K, V, gate, o_proj
  r-steer:  h_last += α·r*

uv run scripts/04_recall_query_steer.py
"""
import argparse

import torch
import torch.nn.functional as F
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--layers", default="19,23,27,31")
p.add_argument("--q_alphas", default="2,4,8,16")
p.add_argument("--r_alphas", default="0.125,0.25,0.5,1")
p.add_argument("--n_gen", type=int, default=40)
p.add_argument("--device", default="cuda")
args = p.parse_args()

FILLER_A = " Yesterday I walked along the river, watched some boats drift past, and later had a long lunch with an old friend from school."
FILLER_B = " This morning my neighbour fixed his old bicycle, painted the garden fence, and then cooked a big pot of soup for his whole family."
FIT = ["violin", "tornado", "volcano", "cathedral"]
TEST = ["needle", "elephant", "dragon", "pirate", "wizard"]
POS, NEG = " Quick reminder, the secret word is", " Anyway, the weather today is"
ENDINGS = [" Anyway, the weather today is", " After lunch we decided to", " My favourite food is", " The meeting will start at"]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device).eval()
cfg = getattr(model.config, "text_config", model.config)
LAYERS = [int(x) for x in args.layers.split(",")]
assert all(cfg.layer_types[L] == "full_attention" for L in LAYERS)
decoder = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers

S = {"mode": "off", "alpha": 0.0, "q_star": {}, "r_star": {}, "cap_q": {}, "cap_h": {}}
orig_attn_forward = Qwen3_5Attention.forward


def attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    B, T, _ = hidden_states.shape
    hs = (B, T, -1, self.head_dim)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
    q = self.q_norm(q.view(hs)).transpose(1, 2)  # [B,H,T,d], before RoPE
    if S["mode"] == "capture" and self.layer_idx in LAYERS:
        S["cap_q"][self.layer_idx] = q[0, :, -1].float()
    if S["mode"] == "q" and self.layer_idx in LAYERS:
        q = q.clone()
        q[0, :, -1] += S["alpha"] * S["q_star"][self.layer_idx].to(q.dtype)
    k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    g = self.num_key_value_groups
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
    out = out.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))
    return self.o_proj(out), None


def resid_hook(layer_idx):
    def hook(module, args_, kwargs):
        h = kwargs["hidden_states"] if "hidden_states" in kwargs else args_[0]
        if S["mode"] == "capture":
            S["cap_h"][layer_idx] = h[0, -1].float()
        if S["mode"] == "r":
            h = h.clone()
            h[0, -1] += S["alpha"] * S["r_star"][layer_idx].to(h.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = h
                return args_, kwargs
            return (h, *args_[1:]), kwargs
    return hook


_ids = tok("The quick brown fox jumps over the lazy dog because", return_tensors="pt").input_ids.to(args.device)
with torch.no_grad():
    ref = model(_ids).logits.float()
    Qwen3_5Attention.forward = attn_forward
    err = (model(_ids).logits.float() - ref).abs().max().item()
logger.info(f"patched vs original max|Δlogit| = {err:.3f} (SHOULD be bf16 noise, <1)")
assert err < 1.0
for L in LAYERS:
    decoder[L].register_forward_pre_hook(resid_hook(L), with_kwargs=True)


@torch.no_grad()
def forward_last(text):
    ids = tok(text, return_tensors="pt").input_ids.to(args.device)
    return model(ids).logits[0, -1].float().log_softmax(-1)


# ---- extract on fit pairs
dq, dr = {L: [] for L in LAYERS}, {L: [] for L in LAYERS}
for n in FIT:
    ctx = f"The secret word is {n}. Remember it.{FILLER_A}"
    caps = []
    for end in (POS, NEG):
        S["mode"] = "capture"
        forward_last(ctx + end)
        caps.append(({L: S["cap_q"][L] for L in LAYERS}, {L: S["cap_h"][L] for L in LAYERS}))
    for L in LAYERS:
        dq[L].append(caps[0][0][L] - caps[1][0][L])
        dr[L].append(caps[0][1][L] - caps[1][1][L])
S["q_star"] = {L: torch.stack(dq[L]).mean(0) for L in LAYERS}
S["r_star"] = {L: torch.stack(dr[L]).mean(0) for L in LAYERS}
S["mode"] = "off"
for L in LAYERS:
    logger.info(f"L{L}: |q*|/|q| per head med {(S['q_star'][L].norm(dim=-1) / S['cap_q'][L].norm(dim=-1)).median():.2f}; "
                f"|r*|/|h| {S['r_star'][L].norm() / S['cap_h'][L].norm():.2f}")


@torch.no_grad()
def generate(text, n=args.n_gen):
    ids = tok(text, return_tensors="pt").input_ids.to(args.device)
    for _ in range(n):  # full recompute; steering applies at the last position each step
        nxt = model(ids).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
    return tok.decode(ids[0, -n:])


def prompt(n, end):
    return f"The secret word is {n}. Remember it.{FILLER_B}{end}"


configs = [("base", "off", 0.0)] + [(f"{m}-steer α={a}", m, float(a)) for m, al in (("q", args.q_alphas), ("r", args.r_alphas)) for a in al.split(",")]
rows, demos = [], {}
for name, mode, a in configs:
    dlp, kls, hits = [], [], []
    for n in TEST:
        nid = tok(" " + n).input_ids
        assert len(nid) == 1, n
        for end in ENDINGS:
            S["mode"] = "off"
            lp0 = forward_last(prompt(n, end))
            S["mode"], S["alpha"] = mode, a
            lp = forward_last(prompt(n, end))
            dlp.append((lp[nid[0]] - lp0[nid[0]]).item())
            kls.append(F.kl_div(lp, lp0, log_target=True, reduction="sum").item())
            g = generate(prompt(n, end))
            hits.append(n in g.lower())
            demos[(name, n, end)] = g
    rows.append(dict(config=name, needle_mentioned=sum(hits) / len(hits), d_logp_needle=sum(dlp) / len(dlp), kl=sum(kls) / len(kls)))
    logger.info(f"{name}: {rows[-1]}")
print(f"test: {len(TEST)} held-out needles × {len(ENDINGS)} endings, filler B; fit: {FIT}, filler A")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".2f"))
for name, _, _ in configs:
    for end in ENDINGS:
        print(f"{name:14s} | needle |{end}{demos[(name, 'needle', end)]!r}")
