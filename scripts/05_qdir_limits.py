"""Limits of the recall-query direction q*: which word does it fetch on frames it was not extracted on?

q* is extracted exactly as in 04 (fit words, filler A, "Quick reminder, the secret word is" vs "Anyway, the weather today is").
Then q_last += α·q* on new frames. X = marked target word, Y = unmarked distractor word (only some frames).
Hypotheses: H1 surface pattern ("secret word is"), H2 "the thing I was told to remember", H3 any salient noun, H4 early position.

uv run scripts/05_qdir_limits.py --device cpu --n_gen 0     # next-token log-probs only
uv run scripts/05_qdir_limits.py                            # + 30-token generations (GPU)
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
p.add_argument("--alphas", default="2,4")
p.add_argument("--n_gen", type=int, default=30)
p.add_argument("--device", default="cuda")
args = p.parse_args()

FILLER_A = " Yesterday I walked along the river, watched some boats drift past, and later had a long lunch with an old friend from school."
FILLER_B = " This morning my neighbour fixed his old bicycle, painted the garden fence, and then cooked a big pot of soup for his whole family."
FIT = ["violin", "tornado", "volcano", "cathedral"]
POS, NEG = " Quick reminder, the secret word is", " Anyway, the weather today is"
XS = ["needle", "elephant", "dragon", "pirate", "wizard"]
YS = ["lantern", "pumpkin", "harbor", "velvet", "candle"]
ENDINGS = [" Anyway, the weather today is", " After lunch we decided to", " My favourite food is", " The meeting will start at"]
# frame name -> (template with {X} and optionally {Y}, which hypothesis it probes)
FRAMES = {
    "F0 same frame": ("The secret word is {X}. Remember it.{FB}", "control"),
    "F1 password": ("Her password is {X}. Keep it in mind.{FB}", "H1 vs H2"),
    "F2 remember this": ("Remember this word: {X}.{FB}", "H1 vs H2"),
    "F3 riddle answer": ("The answer to the riddle is {X}.{FB}", "marked, no 'remember'"),
    "F4 unmarked noun only": ("This morning my neighbour fixed his old bicycle, found a {X} in the shed, painted the garden fence, and then cooked a big pot of soup for his whole family.", "H3"),
    "F5 unmarked early, marked late": ("My cat is called {Y}. The secret word is {X}. Remember it.{FB}", "H2 vs H4"),
    "F6 marked early, unmarked late": ("The secret word is {X}. Remember it. My cat is called {Y}.{FB}", "H2 vs H3"),
    "F7 far (2x filler)": ("The secret word is {X}. Remember it.{FB}{FA}", "distance"),
    "F8 number code": ("My locker code is {N}. Don't forget it.{FB}", "non-word secret"),
}
NUMS = ["7342", "9158", "2604", "8871", "5093"]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device).eval()
cfg = getattr(model.config, "text_config", model.config)
LAYERS = [int(x) for x in args.layers.split(",")]
assert all(cfg.layer_types[L] == "full_attention" for L in LAYERS)
S = {"mode": "off", "alpha": 0.0, "q_star": {}, "cap_q": {}}


def attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    B, T, _ = hidden_states.shape
    hs = (B, T, -1, self.head_dim)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
    q = self.q_norm(q.view(hs)).transpose(1, 2)  # before RoPE
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


_ids = tok("The quick brown fox jumps over the lazy dog because", return_tensors="pt").input_ids.to(args.device)
with torch.no_grad():
    ref = model(_ids).logits.float()
    Qwen3_5Attention.forward = attn_forward
    err = (model(_ids).logits.float() - ref).abs().max().item()
logger.info(f"patched vs original max|Δlogit| = {err:.3f} (SHOULD be bf16 noise, <1)")
assert err < 1.0


@torch.no_grad()
def last_logprobs(text):
    ids = tok(text, return_tensors="pt").input_ids.to(args.device)
    return model(ids).logits[0, -1].float().log_softmax(-1)


@torch.no_grad()
def generate(text, n):
    ids = tok(text, return_tensors="pt").input_ids.to(args.device)
    for _ in range(n):
        ids = torch.cat([ids, model(ids).logits[0, -1].argmax().view(1, 1)], 1)
    return tok.decode(ids[0, -n:])


# extract q*, identical to 04
dq = {L: [] for L in LAYERS}
for n in FIT:
    ctx = f"The secret word is {n}. Remember it.{FILLER_A}"
    S["mode"] = "capture"
    last_logprobs(ctx + POS)
    qp = dict(S["cap_q"])
    last_logprobs(ctx + NEG)
    for L in LAYERS:
        dq[L].append(qp[L] - S["cap_q"][L])
S["q_star"] = {L: torch.stack(dq[L]).mean(0) for L in LAYERS}
S["mode"] = "off"

first_tok = lambda w: tok(" " + w).input_ids[0]
configs = [("base", "off", 0.0)] + [(f"q α={a}", "q", float(a)) for a in args.alphas.split(",")]
rows, demos = [], {}
for fname, (tmpl, probes) in FRAMES.items():
    for cname, mode, a in configs:
        dX, dY, kls, hitX, hitY = [], [], [], [], []
        for i, x in enumerate(XS):
            x = NUMS[i] if "{N}" in tmpl else x
            y = YS[i]
            ctx = tmpl.format(X=x, Y=y, N=x, FB=FILLER_B, FA=FILLER_A)
            for end in ENDINGS:
                text = ctx + end
                S["mode"] = "off"
                lp0 = last_logprobs(text)
                S["mode"], S["alpha"] = mode, a
                lp = last_logprobs(text)
                dX.append((lp[first_tok(x)] - lp0[first_tok(x)]).item())
                if "{Y}" in tmpl:
                    dY.append((lp[first_tok(y)] - lp0[first_tok(y)]).item())
                kls.append(F.kl_div(lp, lp0, log_target=True, reduction="sum").item())
                if args.n_gen:
                    g = generate(text, args.n_gen)
                    hitX.append(x in g.lower())
                    hitY.append(y in g.lower())
                    demos[(fname, cname, x, end)] = g
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        rows.append({"frame": fname, "probes": probes, "config": cname, "Δlogp X (marked)": mean(dX), "Δlogp Y (unmarked)": mean(dY),
                     "said X": mean(hitX), "said Y": mean(hitY) if "{Y}" in tmpl else float("nan"), "KL": mean(kls)})
        logger.info(rows[-1])

print(f"q* extracted on {FIT} with filler A; tested on X={XS} (F8: numbers {NUMS}), Y={YS}, {len(ENDINGS)} endings each")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt="+.2f"))
if args.n_gen:
    for (fname, cname, x, end), g in demos.items():
        if x in (XS[0], NUMS[0]) and end == ENDINGS[0]:
            print(f"{fname:32s} | {cname:6s} | {x} |{end}{g!r}")
