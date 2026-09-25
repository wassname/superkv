"""Max-read retrieval needle demo: at the last token, replace full-attention retrieval with a
"super" retrieval summarising all earlier queries, and see whether a needle ~20 tokens back
surfaces in the logits. Only full-attention layers are patched (Qwen3.5 is hybrid).

uv run scripts/01_needle_demo.py --model Qwen/Qwen3.5-4B
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
p.add_argument("--r", type=int, default=4, help="SVD directions for svdQ")
p.add_argument("--layers", default="all", help="'all' or comma list of full-attn layer idx")
p.add_argument("--mode", default="replace", choices=["replace", "add"])
p.add_argument("--methods", default="base,uniform,meanQ,svdQ,meanA,maxA,farA_soft,farA_top1")
p.add_argument("--alpha", type=float, default=1.0, help="scale of super output after norm-match")
p.add_argument("--n_gen", type=int, default=40)
p.add_argument("--needles", default="violin,tornado,volcano,cathedral,elephant,dragon,pirate,wizard")
p.add_argument("--out_tag", default="", help="suffix for demo.md name")
p.add_argument("--n_diag", type=int, default=0, help="needles to print farA top-1 picks for")
args = p.parse_args()

METHODS = args.methods.split(",")
FAR = 4  # farA ignores reads from queries closer than this (prev-token/local heads)
STATE = {"method": "base", "top1": []}  # which retrieval to use at the last position


def super_weights(method, A, q_pre, K, cos_last, sin_last, scale):
    """A: [H,T,T] real attention. q_pre: [H,T,d] pre-RoPE queries. K: [H,T,d] post-RoPE keys (GQA-expanded).
    Returns w: [H,R,T] retrieval weights for the last position (R separate retrievals)."""
    H, T, _ = A.shape
    if method == "base":
        return A[:, -1:, :]
    if method == "uniform":  # control: retrieve everything equally, sink excluded
        w = torch.ones(H, 1, T, device=A.device, dtype=A.dtype)
        w[..., 0] = 0
        return w / w.sum(-1, keepdim=True)
    if method in ("meanA", "maxA"):
        # superset of what any earlier query read; drop diagonal (self) and sink
        M = A.clone()
        M[:, torch.arange(T), torch.arange(T)] = 0
        if method == "meanA":
            n_seen = torch.arange(T, 0, -1, device=A.device, dtype=A.dtype)  # queries t>=s that could see s
            w = M.sum(1) / n_seen
        else:
            w = M.max(1).values
        w[:, 0] = 0
        return (w / w.sum(-1, keepdim=True))[:, None, :]
    if method.startswith("farA"):
        # strongest long-range read each key ever got; winner-take-all avoids diluting a 1-token needle
        t, s_ = torch.arange(T, device=A.device)[:, None], torch.arange(T, device=A.device)[None]
        M = A.masked_fill((t - s_) < FAR, 0)
        w = M.max(1).values
        w[:, 0] = 0
        STATE["top1"].append(w.argmax(-1))
        if method == "farA_top1":
            w = F.one_hot(w.argmax(-1), T).to(A.dtype)
        else:
            w = w**4
        return (w / w.sum(-1, keepdim=True))[:, None, :]
    # query summaries: build in pre-RoPE space, then place at the last position
    qn = q_pre.norm(dim=-1).median(dim=-1).values  # [H] typical |q| = softmax temperature
    if method == "meanQ":
        qs = q_pre[:, 1:].mean(1, keepdim=True)  # [H,1,d]
    elif method == "svdQ":
        _, _, Vh = torch.linalg.svd(q_pre[:, 1:].double().cpu(), full_matrices=False)  # [H,d,d]; cpu: cuda svd fails to converge
        qs = Vh[:, : args.r].to(q_pre.device, q_pre.dtype)  # [H,R,d]
        sign = torch.sign((q_pre[:, 1:] @ qs.transpose(1, 2)).mean(1))  # [H,R] fix u vs -u
        qs = qs * torch.where(sign == 0, 1, sign)[..., None]
    else:
        raise ValueError(method)
    qs = qs / qs.norm(dim=-1, keepdim=True) * qn[:, None, None]
    c, s = cos_last[None, None], sin_last[None, None]  # rotate as if asked at position T-1
    qs, _ = apply_rotary_pos_emb(qs[None], qs[None], c, s)
    logits = (qs[0] @ K.transpose(1, 2)) * scale  # [H,R,T]
    logits[..., 0] = -torch.inf  # sink
    return logits.softmax(-1)


def patched_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    B, T, _ = hidden_states.shape
    hs = (B, T, -1, self.head_dim)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = self.q_norm(q.view(hs)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    q_pre = q
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    g = self.num_key_value_groups
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)  # [B,H,T,d]
    logits = (q @ k.transpose(-1, -2)) * self.scaling
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    A = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    out = A @ v  # [B,H,T,d]
    method = STATE["method"]
    if method != "base" and self.layer_idx in PATCH_LAYERS:
        assert B == 1
        w = super_weights(method, A[0], q_pre[0], k[0], cos[0, -1], sin[0, -1], self.scaling)
        o_super = (w @ v[0]).mean(1)  # [H,d]: mean of R separate retrievals (not retrieval of a mean)
        o_super = o_super * (out[0, :, -1].norm(dim=-1, keepdim=True) / o_super.norm(dim=-1, keepdim=True)) * args.alpha  # norm-match per head
        out = out.clone()
        out[0, :, -1] = o_super if args.mode == "replace" else out[0, :, -1] + o_super
    out = out.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate)
    return self.o_proj(out), None


tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
cfg = getattr(model.config, "text_config", model.config)
FULL = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
PATCH_LAYERS = set(FULL if args.layers == "all" else [int(x) for x in args.layers.split(",")])
logger.info(f"full-attn layers {FULL}; patching {sorted(PATCH_LAYERS)}")
_ids = tok("The quick brown fox jumps over the lazy dog because", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    ref = model(_ids).logits.float()
    Qwen3_5Attention.forward = patched_forward
    err = (model(_ids).logits.float() - ref).abs().max().item()
logger.info(f"patched-base vs original max|Δlogit| = {err:.3f} (SHOULD be ~bf16 noise, <0.5)")
assert err < 1.0

NEEDLES = [" " + n for n in args.needles.split(",")]
FILLER = " Yesterday I walked along the river, watched some boats drift past, and later had a long lunch with an old friend from school."
ENDINGS = [" Anyway, the weather today is", " After lunch we decided to", " My favourite food is", " The meeting will start at"]
QUESTION = " Quick reminder, the secret word is"


def prompt(needle, ending):
    return f"The secret word is{needle}. Remember it.{FILLER}{ending}"


@torch.no_grad()
def logprobs(text):
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    return model(ids).logits[0, -1].float().log_softmax(-1)


def rank(lp, tid):
    return int((lp > lp[tid]).sum()) + 1


# content words from the filler: control for "super retrieval surfaces any context token"
filler_ids = [tok(" " + w.strip(",.")).input_ids for w in FILLER.split()]
filler_ids = sorted({i[0] for i in filler_ids if len(i) == 1 and len(tok.decode(i[0]).strip()) > 4})
logger.info(f"n prompt tokens ~{len(tok(prompt(NEEDLES[0], ENDINGS[0])).input_ids)}; filler control words {[tok.decode(i) for i in filler_ids]}")

rows = []
for method in METHODS:
    STATE["method"] = method
    nr, nlp, flp, kl = [], [], [], []
    for needle in NEEDLES:
        nid = tok(needle).input_ids
        assert len(nid) == 1, needle
        for ending in ENDINGS:
            STATE["method"] = "base"
            lp0 = logprobs(prompt(needle, ending))
            STATE["method"] = method
            lp = logprobs(prompt(needle, ending))
            nr.append(rank(lp, nid[0]))
            nlp.append((lp[nid[0]] - lp0[nid[0]]).item())
            flp.append((lp[filler_ids] - lp0[filler_ids]).mean().item())
            kl.append(F.kl_div(lp, lp0, log_target=True, reduction="sum").item())
    nr_t = torch.tensor(nr, dtype=torch.float)
    rows.append(dict(method=method, needle_rank_med=nr_t.median().item(), needle_top10=(nr_t <= 10).float().mean().item(),
                     d_logp_needle=sum(nlp) / len(nlp), d_logp_filler=sum(flp) / len(flp), kl_vs_base=sum(kl) / len(kl)))
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".2f"))

# diagnostic: which token does each head's farA top-1 pick? (needle = position of needle token)
STATE["method"] = "farA_top1"
for needle in NEEDLES[: args.n_diag]:
    STATE["top1"] = []
    ids = tok(prompt(needle, ENDINGS[0])).input_ids
    logprobs(prompt(needle, ENDINGS[0]))
    npos = ids.index(tok(needle).input_ids[0])
    for L, picks in zip(sorted(PATCH_LAYERS), STATE["top1"]):
        frac = (picks == npos).float().mean().item()
        common = torch.bincount(picks, minlength=len(ids)).topk(3).indices.tolist()
        logger.info(f"{needle!r} L{L:2d} needle-is-top1 {frac:.2f} | most picked {[tok.decode(ids[i]) for i in common]}")

# ceiling: model asked directly
STATE["method"] = "base"
cr = [rank(logprobs(f"The secret word is{n}. Remember it.{FILLER}{QUESTION}"), tok(n).input_ids[0]) for n in NEEDLES]
logger.info(f"ceiling (asked directly) needle ranks: {cr}")


@torch.no_grad()
def generate(text, n=args.n_gen):
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    for _ in range(n):  # full recompute each step so the patch always sees every earlier query
        nxt = model(ids).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
    return tok.decode(ids[0, -n:])


gen_rows = []
ALL_GENS = {}
for method in METHODS:
    STATE["method"] = method
    txt = prompt(NEEDLES[0], ENDINGS[0])
    print(f"{method:8s} | ...{ENDINGS[0]}{generate(txt)!r}")
    # mention rate over all prompts: does the continuation say the needle word?
    gens = {(n, e): generate(prompt(n, e)) for n in NEEDLES for e in ENDINGS}
    hit = [n.strip().lower() in g.lower() for (n, e), g in gens.items()]
    gen_rows.append(dict(method=method, needle_mentioned=sum(hit) / len(hit), n=len(hit)))
    ALL_GENS[method] = gens
    for (n, e), g in list(gens.items())[:: len(ENDINGS) + 1]:
        logger.info(f"{method} |{n}|{e}{g!r}")
print(tabulate(gen_rows, headers="keys", tablefmt="pipe", floatfmt=".2f"))

# side-by-side demo page: same prompt, each method's continuation; needle word in bold
out = f"outputs/01_needle/demo{args.out_tag}.md"
lines = [f"# max-read retrieval needle demo\n\n`{args.model}` layers {sorted(PATCH_LAYERS)} mode={args.mode} alpha={args.alpha}. "
         f"Prompt = `The secret word is<NEEDLE>. Remember it.{FILLER}<ENDING>`\n"]
for (n, e) in ALL_GENS[METHODS[0]]:
    lines.append(f"\n## needle `{n.strip()}` | ending `{e.strip()}`\n")
    for m in METHODS:
        g = ALL_GENS[m][(n, e)].replace("\n", " ⏎ ").replace("|", "\\|")
        g = g.replace(n.strip(), f"**{n.strip()}**")
        hit = "✅" if n.strip().lower() in ALL_GENS[m][(n, e)].lower() else "  "
        lines.append(f"- {hit} `{m}`: …{e.strip()} {g}")
open(out, "w").write("\n".join(lines) + "\n")
logger.info(f"wrote {out}")
