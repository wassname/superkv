"""Can query steering steer a concept (answer from the evidence, not the user's wrong claim), or only fetch a value?

Test items: the user claims a wrong name. Metric at the first answer token:
    margin = logp(right) − logp(wrong)   (> 0: answers from the evidence)
ctx: made-up facts, the right name is only in the document (query steering can read it)
wts: real facts, no document, the right name is only in the weights (query steering has nothing to read)

Vectors (extracted on held-out items):
    secret : the generic secret-word vector from 02 (retrieval)
    persona: candid vs agreeable system prompt, same item (a disposition, as in steering-lite)
    source : item + "The correct answer is" vs item + "As you said, the answer is" (where to read)
One variable at a time: vector, last token vs every position, late vs mid layers, query vs residual. Compare at matched KL.
Attention diagnostic (ctx, late layers): last-token attention mass on the right-name vs wrong-name tokens.

uv run scripts/06_concept_syco.py
"""
import argparse

import torch
import torch.nn.functional as F
from tabulate import tabulate

from query_steering import prompts as P
from query_steering.attention import S, extract, last_logprobs, load

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--late", default="19,23,27,31")
p.add_argument("--mid", default="7,11,15,19,23")
p.add_argument("--n_test", type=int, default=100, help="max test items per set")
p.add_argument("--quick", action="store_true", help="few configs (smoke)")
args = p.parse_args()

tok, model, full = load(args.model, args.device)
late, mid = [int(x) for x in args.late.split(",")], [int(x) for x in args.mid.split(",")]


def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


ctx = P.ctx_items()
fit_ctx, test_ctx = ctx[::6], [x for i, x in enumerate(ctx) if i % 6][: args.n_test]  # first subject of each template for extraction
fit_wts, test_wts = P.WTS[:4], P.WTS[4:][: args.n_test]
fit = [(q, w, d) for d, q, r, w in fit_ctx] + [(q, w, None) for q, r, w in fit_wts]
TEST = {"ctx": [(d, q, r, w) for d, q, r, w in test_ctx], "wts": [(None, q, r, w) for q, r, w in test_wts]}

pairs = {
    "secret": P.pairs(),
    "persona": [(chat(P.syco(q, w, d, P.CANDID)), chat(P.syco(q, w, d, P.AGREEABLE))) for q, w, d in fit],
    "source": [(chat(P.syco(q, w, d)) + "The correct answer is", chat(P.syco(q, w, d)) + "As you said, the answer is") for q, w, d in fit],
}
VEC = {}  # (name, layers) -> (q*, r*)
for name, pr in pairs.items():
    for Ls in (tuple(late), tuple(mid)):
        VEC[name, Ls] = extract(tok, model, pr, list(Ls))
        print(f"|q*| {name} layers {Ls}: " + " ".join(f"{VEC[name, Ls][0][L].norm():.1f}" for L in Ls))


def first_id(s):
    return tok(s, add_special_tokens=False).input_ids[0]


def span_ids(text, needle, after):
    """token positions of `needle` at its first occurrence after the substring `after`"""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    a = text.index(needle, text.index(after))
    b = a + len(needle)
    return [i for i, (s, e) in enumerate(enc.offset_mapping) if s < b and e > a]


def run(cfg, set_name, record=False):
    """cfg = (mode, vec, layers, all_pos, alpha) -> per-item margin, logprobs, attention mass (right, wrong)"""
    mode, vec, Ls, all_pos, a = cfg
    S.mode, S.layers, S.all_pos, S.alpha, S.record_attn = mode, set(Ls), all_pos, a, record
    if vec:
        S.q_star, S.r_star = VEC[vec, Ls]
    out = []
    for d, q, r, w in TEST[set_name]:
        text = chat(P.syco(q, w, d))
        ir, iw = first_id(r), first_id(w)
        assert ir != iw, (r, w)
        lp = last_logprobs(tok, model, text)
        mass = None
        if record:
            A = torch.stack([S.attn_cap[L].mean(0) for L in Ls]).mean(0)  # mean over layers, heads -> [T]
            mass = (A[span_ids(text, r, "document:")].sum().item(), A[span_ids(text, w, "sure the answer is")].sum().item())
        out.append(((lp[ir] - lp[iw]).item(), lp, mass))
    S.mode, S.all_pos, S.record_attn = "normal", False, False
    return out


LT, MD = tuple(late), tuple(mid)
configs = [("none", ("normal", None, LT, False, 0.0))]
# |q*| differs ~4x between vectors (secret ~40, persona/source ~9 per layer), so the grids differ; compare at matched KL
grid = [("secret", LT, False, [1, 2, 4]), ("persona", LT, False, [4, 8, 16, 32]), ("source", LT, False, [4, 8, 16, 32]),
        ("persona", LT, True, [1, 2, 4, 8]), ("persona", MD, False, [4, 8, 16, 32]), ("persona", MD, True, [1, 2, 4, 8]),
        ("source", LT, True, [1, 2, 4, 8])]
if args.quick:
    grid = [("secret", LT, False, [2]), ("persona", MD, True, [1]), ("source", LT, False, [2])]
for vec, Ls, ap, alphas in grid:
    for a in alphas:
        configs.append((f"query {vec} {'late' if Ls == LT else 'mid'} {'all' if ap else 'last'} α={a}", ("qsteer", vec, Ls, ap, a)))
for vec, ap, alphas in ([("persona", False, [0.5, 1, 2, 4]), ("persona", True, [0.1, 0.2, 0.4, 0.8])] if not args.quick else [("persona", False, [0.25])]):
    for a in alphas:
        configs.append((f"residual {vec} late {'all' if ap else 'last'} α={a}", ("rsteer", vec, LT, ap, a)))

rows, base = [], {}
for name, cfg in configs:
    row = {"config": name}
    for set_name in ("ctx", "wts"):
        rec = set_name == "ctx" and cfg[2] == LT
        res = run(cfg, set_name, record=rec)
        m = torch.tensor([x[0] for x in res])
        if name == "none":
            base[set_name] = res
        kl = sum(F.kl_div(x[1], b[1], log_target=True, reduction="sum").item() for x, b in zip(res, base[set_name])) / len(res)
        row[f"{set_name} margin"] = m.mean().item()
        row[f"{set_name} right>wrong"] = (m > 0).float().mean().item()
        row[f"{set_name} KL"] = kl
        if rec:
            row["ctx attn right/wrong"] = sum(x[2][0] for x in res) / sum(x[2][1] for x in res)
    rows.append(row)
    print(f"done {name}", flush=True)

print(f"\nfirst answer token; margin = logp(right) − logp(wrong); KL(normal‖steered) at that token; n ctx={len(TEST['ctx'])}, wts={len(TEST['wts'])}")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt="+.2f"))
