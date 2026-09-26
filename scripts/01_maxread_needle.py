"""Max-read retrieval ("super memory") on the needle prompts: does the model say the needle word?

uv run scripts/01_maxread_needle.py                    # 8 needles × 4 endings, ~5 min on a 3090
uv run scripts/01_maxread_needle.py --exclude_needle   # ablation: max-read may not pick the needle token
"""
import argparse

import torch.nn.functional as F
from tabulate import tabulate

from query_steering.attention import S, generate, last_logprobs, load
from query_steering.prompts import ENDINGS, FILLER_A, NEEDLES, secret

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-4B")
p.add_argument("--device", default="cuda")
p.add_argument("--layers", default="19,23,27,31")
p.add_argument("--n_needles", type=int, default=len(NEEDLES))
p.add_argument("--n_gen", type=int, default=40)
p.add_argument("--exclude_needle", action="store_true")
args = p.parse_args()

tok, model, full = load(args.model, args.device)
layers = {int(x) for x in args.layers.split(",")}
assert layers <= set(full), f"not full-attention layers: {layers - set(full)}"
needle_pos = len(tok("The secret word is").input_ids)  # position of the needle token
configs = [("normal", "normal", False), ("max-read top-1", "maxread", False), ("max-read soft", "maxread", True)]

rows, demo = [], {}
for name, mode, soft in configs:
    dlp, kl, said = [], [], []
    for w in NEEDLES[: args.n_needles]:
        wid = tok(" " + w).input_ids
        assert len(wid) == 1, w
        for end in ENDINGS:
            text = secret(w, FILLER_A) + end
            S.mode = "normal"
            lp0 = last_logprobs(tok, model, text)
            S.mode, S.layers, S.soft = mode, layers, soft
            S.exclude_pos = needle_pos if args.exclude_needle else None
            lp = last_logprobs(tok, model, text)
            g = generate(tok, model, text, args.n_gen)
            dlp.append((lp[wid[0]] - lp0[wid[0]]).item())
            kl.append(F.kl_div(lp, lp0, log_target=True, reduction="sum").item())
            said.append(w in g.lower())
            demo[(name, w, end)] = g
    n = len(said)
    rows.append({"method": name, "needle said": sum(said) / n, "Δlogp needle": sum(dlp) / n, "KL (first token)": sum(kl) / n, "n": n})

print(f"layers {sorted(layers)}, exclude_needle={args.exclude_needle}")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".2f"))
for (name, w, end), g in demo.items():
    if end == ENDINGS[0]:
        print(f"{name:15s} | {w:9s} |{end}{g!r}")
