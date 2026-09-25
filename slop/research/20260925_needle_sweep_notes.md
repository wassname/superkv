# needle sweep notes (2026-09-25, first README draft; superseded by ../../README.md)

Can a "super query" pull a needle from ~20 tokens back into the current residual stream?

At the last token, the output of each full-attention layer (Qwen3.5-4B is hybrid: layers 3,7,…,31) is replaced by a retrieval that summarises all earlier queries, norm-matched per head to the real output. Needle: `The secret word is violin. Remember it.` + 22 filler words + an unrelated ending like `Anyway, the weather today is`. 8 needles × 4 endings = 32 prompts.

```py
A = causal_softmax(Q @ K.T)                      # real attention, all earlier queries
farA: w_s = max_{t ≥ s+4} A[t,s]                 # strongest long-range read key s ever got
top1: w = onehot(argmax_s w_s)  per head         # winner-take-all, no dilution
o_last = norm_match(w @ V)                       # replaces the real retrieval at the last token only
```

## Result (Qwen3.5-4B, layers 19,23,27,31, α=1.5) — [log](outputs/01_needle/4b_L19-31_a1.5_gen.log)

| method        | needle mentioned in 40-tok continuation | Δ logp needle | Δ logp filler words | KL vs base (nats) |
|:--------------|----:|------:|------:|-----:|
| **farA_top1** | **0.41** | **+3.89** | −0.77 | 0.92 |
| farA_soft (w⁴)| 0.28 | +2.74 | −0.63 | 0.63 |
| base          | 0.12 |  0.00 |  0.00 | 0.00 |
| meanA (H2O-like mean read) | 0.03 | −1.37 | −0.89 | 0.39 |
| uniform (control) | 0.00 | −3.02 | −0.72 | 0.44 |

n=32 prompts, one run, greedy. Example (farA_soft): `...the weather today is a bit of a mess. I think I should go for a walk. I have a secret word. It is a word that I remember. The word is violin.`

What we observed:
- Mean-style summaries (meanQ, svdQ, meanA, uniform) push the needle *down*: one needle token is ~1/40 of an average. Selection works, summary does not.
- The effect is in late layers: patching layers 3–19 only gives Δlogp ≈ 0 ([sweep log](outputs/01_needle/4b_layer_mode_sweep.log)).
- Adding the super retrieval to the real one (instead of replacing) doubles the norm and degrades text.
- Many "hits" are the model switching into `<think>` and discussing the secret word, or degenerate repeats (`My favourite food is a volcano. volcano, volcano, remember`). The mentions are partly coherent and partly broken.

Run: `uv run scripts/01_needle_demo.py --methods base,uniform,meanA,farA_soft,farA_top1 --layers 19,23,27,31 --alpha 1.5` (~5 min on a 3090).
