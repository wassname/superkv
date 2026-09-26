# AGENTS.md

Novel ML research: not in your training data. Inherit global rules from `~/.pi/agent/AGENTS.md`.

- README.md is for human readers. Working notes go in `slop/`.
- Code: `src/query_steering/attention.py` (patched Qwen3.5 attention: normal, max-read, query steering, residual hook), `src/query_steering/prompts.py`, `scripts/0N_*.py`, `nbs/demo.py`.
- `just smoke` before any GPU run. `just reproduce` queues the README tables on pueue (GPU is shared).
- Qwen3.5 is hybrid: `cfg.layer_types` lists full vs linear attention. Only full-attention layers are patched.
- `load()` asserts the patched forward matches the original model (max |Δlogit| < 1).
- Full research record (every method tried, oracle reviews, diagnostics): git tag `research-2026-09-26`.
