# AGENTS.md

Novel ML research: not in your training data. Inherit global rules from `~/.pi/agent/AGENTS.md`.

- README.md is for human readers. Run notes, sweeps and reviews go in `slop/`.
- `scripts/01_needle_demo.py` is the experiment; `scripts/02_setup_figure.py` redraws the figure (CPU ok).
- The GPU is shared: 4B needs ~9 GB. Anything over a minute goes through pueue (`pueue` skill).
- Qwen3.5 is hybrid: `cfg.layer_types` lists full vs linear attention. Only full-attention layers are patched.
- `patched_forward` must match the original model when method=base; the script asserts this at start (max |Δlogit| < 1).
- Open question: does the effect survive an unprimed needle (no "secret word … Remember it")?
