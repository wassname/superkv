# smoke: every script on Qwen3.5-0.8B, CPU, tiny sizes (numbers are meaningless, checks the code runs)
smoke:
    uv run scripts/01_maxread_needle.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n_needles 1 --n_gen 3
    uv run scripts/02_qsteer.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n_test 1 --n_gen 3 --q_alphas 2 --r_alphas 0.25
    uv run scripts/03_qsteer_limits.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n_test 1 --n_gen 3 --alphas 2
    uv run scripts/05_secret_demos.py --model Qwen/Qwen3.5-0.8B --device cpu --layers 19,23 --n 1 --n_gen 3 --q_alphas 2

# the README numbers, Qwen3.5-4B on the GPU queue
reproduce:
    pueue add -w "$PWD" -l "query-steering: max-read needle demo" -- "uv run scripts/01_maxread_needle.py 2>&1 | tee outputs/01_maxread_needle.log"
    pueue add -w "$PWD" -l "query-steering: max-read, needle excluded (ablation)" -- "uv run scripts/01_maxread_needle.py --exclude_needle 2>&1 | tee outputs/01_maxread_needle_excluded.log"
    pueue add -w "$PWD" -l "query-steering: query vs residual steering" -- "uv run scripts/02_qsteer.py 2>&1 | tee outputs/02_qsteer.log"
    pueue add -w "$PWD" -l "query-steering: limits of the query steering vector" -- "uv run scripts/03_qsteer_limits.py 2>&1 | tee outputs/03_qsteer_limits.log"
    pueue add -w "$PWD" -l "query-steering: agent demos (hidden file, werewolf, eval awareness) + nulls" -- "uv run scripts/05_secret_demos.py 2>&1 | tee outputs/05_secret_demos.log"

figure:
    uv run scripts/04_figure.py

# demo notebook: edit live, or export to HTML headless
demo:
    uv run marimo edit nbs/demo.py
demo-html:
    uv run marimo export html nbs/demo.py -o outputs/demo.html
