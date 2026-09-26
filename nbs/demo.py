import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    from superkv.attention import S, extract, generate, load
    from superkv.prompts import FILLER_A, FILLER_B, pairs, secret

    mo.md(
        """
        # superkv demo

        Two interventions on the attention of Qwen3.5-4B, at the last token only, in the full-attention layers 19, 23, 27, 31.

        1. **Max-read retrieval** ("super memory"): each head reads the one earlier token that later tokens looked back at hardest.
        2. **Query steering**: add a vector to the last token's query. The vector comes from contrast pairs; the head then reads the current prompt as usual.

        Needs ~9 GB of GPU memory (falls back to CPU, which is slow).
        """
    )
    return (
        FILLER_A,
        FILLER_B,
        S,
        extract,
        generate,
        load,
        mo,
        pairs,
        secret,
        torch,
    )


@app.cell
def _(load, torch):
    MODEL = "Qwen/Qwen3.5-4B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LAYERS = [19, 23, 27, 31]
    N_GEN = 30
    tok, model, full_layers = load(MODEL, DEVICE)
    return LAYERS, N_GEN, model, tok


@app.cell
def _(mo):
    mo.md("""
    ## 1. Max-read retrieval

    ```py
    score[s] = max over t ≥ s+4 of A[t, s]     # strongest long-range read token s got, per head
    o_last   = V[argmax score]                  # the last token reads that one token
    ```

    The prompt mentions the secret word once, then talks about something else.
    """)
    return


@app.cell
def _(FILLER_A, LAYERS, N_GEN, S, generate, mo, model, secret, tok):
    text1 = secret("needle", FILLER_A) + " Anyway, the weather today is"
    S.layers = set(LAYERS)
    S.mode = "normal"
    g_normal = generate(tok, model, text1, N_GEN)
    S.mode, S.soft = "maxread", False
    g_maxread = generate(tok, model, text1, N_GEN)
    S.mode = "normal"
    fmt = lambda g: g.replace("\n", "⏎").replace("<", "&lt;")
    mo.md(f"> {text1}\n\n| | continuation |\n|:--|:--|\n| normal | {fmt(g_normal)} |\n| max-read | {fmt(g_maxread)} |")
    return (fmt,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Query steering

    Extract once, on contrast pairs that differ only in the ending:

    - pos: "…Quick reminder, the secret word is" (the model fetches the secret word)
    - neg: "…Anyway, the weather today is" (it does not)

    ```py
    q* = mean(q_pos − q_neg)      # per layer and head, last token, before RoPE
    q_last += α · q*              # at test time, on new prompts
    ```

    The extraction words are violin, tornado, volcano, cathedral. The test prompt below uses a new word, a new story and an unrelated ending.
    Because only the query changes, the head can only read tokens in the current prompt: it says the new word, never an extraction word.
    Residual steering (`h_last += α · r*`, same pairs) is shown for comparison.
    """)
    return


@app.cell
def _(LAYERS, S, extract, model, pairs, tok):
    S.q_star, S.r_star = extract(tok, model, pairs(), LAYERS)
    return


@app.cell
def _(FILLER_B, LAYERS, N_GEN, S, fmt, generate, mo, model, secret, tok):
    SECRET = "red"
    Q_ALPHA, R_ALPHA = 4.0, 0.25
    text2 = secret(SECRET, FILLER_B) + " Anyway, the weather today is"
    S.layers = set(LAYERS)
    outs = {}
    for name, mode, a in [("normal", "normal", 0.0), (f"q-steer α={Q_ALPHA}", "qsteer", Q_ALPHA), (f"residual α={R_ALPHA}", "rsteer", R_ALPHA)]:
        S.mode, S.alpha = mode, a
        outs[name] = generate(tok, model, text2, N_GEN)
    S.mode = "normal"
    rows = "\n".join(f"| {k} | {fmt(v)} |" for k, v in outs.items())
    mo.md(f"> {text2}\n\n| | continuation |\n|:--|:--|\n{rows}")
    return


if __name__ == "__main__":
    app.run()
