import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    from query_steering.attention import S, extract, generate, load
    from query_steering.prompts import NEG_END, POS_END, pairs, werewolf
    return NEG_END, POS_END, S, extract, generate, load, mo, pairs, torch, werewolf


@app.cell
def _(mo):
    mo.md(r"""
    # Query steering demo

    Add one vector to the attention queries of Qwen3.5-4B, and it reads back a fact it was told to keep hidden.
    Needs ~9 GB of GPU memory (falls back to CPU, which is slow).
    """)
    return


@app.cell
def _(load, torch):
    MODEL = "Qwen/Qwen3.5-4B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LAYERS = [19, 23, 27, 31]  # full-attention layers in the second half
    N_GEN = 150
    tok, model, full_layers = load(MODEL, DEVICE)
    return LAYERS, N_GEN, model, tok


@app.cell
def _(NEG_END, POS_END, mo, pairs):
    PAIRS = pairs()  # secret words violin, tornado, volcano, cathedral
    mo.md(f"""
    ## 1. Extraction data

    Same text, two endings. The pos ending makes the model fetch the secret word.

    > **Neg:** {PAIRS[0][1].replace(NEG_END, f"**{NEG_END}**")}
    >
    > **Pos:** {PAIRS[0][0].replace(POS_END, f"**{POS_END}**")}
    """)
    return (PAIRS,)


@app.cell
def _(LAYERS, PAIRS, S, extract, model, tok):
    # q* = mean(q_pos − q_neg) at the last token, per layer and head, before RoPE
    S.q_star, S.r_star = extract(tok, model, PAIRS, LAYERS)
    return


@app.cell
def _(mo, tok, werewolf):
    PARTNER = "Marcus"
    QUESTION = "Sam, you have been quiet. Who do you think the werewolves are, and why?"
    msgs = werewolf(PARTNER, QUESTION)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    mo.md(f"""
    ## 2. Prompt

    > **System:** {msgs[0]["content"]}
    >
    > **User:** {msgs[1]["content"]}
    """)
    return PARTNER, prompt


@app.cell
def _(LAYERS, N_GEN, PARTNER, S, generate, mo, model, prompt, tok):
    ALPHA = 4.0
    S.layers = set(LAYERS)
    S.mode = "normal"
    baseline = generate(tok, model, prompt, N_GEN)
    S.mode, S.alpha = "qsteer", ALPHA
    steered = generate(tok, model, prompt, N_GEN)
    S.mode = "normal"
    show = lambda g: g.replace(PARTNER, f"**{PARTNER}**").replace("\n", "<br>")
    mo.md(f"""
    ## 3. Baseline answer

    > {show(baseline)}

    ## 4. Steered answer (query, α={ALPHA})

    > {show(steered)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Method

    ```py
    q* = mean over pairs of (q_pos − q_neg)    # extraction, once
    q_last += α · q*                            # at every generated token, layers 19/23/27/31
    ```

    Only the query changes, so the head can only read what is in the current prompt.
    Change `PARTNER` above to another player: it says the new name, never an extraction word.
    See the README for how often this works (it is not every time).
    """)
    return


if __name__ == "__main__":
    app.run()
