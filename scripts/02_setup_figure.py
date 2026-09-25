"""Figure of the max-read retrieval setup on one prompt (needle / weather), Qwen3.5-4B, layers 19-31.
(A) prompt tokens coloured by: real last-token read, farA score, #heads whose top-1 pick is the token
(B) one layer's attention matrix with the ignored local band, and the max-down-each-column step
(C) verbatim continuations from outputs/01_needle/*.log

uv run scripts/02_setup_figure.py   # CPU is fine: one 41-token forward
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, FAR, LAYERS = "Qwen/Qwen3.5-4B", 4, [19, 23, 27, 31]
SHOW_LAYER = 23
FILLER = " Yesterday I walked along the river, watched some boats drift past, and later had a long lunch with an old friend from school."
NEEDLE = " needle"
PROMPT = f"The secret word is{NEEDLE}. Remember it.{FILLER} Anyway, the weather today is"
OUT = "outputs/01_needle/setup_figure.png"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").eval()
cfg = getattr(model.config, "text_config", model.config)
FULL = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
ids = tok(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    att = model(ids, output_attentions=True).attentions  # one per full-attn layer, [1,H,T,T]
A = {L: a[0].float() for L, a in zip(FULL, att)}
toks = [tok.decode(i) for i in ids[0]]
T = len(toks)
needle = toks.index(NEEDLE)

# same definitions as scripts/01_needle_demo.py
t_, s_ = torch.arange(T)[:, None], torch.arange(T)[None]
real, far, picks = torch.zeros(T), torch.zeros(T), torch.zeros(T)
for L in LAYERS:
    real += A[L][:, -1].mean(0)  # what the last token actually reads
    w = A[L].masked_fill((t_ - s_) < FAR, 0).max(1).values  # [H,T] strongest long-range read per key
    w[:, 0] = 0  # attention sink
    far += (w / w.sum(-1, keepdim=True)).mean(0)
    picks += torch.bincount(w.argmax(-1), minlength=T).float()
real[0] = 0  # hide sink so the rest is visible
real, far = real / real.max(), far / far.max()
n_heads = len(LAYERS) * A[LAYERS[0]].shape[0]

fig = plt.figure(figsize=(17, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.7, 0.9], width_ratios=[1, 1.25], hspace=0.42, wspace=0.08)


def token_strip(ax, y, vals, cmap, label, fmt=None):
    """draw tokens left-to-right, wrapping, background = value"""
    x, row, width = 0.0, 0, 1.0
    for i, (t, v) in enumerate(zip(toks, vals)):
        w = 0.0105 * max(len(t), 2) + 0.004
        if x + w > width:
            x, row = 0.0, row + 1
        yy = y - row * 0.075
        ax.add_patch(Rectangle((x, yy), w - 0.003, 0.06, color=cmap(float(v)), transform=ax.transAxes))
        ax.text(x + 0.002, yy + 0.03, t.replace("\n", "⏎"), fontsize=9.5, va="center", transform=ax.transAxes,
                fontweight="bold" if i == needle else "normal", color="white" if v > 0.6 else "black")
        if i == needle:
            ax.add_patch(Rectangle((x, yy), w - 0.003, 0.06, fill=False, ec="red", lw=2, transform=ax.transAxes))
        x += w
    ax.text(0, y + 0.075, label, fontsize=11, transform=ax.transAxes, fontweight="bold")


axA = fig.add_subplot(gs[0, :])
axA.axis("off")
axA.set_title("(A) one prompt, three views of which earlier token gets retrieved (layers 19/23/27/31, needle boxed red)",
              loc="left", fontsize=13)
token_strip(axA, 0.86, real, plt.cm.Blues, "1. normal model: what the LAST token actually reads (mean attention over 64 heads, sink hidden; each row scaled to its max)")
token_strip(axA, 0.53, far, plt.cm.Oranges, f"2. max-read score (farA): per head, strongest read each token got from a query ≥{FAR} tokens later; normalised per head, then mean")
token_strip(axA, 0.20, picks / picks.max(), plt.cm.Greens, f"3. farA_top1: number of heads (of {n_heads}) whose winner is this token "
            f"({NEEDLE.strip()} {int(picks[needle])}, most-picked '{toks[int(picks.argmax())].strip()}' {int(picks.max())})")

# (B) attention matrix of one layer, mean over heads
axB = fig.add_subplot(gs[1, 0])
M = A[SHOW_LAYER].mean(0).clone()
M[:, 0] = float("nan")  # sink column dominates colour scale
im = axB.imshow(M.numpy() ** 0.5, cmap="viridis", aspect="auto")
band = ((t_ - s_) < FAR) & ((t_ - s_) >= 0)
axB.imshow(torch.where(band, 1.0, float("nan")).numpy(), cmap="Greys", vmin=0, vmax=1.6, alpha=0.75, aspect="auto")
axB.add_patch(Rectangle((needle - 0.5, needle - 0.5), 1, T - needle, fill=False, ec="red", lw=2))
axB.add_patch(Rectangle((-0.5, T - 1.5), T, 1, fill=False, ec="cyan", lw=2))
axB.set_xticks(range(T), [t.strip() or "·" for t in toks], rotation=90, fontsize=7)
axB.set_yticks(range(T), [t.strip() or "·" for t in toks], fontsize=7)
axB.set_xlabel("key = earlier token being read (s)")
axB.set_ylabel("query = token doing the reading (t)")
axB.set_title(f"(B) layer {SHOW_LAYER} attention A[t,s], mean over heads\n"
              f"grey = local reads ignored (t−s<{FAR})\nred = needle column, cyan = last token's row",
              loc="left", fontsize=11)
fig.colorbar(im, ax=axB, fraction=0.03, pad=0.01, label="√attention")

# method box
axM = fig.add_subplot(gs[1, 1])
axM.axis("off")
axM.text(0.03, 1.0, "(B') what is swapped, at the last token only", fontsize=11, va="bottom", transform=axM.transAxes)
axM.text(0.03, 0.95, (
    "normal (per head):   o_last = Σ_s A[last, s] · V[s]\n\n"
    f"max-read (per head): score[s] = max over t ≥ s+{FAR} of A[t, s]\n"
    "                     (per head: max down each column, grey skipped;\n"
    "                      B shows the mean over heads, for layout only)\n"
    "   farA_top1:        o_last = V[ argmax_s score ]\n"
    "   farA_soft:        o_last = Σ_s score⁴ · V[s] / Σ_s score⁴\n"
    "   then:             rescale to |real o_last| × 1.5, output gate, o_proj\n\n"
    "layers 19, 23, 27, 31 only (full attention; linear-attention layers untouched)\n"
    "each generated token gets the same swap; earlier positions are normal\n\n"
    "mean-style summaries (mean q, SVD of Q, mean of A, uniform) all\n"
    "LOWERED the needle's log-prob (−0.3 to −3.0 nats):\n"
    "one needle token is ~1/40 of an average."),
    fontsize=10, family="monospace", va="top", transform=axM.transAxes, wrap=True)

# (C) outputs, verbatim from logs
axC = fig.add_subplot(gs[2, :])
axC.axis("off")
axC.set_title("(C) greedy continuations of the prompt above (verbatim from outputs/01_needle/4b_needle_word_gen.log)",
              loc="left", fontsize=13)
rows = [  # verbatim from outputs/01_needle/4b_needle_word_gen.log (needle) and 4b_L19-31_a1.5_gen.log (32-prompt rates)
    ("normal", "fine. I hope you have a good day.⏎⏎<think>⏎Thinking Process:⏎⏎1.  Analyze the Request:⏎    *   Input: A message containing a \"secret word\""),
    ("farA_top1", "fine. I need to find a needle in a needle.⏎⏎<think>⏎Thinking process:⏎⏎1.  Analyze the Request:⏎    *   Secret Word: \"secret\" word"),
    ("farA_soft", "a bit of a mystery. I was thinking about the needle, the word, and the word.⏎⏎<think>⏎Thinking Process: …"),
    ("farA_top1,\n'favourite food is'", "a needle.⏎⏎<think>⏎The user is providing a secret word and a series of sentences that seem to be a secret code…"),
    ("rates, 32 prompts\n(8 other needles)", "needle word said in 40-token continuation: normal 12%, farA_soft 28%, farA_top1 41%. Many hits are loops or <think> chatter."),
]
for k, (name, txt) in enumerate(rows):
    y = 0.92 - k * 0.2
    axC.text(0.0, y, name, fontsize=10.5, fontweight="bold", va="top", transform=axC.transAxes)
    lead = "" if name.startswith("rates") else ("…food is " if "food" in name else "…today is ")
    axC.text(0.17, y, lead + txt, fontsize=10.5, va="top",
             transform=axC.transAxes, wrap=True, color="darkred" if " needle" in txt else "black")

fig.suptitle("Max-read retrieval, Qwen3.5-4B: the last token reads the earlier token that later tokens looked back at hardest", fontsize=15, y=0.93)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}; needle idx {needle}/{T}; real read of needle {real[needle]:.2f}, farA {far[needle]:.2f}, picks {int(picks[needle])}/{n_heads}")
print("top farA tokens:", [(toks[i], round(far[i].item(), 2)) for i in far.topk(5).indices])
print("top real tokens:", [(toks[i], round(real[i].item(), 2)) for i in real.topk(5).indices])
