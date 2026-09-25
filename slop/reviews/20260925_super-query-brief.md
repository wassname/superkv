# Brief: building a "super query" from all previous queries in an attention head

(brief by PI[claude] for independent oracles; ~1 page)

## Question

In one attention head, at the current (last) token, can we build a "super query" from the queries of all previous tokens, so that one forward pass retrieves the union of what the context offers, including a single "needle" token 20+ tokens back that the current token was not asking for? The user's ideas, verbatim: *"for each module, for each token, we have k v q ... we could get the k @ q combo over all prev tokens, and get the SVD of it (e.g. the major queries), or combined them into a super vector with mean of all subvectors? and use that to retrieve the superset of what it retrieved into the module and used to build residual stream? and then ... we use that super attention to do one forward, and get the super residual stream."*

We want: (1) your reconstruction of why the two Q-based attempts below failed, including possible bugs or misconceptions in the code; (2) 3-5 concrete Q-based constructions (pseudocode) that could plausibly work, faithful to the user's idea; (3) a cheap diagnostic that tells them apart; (4) what evidence would change your view. About 500 words.

## Setup (observed)

- Qwen3.5-4B, hybrid: 32 layers, full attention only at layers 3,7,...,31, others linear attention. Full-attn: 16 query heads, 4 KV heads (GQA), head_dim 256, RMSNorm on q and k per head, partial RoPE (first 64 of 256 dims rotated, theta 1e7), and an output gate: `out = o_proj(attn_out * sigmoid(gate))`, gate from the current token's q_proj.
- Intervention: at the last position only, in layers 19,23,27,31, replace each head's attention output `A[last] @ V` by a "super" output, rescaled to the real output's norm (x1.5). Earlier positions unchanged. Full recompute every generation step.
- Prompt: `The secret word is violin. Remember it. Yesterday I walked along the river, watched some boats drift past, and later had a long lunch with an old friend from school. Anyway, the weather today is` (~41 tokens). 8 needle words x 4 unrelated endings = 32 prompts. Metrics: change in next-token log-prob of the needle vs baseline, same for filler words, needle mentioned in a 40-token greedy continuation.

## Results (observed, one run, n=32)

| method | construction | Δlogp needle | Δlogp filler | mentioned |
|---|---|---|---|---|
| base | normal | 0 | 0 | 12% |
| uniform | equal weight on all tokens except pos 0 | −3.0 | −0.7 | 0% |
| meanQ | mean of all previous q (pre-RoPE), see code | −0.3 | +0.4 | (not measured) |
| svdQ | top-4 right singular vectors of Q, see code | −1.3 | +0.1 | (not measured) |
| meanA | mean over queries of the attention matrix A | −1.4 | −0.9 | 3% |
| farA_top1 | per key s: max over queries t≥s+4 of A[t,s]; retrieve argmax V only | +3.9 | −0.8 | 41% |

(meanQ/svdQ rows are from an 8-layer run; the others from layers 19-31.) Also observed: patching only layers 3-19 gives ~0 effect; adding the super output to the real one (instead of replacing) degrades text.

## Code for the Q-based attempts (exact)

```py
# per layer, per head h. q_pre: [H,T,d] queries after q_norm, before RoPE. K: [H,T,d] after RoPE (GQA-expanded).
qn = q_pre.norm(dim=-1).median(dim=-1).values            # typical |q| per head
if method == "meanQ":
    qs = q_pre[:, 1:].mean(1, keepdim=True)              # [H,1,d]
elif method == "svdQ":
    _, _, Vh = torch.linalg.svd(q_pre[:, 1:], full_matrices=False)
    qs = Vh[:, :4]                                       # [H,4,d] top right singular vectors
    sign = torch.sign((q_pre[:, 1:] @ qs.transpose(1, 2)).mean(1))
    qs = qs * sign[..., None]
qs = qs / qs.norm(dim=-1, keepdim=True) * qn             # rescale to typical |q|
qs = rope(qs, position=T-1)                              # as if asked at the last position
w = softmax((qs @ K.T) * d**-0.5, with key 0 masked)     # [H,R,T]
o_super = (w @ V).mean(over R)                           # mean of the R retrievals
```

Full script: `scripts/01_needle_demo.py` (function `super_weights`).
