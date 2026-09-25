Both files read. Here is my answer to (1)–(4).

**(1) Why meanQ/svdQ failed.** Three reasons, in likely order of importance.

- *Consensus ≠ union.* Averaging queries, then applying one softmax, finds what tokens collectively ask for. A needle requested by ~1 of 40 tokens contributes ~2.5% of the variance, so it vanishes from the mean and from top-4 singular vectors. The uniform control (−3.0) shows broad retrieval actively hurts; meanQ (−0.3 needle, **+0.4 filler**) is a weighted uniform — it retrieves the bulk, not the outlier. SVD of Q extracts majority intents (position/format queries), the opposite of a superset.
- *Pre-RoPE arithmetic is invalid for replay.* Scores depend on `q_t · R(t−s) k_s`. Rotating a summary query at T−1 assigns every intent the relative phase T−1−s, which no real query had; with θ=1e7 the lowest rotary pair turns ~1 rad/position, so far-key logits are geometrically unreproducible. Only the 192 unrotated dims carry position-free content — explaining why the failure is mild rather than catastrophic.
- *Dilution and confounds.* svdQ averages its R=4 retrievals, quartering even a correct direction. Also, meanQ/svdQ came from an 8-layer run vs. late-layers-only for the others; since early layers contribute ~0 this is roughly comparable but not clean. farA works because the max is taken **after** per-query softmax at each query's own position — the union happens in probability space.

**(2) Constructions** (all pool after per-query RoPE+softmax — that is the fix):

```python
# P1 outlier-topm: keep needle-intent queries
mu, Vh = Q[1:].mean(0), svd(Q[1:]).V[:4]
res = Q - (Q @ Vh.T) @ Vh                  # residual intents
idx = res.norm(-1).topk(m).indices
w_i = softmax(rope(Q[idx], pos=idx) @ K.T) # each at its OWN position
w = normalize(w_i.max(0)); o = w @ V

# P2 k-means mixture (svdQ generalized from subspace to clusters)
C, p = kmeans(Q[1:], k=12, return_mean_positions=True)
w = normalize(max_i softmax(rope(C[i], pos=p[i]) @ K.T)); o = w @ V

# P3 logsumexp union (smooth farA)
L_t = rope(q_t, pos=t) @ K.T * scale       # all earlier queries
w = softmax_keys(τ⁻¹ · logsumexp_t(τ·L_t)); o = w @ V

# P4 sharpness-weighted: keep low-entropy retrieval queries only
keep = topk(max_k softmax(L_t), m)
w = normalize(max over kept of softmax(L_t)); o = w @ V
```

**(3) Diagnostic.** Before touching V: per head, check each construction's weight vector `w` for needle-key weight/rank, and KL against the union baseline `max_t A[t,·]`. If meanQ's `w` already ranks the needle top-5 yet logits still drop, the bug is downstream (V-mixing, norm-match, gate), not in the query. Also report the fraction of heads where any far query row has the needle as argmax — a retrievability ceiling.

**(4) Evidence that would change my view.** Per-head inspection showing meanQ already retrieves the needle (→ my dilution story wrong); P1/P3 failing while A-based farA succeeds (→ information lives in A, not q·k replay, or my RoPE reasoning is wrong); svdQ at r=32 recovering the needle (→ rank too low, concept fine); the needle never being any far row's argmax (→ winner-take-all is picking up something else).

*Uncertainty: analysis is from brief+code only, no runs executed; RoPE claims assume the standard inv-freq formula for the 64-dim partial rotary.*