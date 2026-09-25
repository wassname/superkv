# Research: why Q-based "super query" retrieval failed, and what could work

## Summary
The two Q-based attempts fail for statistical reasons, not bugs in the intervention machinery (I verified the shapes: `qs` [H,R,256], logits `[H,R,T]`, `mean(1)` combine — all consistent). A mean of queries points at the common-mode intent of a head — a near-uniform softmax over ~40 keys in which a single needle contributes ~1/40 of the output; SVD top right-singular vectors of an *uncentered* query matrix capture variance directions (mostly positional "which window do I read"), not the sparse reads that touch the needle. Attention output is a softmax *mixture*, not a *union*: one super query can never retrieve a union. Only per-key max statistics over queries (exactly what farA does, hence +3.9) preserve outlier reads. Fixes: summarize the k@q interaction matrix or cluster queries, combine by max (never mean), mask the near-diagonal, and patch only heads/layers where the needle is provably read.

## Findings

1. **No shape bug; the failure is the summarized statistic.** After per-head RMSNorm every query has equal norm (√256=16) — the *mean* of queries is the centroid direction of what the head asks about (mostly local next-token prediction). Softmaxed against 40+ post-RoPE keys it gives near-uniform weights, so the needle contributes ~1/40 of the output vector: below logit threshold (Δlogp −0.3, i.e. ~base). An average over queries mathematically buries the single query that read the needle. [brief table](slop/reviews/20260925_super-query-brief.md), [super_weights](scripts/01_needle_demo.py)

2. **svdQ captures variance, not targets.** On an uncentered 40×256 matrix the top right-singular vector ≈ the centroid (same as meanQ), and the next 3 directions span the largest across-position variation — for these heads that is positional/mechanical (which token window each position reads), not semantic. The sign-fix `sign((q_pre @ qsᵀ).mean(1))` actively biases toward the mean; R=4 of 256 dims loses fine alignment; `(w@V).mean(1)` over R retrievals is a mean-of-means dilution. In replace mode this perturbs the residual stream away from any weak needle signal the normal read had — consistent with it being *worse* than meanQ (−1.3). [super_weights](scripts/01_needle_demo.py)

3. **RoPE repositioning taxes every read with a long-range offset.** Summaries are built pre-RoPE then rotated as if asked at T−1. With partial RoPE (first 64 of 256 dims, theta 1e7) the high-frequency dims wrap multiple times at offset ~36, so the super query pays worst-case phase even for keys a nearby real query read cleanly (e.g. the needle's own local reads). This is a big reason A-based methods (already post-RoPE, position-free) dominate the Q-based ones. [patched_forward](scripts/01_needle_demo.py)

4. **Inherent limit: one query ⇒ one mixture.** `out = Σ softmax(q·k) v` reweights keys; it cannot emit multiple peaks on distant unrelated keys. "Union" needs R>1 queries or a per-key/per-retrieval **max**. farA_top1's +3.9 and 41% mention rate prove the per-key-max oracle reads *exist* in A (some head reads the needle at distance ≥4) — the needle is reachable, just not through mean/SVD-of-queries summaries. [table](slop/reviews/20260925_super-query-brief.md)

5. **Confounds worth stating.** meanQ/svdQ were measured on an 8-layer run while the A-methods used layers 19–31 (rows not directly comparable); the output gate still comes from the *un-asking* current token and can attenuate any super read before `o_proj`; replace mode destroys any weak baseline needle signal; "mentioned" is one greedy rollout. [brief](slop/reviews/20260925_super-query-brief.md)

## Proposals (Q-based, faithful to the user's idea)

**P1 – per-key max over query logits (the "k@q combo" itself):**
```py
L = (q_rope @ k.T) * scaling          # [H,T,T] real logits
L = causal_mask(L); L[...,0] = -inf
L = L.masked_fill((t - s) < FAR, -inf) # kill locality, like farA
w = softmax(L.max(dim=0).values / tau) # "did ANY query want key s?"
o_super = w @ v                       # one retrieval, no mean
```

**P2 – SVD of the k@q matrix (user's literal idea) in key-side space:**
```py
G = masked_far(q_rope @ k.T)[1:,1:]   # [H,T-1,T-1]
U, S, Vh = torch.linalg.svd(G)        # v_i = key-side interaction patterns
wkey = sum_{i<6} S[:,i,i] * Vh[:,i]   # combine principal reads
w = softmax(wkey / tau); o_super = w @ v[1:]
```

**P3 – cluster super queries (fixes "mean of subvectors"):**
```py
dirs = q_pre[:,1:] / q_pre[:,1:].norm(dim=-1,keepdim=True)
cent = kmeans(dirs, R=8)              # [H,R,d], pre-RoPE
cent = rope(cent, pos=T-1)
logitsR = cent @ k.T * scaling        # [H,R,T]
w = softmax(logitsR.max(dim=1).values / tau)  # max over R, NOT mean
o_super = w @ v
```

**P4 – GQA group-pooled SVD + selective patching:** SVD over the 4 same-group heads' 160 combined queries (they share one K,V via repeat_interleave; more samples ⇒ more stable major directions), R=8, same per-key-max combine; patch only heads where the diagnostic oracle says the needle is readable.

## Diagnostic (cheap: one prompt, no generation)
With a known needle at `npos`, per proposal compute `w` at the last token, then three scalars: `needle_w = mean_h w[h,npos]`; spread `H(w)/log T`; `corr(w, A.max(dim=0))` (oracle per-key max — already computed for farA; the natural recovery target). Plus needle-flip sensitivity: swap "violin"→"guitar", measure `||Δo_super||` in patched heads. Decisions: corr ≥ ~0.5 and needle_w ≫ 1/T ⇒ run the full benchmark; ||Δo|| ≈ 0 ⇒ proposal is needle-invariant, drop. Also use the oracle to list (head, layer) where the needle is a per-key-max top-1 and patch only those.

## Evidence that would change my view
- If `max_t L[t, npos]` is at noise level **yet** a Q-based method lifts the needle ⇒ refutes "needle unreachable from Q-space".
- If meanQ re-run on layers 19–31 recovers the needle ⇒ failure is construction/layer-choice, not statistic-class.
- If a *single-query* method reproduces the full oracle per-key-max pattern on held-out prompts (corr ≥ 0.7, Δlogp needle ≥ +3, filler ≈ 0) ⇒ "one query ⇒ mixture, not union" is wrong for these heads (mathematically implausible unless temperature→0 with near-ties — I'd update but expect not to).
- If SVD-of-logits (P2) works while per-key max (P1) fails ⇒ needle reads are structured patterns, not isolated sparse outliers.

## Sources
- Kept: `slop/reviews/20260925_super-query-brief.md` — question, setup, results table, exact meanQ/svdQ code.
- Kept: `scripts/01_needle_demo.py` — `super_weights`, `patched_forward`, farA logic, diagnostics.
- Dropped: none (no web search needed by design).

## Gaps
No per-head/per-layer attribution of where the needle is actually read (the oracle diagnostic hasn't been run meaningfully — only the `most picked` print exists); single observed run, n=32, no seeds; meanQ/svdQ only on an 8-layer config; no KL/rank data for the Q methods; effect of gate attenuation (could zero the gate and re-run farA to bound it) untested. All mechanism claims above are inferred from attention statistics, not verified per head.