## (1) Why these attempts plausibly failed

**A single query generally cannot represent a union of retrievals.** Opposing queries can retrieve different keys yet average to zero; averaging their softmax outputs instead dilutes each retrieval. Neither operation preserves arbitrary sets of values in one head-sized vector.

In `scripts/01_needle_demo.py`, `"q_pre[:, 1:].mean(1, keepdim=True)"` includes the **current** query, contrary to “all previous queries.” More importantly, historical queries are repositioned using `"cos[0, -1], sin[0, -1]"`. This changes their relative-position scores against historical keys. Whether that explains much here is uncertain: only part of each vector rotates.

SVD maximizes query reconstruction energy, not needle retrieval. Uncentered dominant directions may describe common features rather than rare queries. Choosing one sign discards the opposite retrieval; assigning equal norms and averaging outputs discards singular-value weighting without establishing a better alternative. Normalizing a nearly cancelled mean can amplify an unrepresentative direction.

The actual script fixes two apparent problems in the brief’s abbreviated code: broadcasting uses `"qn[:, None, None]"`, and zero signs are replaced with one. Zero-norm divisions remain unguarded.

Finally, current-token gating may suppress retrieved features, and replacement removes useful ordinary attention. The brief explicitly says Q methods used an “8-layer run”; therefore their scores are not controlled comparisons with the later-layer alternatives. These are competing explanations, not demonstrated causes.

## (2) Four Q-based constructions

Per head, let `Q[t]` be historical queries **after their original RoPE**, excluding the current position. All proposals need safe normalization and identical layer/scale settings.

**A. Historical-query union pooling**
```text
P[t,s] = softmax_s(Q[t] @ K[s] / sqrt(d), historical_causal_mask)
u[s] = max_{t >= s+4} P[t,s]
w = normalize(mask_sink(u)^beta)
output = w @ V
```
This reconstructs the successful far-attention idea directly from Q/K. Finite `beta` tests whether several retrieved keys survive.

**B. Diverse historical-query representatives**
```text
P = historical_attention(Q, K)
I = greedy_farthest_rows(P, budget=R)
u = max_rows(P[I])
output = normalize(mask_sink(u)^beta) @ V
```
Selecting by retrieval diversity avoids spending every representative on the same common query pattern.

**C. Signed SVD query bank**
```text
D = right_singular_vectors(Q)[:R]
bank = typical_norm * concatenate(D, -D)
P = softmax(bank @ K.T / sqrt(d), sink_mask)
output = normalize(max_rows(P)^beta) @ V
```
Use already-rotated coordinates; do not rotate this bank again. This tests sign loss and averaging dilution, though synthetic directions remain potentially unnatural.

**D. Distill a union into one query**
```text
target = stop_gradient(weights_from_A)
q = initialize_from_best_historical_query()
repeat small_number_of_steps:
    q -= lr * gradient_q(KL(target || softmax(q @ K.T / sqrt(d))))
output = softmax(q @ K.T / sqrt(d)) @ V
```
This directly tests the single-vector constraint; it adds optimization, not merely one ordinary attention evaluation.

## (3) Cheap distinguishing diagnostic

Cache Q/K/V from one baseline prompt; compare methods offline before generation. Record needle attention mass, entropy, top keys, representative overlap, and single-query distillation error. Then patch one matched layer and measure needle Δlogp plus KL against baseline.

Predictions: native-position replay beating repositioning supports positional mismatch; signed directions beating one-sided SVD supports sign loss; max pooling beating averaging supports dilution. Strong retrieval without logit improvement points downstream, not necessarily to bad queries.

## (4) Evidence that would change my view

Matched-layer, matched-scale improvements across held-out **unprimed**, displaced, and multiple-needle prompts would strengthen the union-retrieval interpretation. Random-key and shuffled-query controls matching performance would weaken it. Removing attention to the needle while retaining the gain would weaken direct-retrieval claims.

No execution logs were supplied, so I cannot verify numerical fidelity, uncertainty across prompts, or whether `farA_top1` actually selects needles rather than instruction-associated positions.