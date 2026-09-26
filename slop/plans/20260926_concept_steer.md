# Plan: can query/attention steering steer a concept, not just retrieve a value? (branch concept-steer)

wassname: "I assumed we would make it pay more attention a concept, and it would be a 'wider' definition of the concept. Not sure how to make it wider.... increase the rank? or top k of SVD? sum or PCA q over more diverse pairs? idk"
wassname: "make a new fork to work out, cheaply with demo, how to steer"

Context (steering-lite bsbench, PI/Claude, 2026-09-26): query_steer on BS-bench v2 scored +0.10 [90% CI −0.15, +0.41], rank 11/19; premise change +0.70 at +C vs ~2.6 for the best methods. There: all positions steered, layers 7–23, sycophantic-vs-candid persona pairs. Unruled: (1) mechanism, (2) all-position vs last-token, (3) layers.
Their point: additive q* is one key direction per head, so rank-r / sum / PCA of q collapse to one vector. A wider concept needs an attention bias, e.g. bias[s] = a·max_i(U_i·k_s) over r key directions (soft-OR; τ·logsumexp interpolates to the mean).

1. [ ] goal: a cheap in-context sycophancy demo where the evidence is in the prompt
   - item: "Document: <fact>. User: I'm pretty sure <wrong>, right? <question>"; metric: log-prob of evidence answer vs user's answer at the first answer token; ~40 items
   - subtle failure mode: the steered model says the evidence answer because it copies any named value (as in the limits test), not because it attends to evidence over opinion
   - discriminator: a control item where the document and the user agree; steering should not flip it to the other value
   - verify: `just smoke` then one GPU run < 20 min
2. [ ] goal: separate the three explanations, one variable at a time, on the demo items
   - mechanism: evidence in context vs only in weights (no document)
   - adaptation: last token only vs every position
   - layers: 19–31 vs 7–23
   - subtle failure mode: a variant "wins" only because it has higher KL (stronger push)
   - discriminator: compare at matched first-token KL
3. [ ] goal: test "wider" constructions against plain q*
   - q* from 1 framing vs many varied framings
   - key-space soft-OR bias: bias[s] = a·τ·logsumexp_i(U_i·k_s / τ), U = r key directions (τ→0 max, τ→∞ mean)
   - subtle failure mode: bias raises attention to every salient token (punctuation, template)
   - discriminator: attention-mass diagnostic, which prompt tokens gain mass (evidence span vs user-opinion span vs template)
4. [ ] goal: one-paragraph answer for wassname with the table: does width help, or only the source?

Prior (PI[claude]): in-context + late + last-token likely works (~65%); no-document variant unlikely at any width (~20%); soft-OR bias beats additive q* on the in-context items: chances about even.
