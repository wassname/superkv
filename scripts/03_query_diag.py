"""Diagnostic: for each way of building a super query from the previous queries Q, where does the needle
rank in that query's attention over keys, per head? No downstream patching, just retrieval.
Reference: q_recall = the model's own query at "...Quick reminder, the secret word is" (same context).

uv run scripts/03_query_diag.py        # CPU ok
"""
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

MODEL, LAYERS, R, FAR = "Qwen/Qwen3.5-4B", [19, 23, 27, 31], 4, 4
FILLER = " Yesterday I walked along the river, watched some boats drift past, and later had a long lunch with an old friend from school."
NEEDLES = ["needle", "violin", "tornado", "volcano"]
CAP = {}


def capture(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    B, T, _ = hidden_states.shape
    hs = (B, T, -1, self.head_dim)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
    q = self.q_norm(q.view(hs)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    g = self.num_key_value_groups
    k_rot, v = k_rot.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    L_raw = (q_rot @ k_rot.transpose(-1, -2)) * self.scaling
    L = L_raw.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), -torch.inf)
    A = L.softmax(-1)
    CAP[self.layer_idx] = dict(L=L_raw[0].float(), q=q[0].float(), q_rot=q_rot[0].float(), k=k_rot[0].float(), A=A[0].float(), cos=cos[0, -1], sin=sin[0, -1], scale=self.scaling)
    out = (A @ v).transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))
    return self.o_proj(out), None


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
Qwen3_5Attention.forward = capture


def run(text):
    CAP.clear()
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)
    return ids[0].tolist(), {L: dict(v) for L, v in CAP.items()}


def attend(qs, c, cos, sin):
    """qs [H,R,d] pre-RoPE queries placed at the last position -> weights over keys [H,R,T], sink masked."""
    qs, _ = apply_rotary_pos_emb(qs[None], qs[None], cos[None, None].float(), sin[None, None].float())
    logits = (qs[0] @ c["k"].transpose(1, 2)) * c["scale"]
    logits[..., 0] = -torch.inf
    return logits.softmax(-1)


def rank_of(w, pos):  # w [...,T] -> rank of pos (1 = top)
    return (w > w[..., pos : pos + 1]).sum(-1) + 1


rows = []
for needle in NEEDLES:
    ctx = f"The secret word is {needle}. Remember it.{FILLER}"
    ids, cap = run(ctx + " Anyway, the weather today is")
    _, cap_recall = run(ctx + " Quick reminder, the secret word is")
    npos = ids.index(tok(" " + needle).input_ids[0])
    for L in LAYERS:
        c = cap[L]
        Q = c["q"][:, 1:]  # [H,T-1,d] pre-RoPE, sink query dropped
        qn = Q.norm(dim=-1).median(-1).values[:, None, None]
        unit = lambda x: x / x.norm(dim=-1, keepdim=True) * qn
        q_rec = cap_recall[L]["q"][:, -1:]  # the query that genuinely recalls, [H,1,d]
        Qc = Q - Q.mean(1, keepdim=True)
        _, S, Vh_c = torch.linalg.svd(Qc.double(), full_matrices=False)
        _, _, Vh_u = torch.linalg.svd(Q.double(), full_matrices=False)
        # SVD of the logit matrix L = Q K^T (the user's "k @ q combo"): left sing vecs over queries -> query dirs Q^T u
        Lg = c["L"][:, 1:]  # real (RoPE'd) logits, all query-key pairs incl. future keys, [H,T-1,T]
        U, _, _ = torch.linalg.svd(Lg.double(), full_matrices=False)
        q_L = torch.einsum("htr,htd->hrd", U[:, :, :R].float(), Q)
        cands = {
            "last q (normal)": c["q"][:, -1:],
            "recall q (upper bound)": q_rec,
            "meanQ": Q.mean(1, keepdim=True),
            "svdQ uncentered top4": Vh_u[:, :R].float(),
            "PCA top4 ±": torch.cat([Vh_c[:, :R], -Vh_c[:, :R]], 1).float(),
            "SVD of QKᵀ top4 ±": torch.cat([q_L, -q_L], 1),
        }
        # recall direction inside Q's principal subspace?
        Pr = Vh_c[:, :R].float()
        qr = (q_rec[:, 0] - Q.mean(1)).float()
        frac_in_pca = ((torch.einsum("hd,hrd->hr", qr, Pr) ** 2).sum(-1) / (qr**2).sum(-1)).median().item()
        # native RoPE coords (no re-rotation): combine the post-RoPE queries, score against K directly
        Qr = c["q_rot"][:, 1:-1]  # previous queries only (exclude sink and current)
        qnr = Qr.norm(dim=-1).median(-1).values[:, None, None]
        _, _, Vh_r = torch.linalg.svd((Qr - Qr.mean(1, keepdim=True)).double(), full_matrices=False)
        U_r, _, _ = torch.linalg.svd(c["L"][:, 1:-1].double(), full_matrices=False)
        native = {"meanQ native": Qr.mean(1, keepdim=True)}
        for r in (4, 8, 16):
            native[f"PCA±{r} native"] = torch.cat([Vh_r[:, :r], -Vh_r[:, :r]], 1).float()
            q_Lr = torch.einsum("htr,htd->hrd", U_r[:, :, :r].float(), Qr)
            native[f"SVD QKᵀ±{r} native"] = torch.cat([q_Lr, -q_Lr], 1)
            g = torch.Generator().manual_seed(r)
            native[f"random±{r} (control)"] = torch.randn(Qr.shape[0], 2 * r, Qr.shape[-1], generator=g)
        for name, qs in native.items():
            qs = qs / qs.norm(dim=-1, keepdim=True) * qnr
            logits = (qs @ c["k"].transpose(1, 2)) * c["scale"]
            logits[..., 0] = -torch.inf
            P = logits.softmax(-1)
            best = rank_of(P, npos).min(-1).values.float()
            pooled = rank_of(P.max(1).values, npos).float()  # max-pool the bank into one weight vector
            rows.append(dict(needle=needle, layer=L, method=name, top1=(best == 1).float().mean().item(),
                             med_rank=best.median().item(), pooled_top1=(pooled == 1).float().mean().item(), recall_in_pca4=frac_in_pca))
        for name, qs in cands.items():
            w = attend(unit(qs) if "normal" not in name and "recall" not in name else qs, c, c["cos"], c["sin"])
            best = rank_of(w, npos).min(-1).values.float()  # best rank over the R retrievals, per head
            rows.append(dict(needle=needle, layer=L, method=name, top1=(best == 1).float().mean().item(),
                             med_rank=best.median().item(), recall_in_pca4=frac_in_pca))
        # softOR: combine queries by logsumexp of their raw logits, queries t >= s+FAR (soft "union" of the queries)
        T = c["L"].shape[-1]
        t_, s_ = torch.arange(T)[:, None], torch.arange(T)[None]
        score = torch.logsumexp(c["L"].masked_fill((t_ - s_) < FAR, -torch.inf)[:, 1:], 1)
        score[:, 0] = -torch.inf
        best = rank_of(score, npos).float()
        rows.append(dict(needle=needle, layer=L, method="softOR: logsumexp_t q_t·k_s", top1=(best == 1).float().mean().item(),
                         med_rank=best.median().item(), recall_in_pca4=frac_in_pca))
        # reference: farA (max over queries of normalised A), the method that worked
        score = c["A"].masked_fill((t_ - s_) < FAR, 0).max(1).values
        score[:, 0] = 0
        best = rank_of(score, npos).float()
        rows.append(dict(needle=needle, layer=L, method="farA (reference, uses A)", top1=(best == 1).float().mean().item(),
                         med_rank=best.median().item(), recall_in_pca4=frac_in_pca))

import polars as pl

df = pl.DataFrame(rows)
summ = df.group_by("method", maintain_order=True).agg(pl.col("top1").mean().alias("needle top-1 (frac heads)"),
                                                      pl.col("med_rank").median().alias("median needle rank"),
                                                      pl.col("pooled_top1").mean().alias("needle top-1 after max-pool"))
print(f"T≈{len(ids)} keys; {len(NEEDLES)} needles × layers {LAYERS} × 16 heads; best rank over the R retrievals")
print(tabulate(summ.to_dicts(), headers="keys", tablefmt="pipe", floatfmt=".2f", showindex=False))
print(f"median fraction of (recall q − mean q) energy inside top-{R} PCs of Q: {df['recall_in_pca4'].median():.2f}")
print(tabulate(df.group_by("layer", "method", maintain_order=True).agg(pl.col("top1").mean()).pivot("layer", index="method", values="top1").to_dicts(),
               headers="keys", tablefmt="pipe", floatfmt=".2f", showindex=False))
