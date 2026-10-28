"""Is Clinical UltraBERT's retrieval gain just better geometry? Re-score frozen embeddings after removing it.

  python analysis/whitening.py     # needs the embeddings cached by analysis/compute.py

For each model and retrieval task, recall@10 of relevant documents with the raw cosine, after centering
(subtracting the corpus mean, which removes the shared direction) and after PCA whitening (centering plus
rescaling the top 256 principal axes to unit variance, fitted on the corpus only). Writes
results/analysis/whitening.json.
"""

import json

import numpy as np

import compute as c

TASKS = ["SciFact", "NFCorpus", "MedicalQARetrieval"]
K_WHITEN = 256


def transforms(docs, queries):
    mu = docs.mean(0)
    yield "raw", docs, queries
    yield "centered", docs - mu, queries - mu
    u, s, vt = np.linalg.svd(docs - mu, full_matrices=False)
    w = vt[:K_WHITEN].T / (s[:K_WHITEN] / np.sqrt(len(docs)))
    yield "whitened", (docs - mu) @ w, (queries - mu) @ w


def recall10(docs, queries, rel_index):
    d = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    top = np.argsort(-(q @ d.T), axis=1)[:, :10]
    hits = [int(di in set(top[qi])) for qi, rel in enumerate(rel_index) for di in rel]
    return float(np.mean(hits))


if __name__ == "__main__":
    out = {m: {} for m in c.MODELS}
    for task in TASKS:
        docs, qs, qrels = c.retrieval_data(task)
        doc_ids, q_ids = list(docs), sorted(qrels)
        index = {i: k for k, i in enumerate(doc_ids)}
        rel_index = [[index[x] for x in qrels[q]] for q in q_ids]
        for m in c.MODELS:
            d = c.embed(m, f"{task}_corpus", [docs[i] for i in doc_ids], doc_ids)
            q = c.embed(m, f"{task}_queries", [qs[i] for i in q_ids], q_ids)
            out[m][task] = {name: recall10(dd, qq, rel_index) for name, dd, qq in transforms(d, q)}
            print(f"{m:24s} {task:20s} " + "  ".join(f"{k} {v:.3f}" for k, v in out[m][task].items()), flush=True)
    (c.OUT / "whitening.json").write_text(json.dumps(out, indent=1))
