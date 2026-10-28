"""Latent-space analysis of Clinical UltraBERT vs. its base model and the best medical MLM encoder.

  python analysis/compute.py          # encodes what is missing, writes results/analysis/stats.json

Embeddings of public data are cached in data/analysis/<model>/<name>.npy. MIMIC-IV embeddings are only read
from data/ehr_meds/ (cached by benchmark/ehr_meds.py) and only aggregate statistics leave this script.

Analyses
  geometry     anisotropy (mean cosine of random pairs), effective rank, variance of the top PCs
  retrieval    cosine of each query to its relevant vs. random documents (SciFact, NFCorpus, MedicalQA)
  lexical      recall@10 of relevant documents binned by query-document word overlap
  topics       silhouette and training-free kNN accuracy of disease / specialty labels; t-SNE coordinates
  role_topic   PubMed RCT sentences: similarity by shared abstract (topic) vs. shared rhetorical role
  synonyms     abbreviation / lay-term / brand-name -> canonical term matching
  ehr          geometry of serialized MIMIC-IV records and training-free kNN AUROC
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
import config  # noqa: E402  (HF caches -> data/)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score, silhouette_score  # noqa: E402

MODELS = {"Clinical UltraBERT": "Clinical-UltraBERT", "UltraBERT": "UltraBERT-150M",
          "BioClinical ModernBERT": "BioClinical-ModernBERT"}
EMB = config.DATA_DIR / "analysis"
OUT = config.RESULTS_DIR.parent / "analysis"
SEED = 0
_encoders = {}


def encoder(name):
    from run_benchmark import load_encoder

    if name not in _encoders:
        _encoders.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _encoders[name] = load_encoder(config.MODELS[MODELS[name]], "mps" if torch.backends.mps.is_available() else "cpu")
    return _encoders[name]


def embed(name, key, texts, ids=None):
    """L2-normalized embeddings, cached per model and key. With ids, the cache also stores the id order and is
    rebuilt when it does not match (dict orders can differ between runs)."""
    path = EMB / MODELS[name] / f"{key}.npy"
    id_path = path.with_suffix(".ids.json")
    stale = ids is not None and (not id_path.exists() or json.loads(id_path.read_text()) != list(ids))
    if not path.exists() or stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, encoder(name).encode(list(texts), batch_size=32, show_progress_bar=False))
        if ids is not None:
            id_path.write_text(json.dumps(list(ids)))
    x = np.load(path).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cached(name, path):
    x = np.load(path).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------------- geometry

def geometry(x, rng, n_pairs=20000):
    i, j = rng.integers(0, len(x), n_pairs), rng.integers(0, len(x), n_pairs)
    keep = i != j
    cos = np.sum(x[i[keep]] * x[j[keep]], axis=1)
    centered = x - x.mean(0)
    s = np.linalg.svd(centered, compute_uv=False) ** 2
    p = s / s.sum()
    return {
        "random_cos_mean": float(cos.mean()), "random_cos_std": float(cos.std()),
        "random_cos_sample": np.round(cos[:4000], 4).tolist(),
        "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
        "pc1_var": float(p[0]), "pcs_for_90": int(np.searchsorted(np.cumsum(p), 0.9) + 1),
        "cum_var": np.round(np.cumsum(p)[:200], 4).tolist(),
        "mean_vector_norm": float(np.linalg.norm(x.mean(0))),  # ~1 when every embedding points the same way
    }


# ---------------------------------------------------------------------------------------------------- retrieval data

def retrieval_data(task_name):
    task = config.get_tasks([task_name])[0]
    task.load_data()
    split_name = task.metadata.eval_splits[0]
    if task.dataset is not None:
        split = next(iter(task.dataset.values()))[split_name]
        corpus, queries, qrels = split["corpus"], split["queries"], split["relevant_docs"]
        titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus)
        docs = {i: f"{t} {x}".strip() for i, t, x in zip(corpus["id"], titles, corpus["text"])}
        qs = dict(zip(queries["id"], queries["text"]))
    else:
        pick = lambda x: x.get(task.hf_subsets[0], x)[split_name]
        docs = {i: d if isinstance(d, str) else f"{d.get('title', '')} {d['text']}".strip()
                for i, d in pick(task.corpus).items()}
        qs, qrels = pick(task.queries), pick(task.relevant_docs)
    qrels = {q: {d for d, s in rel.items() if s > 0 and d in docs} for q, rel in qrels.items() if q in qs}
    qrels = {q: r for q, r in qrels.items() if r}
    return docs, qs, qrels


STOP = set("""a an the of in on for to and or with without by from at as is are was were be been being this that these
those it its into than then there their they we our you your what which who whom whose when where why how does do did
not no can could should would may might will shall has have had about after before between during over under more most
other some such only own same so too very s t just also any all each few both""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def retrieval(name, task_name, docs, qs, qrels, rng):
    doc_ids, q_ids = list(docs), sorted(qrels)
    d = embed(name, f"{task_name}_corpus", [docs[i] for i in doc_ids], doc_ids)
    q = embed(name, f"{task_name}_queries", [qs[i] for i in q_ids], q_ids)
    index = {i: k for k, i in enumerate(doc_ids)}
    sims = q @ d.T
    rel_cos, rand_cos, rows = [], [], []
    for qi, qid in enumerate(q_ids):
        rel = [index[x] for x in qrels[qid]]
        order = np.argsort(-sims[qi])
        rank = {doc: r for r, doc in enumerate(order[:1000])}
        rand = rng.integers(0, len(doc_ids), 20)
        rel_cos += sims[qi, rel].tolist()
        rand_cos += sims[qi, rand].tolist()
        qw = content_words(qs[qid])
        for di in rel:
            overlap = len(qw & content_words(docs[doc_ids[di]])) / max(1, len(qw))
            rows.append({"query": qid, "overlap": overlap, "hit10": int(rank.get(di, 10**6) < 10)})
    rel_cos, rand_cos = np.array(rel_cos), np.array(rand_cos)
    pooled = np.sqrt((rel_cos.var() + rand_cos.var()) / 2)
    return {
        "rel_cos": np.round(rel_cos, 4).tolist(), "rand_cos": np.round(rand_cos, 4).tolist(),
        "d_prime": float((rel_cos.mean() - rand_cos.mean()) / pooled),
        "pairs": rows,
    }, d


# ---------------------------------------------------------------------------------------------------- topics

def knn_accuracy(x, y, k=10):
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    pred = [pd.Series(y[row]).mode().iloc[0] for row in nn]
    return float(np.mean(np.array(pred) == y))


def topics(name, dataset, x, y, tsne_classes=None, rng=None):
    out = {"silhouette": float(silhouette_score(x, y, metric="cosine", sample_size=min(3000, len(x)), random_state=SEED)),
           "knn10_acc": knn_accuracy(x, y)}
    if tsne_classes:
        keep = np.isin(y, tsne_classes)
        idx = np.flatnonzero(keep)
        idx = rng.choice(idx, size=min(900, len(idx)), replace=False)
        xy = TSNE(2, metric="cosine", init="pca", perplexity=30, random_state=SEED).fit_transform(x[idx])
        out["tsne"] = {"xy": np.round(xy, 3).tolist(), "label": y[idx].tolist()}
    return out


# ---------------------------------------------------------------------------------------------------- role vs. topic

def role_topic_data(n_abstracts=250):
    from datasets import load_dataset

    df = load_dataset("armanc/pubmed-rct20k", split="test").to_pandas()
    ids = sorted(df["abstract_id"].unique())
    keep = random.Random(SEED).sample(ids, n_abstracts)
    return df[df["abstract_id"].isin(keep)].reset_index(drop=True)


def role_topic(name, df):
    x = embed(name, "pubmedrct_role_topic", df["text"])
    sims = x @ x.T
    a, r = df["abstract_id"].to_numpy(), df["label"].to_numpy()
    same_a, same_r = a[:, None] == a[None, :], r[:, None] == r[None, :]
    off = ~np.eye(len(df), dtype=bool)
    groups = {
        "same_abstract_diff_role": same_a & ~same_r & off,
        "diff_abstract_same_role": ~same_a & same_r,
        "diff_abstract_diff_role": ~same_a & ~same_r,
    }
    base = sims[groups["diff_abstract_diff_role"]]
    mu, sd = base.mean(), base.std()
    out = {k: float(sims[m].mean()) for k, m in groups.items()}
    out["topic_effect_z"] = float((out["same_abstract_diff_role"] - mu) / sd)
    out["role_effect_z"] = float((out["diff_abstract_same_role"] - mu) / sd)
    # Training-free role kNN that may not use sentences of the same abstract.
    s = sims.copy()
    s[same_a] = -np.inf
    nn = np.argsort(-s, axis=1)[:, :10]
    pred = [pd.Series(r[row]).mode().iloc[0] for row in nn]
    out["role_knn10_acc_other_abstracts"] = float(np.mean(np.array(pred) == r))
    out["role_majority"] = float(pd.Series(r).value_counts(normalize=True).iloc[0])
    return out


# ---------------------------------------------------------------------------------------------------- synonyms

SYNONYMS = {
    "abbreviation": [
        ("MI", "myocardial infarction"), ("CHF", "congestive heart failure"),
        ("COPD", "chronic obstructive pulmonary disease"), ("DVT", "deep vein thrombosis"),
        ("PE", "pulmonary embolism"), ("CKD", "chronic kidney disease"), ("GERD", "gastroesophageal reflux disease"),
        ("UTI", "urinary tract infection"), ("HTN", "hypertension"), ("AFib", "atrial fibrillation"),
        ("TIA", "transient ischemic attack"), ("BPH", "benign prostatic hyperplasia"),
        ("ARDS", "acute respiratory distress syndrome"), ("CABG", "coronary artery bypass grafting"),
        ("DKA", "diabetic ketoacidosis"), ("SOB", "shortness of breath"),
    ],
    "lay term": [
        ("heart attack", "myocardial infarction"), ("stroke", "cerebrovascular accident"),
        ("high blood pressure", "hypertension"), ("low blood sugar", "hypoglycemia"),
        ("kidney stones", "nephrolithiasis"), ("nosebleed", "epistaxis"), ("fainting", "syncope"),
        ("heartburn", "gastroesophageal reflux"), ("chickenpox", "varicella"), ("hives", "urticaria"),
        ("pink eye", "conjunctivitis"), ("bedsore", "pressure ulcer"),
    ],
    "brand name": [
        ("Lipitor", "atorvastatin"), ("Tylenol", "acetaminophen"), ("Advil", "ibuprofen"),
        ("Zoloft", "sertraline"), ("Coumadin", "warfarin"), ("Lasix", "furosemide"),
        ("Glucophage", "metformin"), ("Norvasc", "amlodipine"), ("Prilosec", "omeprazole"),
        ("Synthroid", "levothyroxine"), ("Plavix", "clopidogrel"), ("Xanax", "alprazolam"),
    ],
}


def synonyms(name):
    out = {}
    for kind, pairs in SYNONYMS.items():
        src = embed(name, f"syn_{kind}_src", [p[0] for p in pairs])
        tgt = embed(name, f"syn_{kind}_tgt", [p[1] for p in pairs])
        sims = src @ tgt.T
        ranks = (sims > np.diag(sims)[:, None]).sum(1)  # 0 = correct target ranked first
        out[kind] = {"acc@1": float(np.mean(ranks == 0)), "mrr": float(np.mean(1 / (ranks + 1))), "n": len(pairs)}
    return out


# ---------------------------------------------------------------------------------------------------- EHR

def ehr(name, rng):
    out = {}
    for task in ("meds_mortality", "meds_long_los", "meds_readmission_30d"):
        base = config.DATA_DIR / "ehr_meds"
        x = cached(name, base / "embeddings" / MODELS[name] / f"{task}_test.npy")
        y = np.array([json.loads(line)["label"] for line in (base / "examples" / f"{task}_test.jsonl").open()], dtype=int)
        g = geometry(x, rng)
        sims = x @ x.T
        np.fill_diagonal(sims, -np.inf)
        nn = np.argsort(-sims, axis=1)[:, :25]
        knn_score = y[nn].mean(1)  # share of positive neighbours, a training-free risk score
        out[task] = {"random_cos_mean": g["random_cos_mean"], "effective_rank": g["effective_rank"],
                     "pc1_var": g["pc1_var"], "knn25_auroc": float(roc_auc_score(y, knn_score))}
    return out


# ---------------------------------------------------------------------------------------------------- main

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {m: {} for m in MODELS}

    retrieval_tasks = ["SciFact", "NFCorpus", "MedicalQARetrieval"]
    data = {t: retrieval_data(t) for t in retrieval_tasks}
    role_df = role_topic_data()
    clin = {n: {s: pd.read_parquet(config.CLINICAL_CLS_DIR / n / f"{s}.parquet") for s in ("test",)}
            for n in ("MedicalAbstracts", "MTSamples")}
    tsne_classes = ["neoplasms", "digestive system diseases", "cardiovascular diseases"]

    for name in MODELS:  # one model at a time so only one encoder is loaded
        print(f"=== {name}", flush=True)
        s = stats[name]
        s["retrieval"], corpora = {}, {}
        for t in retrieval_tasks:
            s["retrieval"][t], corpora[t] = retrieval(name, t, *data[t], rng)
            print(f"  retrieval {t}: d'={s['retrieval'][t]['d_prime']:.2f}", flush=True)
        s["geometry"] = geometry(corpora["SciFact"], rng)
        print(f"  geometry: mean cos {s['geometry']['random_cos_mean']:.3f}, eff. rank {s['geometry']['effective_rank']:.1f}", flush=True)
        s["topics"] = {}
        for n, frames in clin.items():
            x = cached(name, config.RESULTS_DIR / "embeddings" / MODELS[name] / f"{n}_test.npy")
            y = frames["test"]["label"].to_numpy()
            s["topics"][n] = topics(name, n, x, y, tsne_classes if n == "MedicalAbstracts" else None, rng)
            print(f"  topics {n}: silhouette {s['topics'][n]['silhouette']:.3f}, kNN {s['topics'][n]['knn10_acc']:.3f}", flush=True)
        s["role_topic"] = role_topic(name, role_df)
        print(f"  role/topic: {s['role_topic']['topic_effect_z']:.2f} / {s['role_topic']['role_effect_z']:.2f}", flush=True)
        s["synonyms"] = synonyms(name)
        print(f"  synonyms: {s['synonyms']}", flush=True)
        s["ehr"] = ehr(name, rng)
        print(f"  ehr: {s['ehr']}", flush=True)

    (OUT / "stats.json").write_text(json.dumps(stats))
    print(f"wrote {OUT / 'stats.json'}")
