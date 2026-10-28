"""95% confidence intervals for the non-retrieval benchmark scores (retrieval is reported without intervals).

  python benchmark/confidence.py

Writes results/baselines/ci/<model>/<task>.json = {"score", "lo", "hi", "method"} on the same scale as the
benchmark (0-1). Methods:
  bootstrap   1,000 resamples of the test examples or sentence pairs, percentile interval
  folds       t-interval of the mean over MTEB's 5 cross-validation folds (SciRepEval)
  subsets     t-interval of the mean over MTEB's resampled clustering subsets
"""

import argparse
import json
import math
import sys

import config

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import mteb
from clinical_cls import DATASETS as CLINICAL_CLS, SEED

OUT = config.RESULTS_DIR / "ci"
N_BOOT = 1000


def write(model, task, score, lo, hi, method):
    path = OUT / model / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"score": float(score), "lo": float(lo), "hi": float(hi), "method": method}))


def bootstrap(metric, n, seed=0):
    rng = np.random.default_rng(seed)
    vals = [metric(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    vals = [v for v in vals if not math.isnan(v)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def t_interval(values):
    v = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def mteb_result(model, task):
    cache = mteb.ResultCache(cache_path=str(config.RESULTS_DIR))
    return cache.load_task_result(task, config.result_name(model), "main")


def clinical(model):
    for name in CLINICAL_CLS:
        data = {s: pd.read_parquet(config.CLINICAL_CLS_DIR / name / f"{s}.parquet") for s in ("train", "test")}
        emb = {s: np.load(config.RESULTS_DIR / "embeddings" / model / f"{name}_{s}.npy") for s in data}
        ytr, yte = data["train"]["label"].tolist(), data["test"]["label"].tolist()
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        if not isinstance(ytr[0], str):  # multi-label HoC
            mlb = MultiLabelBinarizer().fit(list(ytr) + list(yte))
            ytr, yte = mlb.transform(ytr), mlb.transform(yte)
            clf = OneVsRestClassifier(clf)
        pred = make_pipeline(StandardScaler(), clf).fit(emb["train"], ytr).predict(emb["test"])
        yte = np.asarray(yte, dtype=object if isinstance(yte[0], str) else None)
        if yte.ndim == 2:
            metric = lambda i: f1_score(yte[i], pred[i], average="micro", zero_division=0)
        else:
            metric = lambda i: f1_score(yte[i], pred[i], average="macro")
        score = metric(np.arange(len(yte)))
        lo, hi = bootstrap(metric, len(yte))
        stored = json.loads((config.RESULTS_DIR / "clinical_cls" / model / f"{name}.json").read_text())["main_score"]
        assert abs(score - stored) < 1e-6, (model, name, score, stored)
        write(model, name, score, lo, hi, "bootstrap")


def biosses(model, encoder):
    task = config.get_tasks(["BIOSSES"])[0]
    task.load_data()
    ds = task.dataset["test"]
    a = encoder.encode(list(ds["sentence1"]), normalize_embeddings=True)
    b = encoder.encode(list(ds["sentence2"]), normalize_embeddings=True)
    cos, gold = (a * b).sum(1), np.array(ds["score"])
    metric = lambda i: stats.spearmanr(cos[i], gold[i]).statistic
    score = metric(np.arange(len(gold)))
    stored = mteb_result(model, "BIOSSES").get_score()
    assert abs(score - stored) < 0.005, (model, score, stored)
    write(model, "BIOSSES", score, *bootstrap(metric, len(gold)), "bootstrap")


def from_mteb(model):
    for task in ("SciRepEvalDRSMClassification", "SciRepEvalMeSHDescriptorsClassification"):
        r = mteb_result(model, task).scores[task_split(task)][0]
        folds = [e["accuracy"] for e in r["scores_per_experiment"]]
        mean, lo, hi = t_interval(folds)
        assert abs(mean - r["main_score"]) < 1e-4
        write(model, task, mean, lo, hi, "folds")
    for task in ("MedrxivClusteringS2S.v2", "ClusTREC-Covid"):
        res = mteb_result(model, task)
        subsets = res.scores["test"]
        # The task score is the mean over subsets; the interval pools the resampled v-measures of all subsets.
        vals = [v for s in subsets for vs in s["v_measures"].values() for v in vs]
        _, lo, hi = t_interval(vals)
        score = res.get_score()
        write(model, task, score, score - (hi - lo) / 2, score + (hi - lo) / 2, "subsets")


def task_split(task):
    return config.get_tasks([task])[0].metadata.eval_splits[0]


if __name__ == "__main__":
    from run_benchmark import load_encoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="*", choices=config.MODELS, default=list(config.MODELS))
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for model in args.model:
        clinical(model)
        from_mteb(model)
        biosses(model, load_encoder(config.MODELS[model], device))
        print(f"{model}: intervals written", flush=True)
