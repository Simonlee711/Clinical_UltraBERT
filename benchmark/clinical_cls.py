"""Clinical / biomedical text classification datasets that MTEB does not cover.

  python benchmark/clinical_cls.py download
  python benchmark/clinical_cls.py evaluate [--model NAME ...] [--dataset NAME ...]

`download` writes data/clinical_cls/<name>/{train,test}.parquet with columns text, label
(label is a list of strings for the multi-label HoC). `evaluate` uses the same protocol
as MTEB classification: frozen mean-pooled embeddings + a logistic-regression probe (on
standardised features) fit on the train split, scored on the test split. Embeddings are
cached in results/baselines/embeddings/<model>/<dataset>_<split>.npy. Results go to
results/baselines/clinical_cls/<model>/<dataset>.json; existing results are skipped.
"""

import argparse
import json
import time

import config

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

SEED = 42
MAX_TRAIN, MAX_TEST = 20000, 10000  # subsample PubMed RCT (176K / 30K sentences)

# name -> description; main metric is macro-F1 (micro-F1 for multi-label HoC).
DATASETS = {
    "MTSamples": "clinical transcription note -> medical specialty (40 classes)",
    "Symptom2Diagnosis": "patient symptom description -> diagnosis (22 classes)",
    "MedicalAbstracts": "PubMed abstract -> disease category (5 classes)",
    "PubMedRCT20k": "RCT abstract sentence -> role, e.g. METHODS / RESULTS (5 classes)",
    "HoC": "PubMed abstract -> hallmarks of cancer (multi-label, 10 classes)",
}


def _hoc_docs(split):
    # Sentence-level rows; the standard (BLURB) setting is per abstract: join the
    # sentences and take the union of their labels, dropping "none".
    df = split.to_pandas()
    df["pmid"] = df["document_id"].str.split("_").str[0]
    df["order"] = df["document_id"].str.split("_").str[1].astype(int)
    df = df.sort_values(["pmid", "order"])
    return df.groupby("pmid", sort=False).agg(
        text=("text", " ".join),
        label=("labels", lambda ls: sorted({l for x in ls for l in x if l != "none"})),
    ).reset_index(drop=True)


def load_source(name):
    """Return {"train": df, "test": df} with text / label columns."""
    if name == "MTSamples":
        ds = load_dataset("hpe-ai/medical-cases-classification-tutorial")
        out = {s: ds[s].to_pandas() for s in ("train", "test")}
        return {s: pd.DataFrame({"text": d["transcription"], "label": d["medical_specialty"].str.strip()})
                for s, d in out.items()}
    if name == "Symptom2Diagnosis":
        ds = load_dataset("gretelai/symptom_to_diagnosis")
        return {s: pd.DataFrame({"text": ds[s]["input_text"], "label": ds[s]["output_text"]})
                for s in ("train", "test")}
    if name == "MedicalAbstracts":
        ds = load_dataset("TimSchopf/medical_abstracts", "default")
        names = load_dataset("TimSchopf/medical_abstracts", "labels")["train"].to_pandas()
        names = dict(zip(names["condition_label"], names["condition_name"]))
        return {s: pd.DataFrame({"text": ds[s]["medical_abstract"],
                                 "label": [names[l] for l in ds[s]["condition_label"]]})
                for s in ("train", "test")}
    if name == "PubMedRCT20k":
        ds = load_dataset("armanc/pubmed-rct20k")
        out = {}
        for s, n in (("train", MAX_TRAIN), ("test", MAX_TEST)):
            d = ds[s].to_pandas()[["text", "label"]]
            out[s] = d.sample(n=min(n, len(d)), random_state=SEED).reset_index(drop=True)
        return out
    if name == "HoC":
        # The hub repo is a loading script; read the hub's own parquet conversion instead.
        url = "hf://datasets/bigbio/hallmarks_of_cancer@refs%2Fconvert%2Fparquet/hallmarks_of_cancer_bigbio_text/{}/0000.parquet"
        ds = load_dataset("parquet", data_files={s: url.format(s) for s in ("train", "test")})
        return {s: _hoc_docs(ds[s]) for s in ("train", "test")}
    raise KeyError(name)


def download(names):
    for name in names:
        out_dir = config.CLINICAL_CLS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, df in load_source(name).items():
            df.to_parquet(out_dir / f"{split}.parquet", index=False)
            print(f"ok    {name:20s} {split:5s} {len(df):6d} rows", flush=True)


def probe(train_x, train_y, test_x, test_y):
    multilabel = isinstance(train_y[0], (list, np.ndarray))
    # Standardise the features: on raw or unit-norm embeddings the default C=1 penalty
    # underfits badly (multi-label HoC collapses to predicting no labels).
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    if multilabel:
        mlb = MultiLabelBinarizer().fit(list(train_y) + list(test_y))
        train_y, test_y = mlb.transform(train_y), mlb.transform(test_y)
        clf = OneVsRestClassifier(clf)
    clf = make_pipeline(StandardScaler(), clf)
    if multilabel:
        pred = clf.fit(train_x, train_y).predict(test_x)
        return {"main_score": f1_score(test_y, pred, average="micro", zero_division=0),
                "f1_micro": f1_score(test_y, pred, average="micro", zero_division=0),
                "f1_macro": f1_score(test_y, pred, average="macro", zero_division=0)}
    pred = clf.fit(train_x, train_y).predict(test_x)
    return {"main_score": f1_score(test_y, pred, average="macro"),
            "f1_macro": f1_score(test_y, pred, average="macro"),
            "accuracy": accuracy_score(test_y, pred)}


def evaluate(model_names, names, batch_size):
    from run_benchmark import load_encoder

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    for model_name in model_names:
        out_dir = config.RESULTS_DIR / "clinical_cls" / model_name
        todo = [n for n in names if not (out_dir / f"{n}.json").exists()]
        if not todo:
            continue
        print(f"=== {model_name} on {device}", flush=True)
        encoder = load_encoder(config.MODELS[model_name], device)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in todo:
            start = time.time()
            data = {s: pd.read_parquet(config.CLINICAL_CLS_DIR / name / f"{s}.parquet") for s in ("train", "test")}
            emb = {}
            for split, d in data.items():
                path = config.RESULTS_DIR / "embeddings" / model_name / f"{name}_{split}.npy"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, encoder.encode(d["text"].tolist(), batch_size=batch_size))
                emb[split] = np.load(path)
            scores = probe(emb["train"], data["train"]["label"].tolist(), emb["test"], data["test"]["label"].tolist())
            scores |= {"model": config.result_name(model_name), "dataset": name,
                       "n_train": len(data["train"]), "n_test": len(data["test"])}
            (out_dir / f"{name}.json").write_text(json.dumps(scores, indent=2))
            print(f"  {name:20s} {100 * scores['main_score']:6.2f}  {time.time() - start:6.0f}s", flush=True)
        del encoder
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "evaluate"])
    parser.add_argument("--dataset", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--model", nargs="*", choices=config.MODELS, default=list(config.MODELS))
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.command == "download":
        download(args.dataset)
    else:
        evaluate(args.model, args.dataset, args.batch_size)
