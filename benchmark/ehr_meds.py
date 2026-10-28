"""MIMIC-IV MEDS binary EHR tasks: serialized patient record -> frozen embedding -> logistic probe.

  python benchmark/ehr_meds.py prepare                         # build train/test examples (local only)
  python benchmark/ehr_meds.py evaluate [--model ...] [--task ...]

Cohorts, labels and leakage-safe patient windows come from the Open-MedJev implementation in
notes/Open-MedJev (imported, not modified): in-hospital mortality and length of stay > 7 days predicted 24 h
after admission, and 30-day readmission predicted at discharge. Each example's state (demographics,
admission, prior-year history, care units, vitals, abnormal-first labs, medications, infusions, procedures,
diagnoses) is serialized to plain text and embedded with each encoder at its full context length.

The protocol matches Open-MedJev's supervised baselines so the numbers are comparable: the same 1,000 training
examples (MEDS train shard 0) and the same 1,000 held-out test examples per task, and a logistic-regression
probe on standardized embeddings with C chosen by 5-fold cross-validated AUROC on the training set only.

MIMIC-IV is credentialed data: examples and embeddings stay in data/ehr_meds/ and only aggregate metrics are
written to results/baselines/ehr/.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import config

# Open-MedJev (cohorts, labels, serialization); set MEDJEV_ROOT if it lives elsewhere.
MEDJEV = Path(os.environ.get("MEDJEV_ROOT", config.ROOT / "notes" / "Open-MedJev"))
EHR_DIR = config.DATA_DIR / "ehr_meds"
os.environ.setdefault("MEDJEV_MEDS_DIR", str(config.DATA_DIR / "MIMIC_MEDS_Extract_v0.0.3_test"))
os.environ.setdefault("MEDJEV_DATA_DIR", str(EHR_DIR))
sys.path.insert(0, str(MEDJEV / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from medjev.metrics import binary_metrics  # noqa: E402
from medjev.tasks import meds  # noqa: E402

TASKS = ["meds_mortality", "meds_long_los", "meds_readmission_30d"]
N_TRAIN = N_TEST = 1000  # Open-MedJev's label budget and evaluation sample
C_GRID = (0.01, 0.1, 1.0, 10.0)  # same grid as Open-MedJev's TF-IDF + LR baseline
OUT = config.RESULTS_DIR / "ehr"


def serialize(state: dict) -> str:
    """Patient state -> plain text, one line per top-level section, e.g.
    'patient: age: 67; sex: female' / 'this admission so far: care units: MICU; labs latest: Lactate: 4.1 mmol/L (high), ...'"""
    def flat(v):
        if isinstance(v, dict):
            return "; ".join(f"{k.replace('_', ' ')}: {flat(x)}" for k, x in v.items())
        if isinstance(v, list):
            return ", ".join(flat(x) for x in v)
        return str(v)
    return "\n".join(f"{k.replace('_', ' ')}: {flat(v)}" for k, v in state.items())


def prepare():
    # Reuse Open-MedJev's cached admission tables when present (copied, never written in notes/).
    src, dst = MEDJEV / "data" / "cohorts", EHR_DIR / "cohorts"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("admissions_*.parquet"):
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
    for task in TASKS:
        for split, ex in (("test", meds.build(task, "held_out", N_TEST)),
                          ("train", meds.build(task, "train", N_TRAIN, shards=1))):
            rows = [{"uid": e.uid, "text": serialize(e.state), "label": bool(e.gold["label"])} for e in ex]
            path = EHR_DIR / "examples" / f"{task}_{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{task:22s} {split:5s} {len(rows)} examples, {sum(r['label'] for r in rows)} positive", flush=True)


def load(task: str, split: str) -> list[dict]:
    return [json.loads(line) for line in (EHR_DIR / "examples" / f"{task}_{split}.jsonl").open()]


def tune_c(x: np.ndarray, y: np.ndarray) -> float:
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    def cv(c):
        return np.mean([roc_auc_score(y[va], LogisticRegression(C=c, max_iter=3000).fit(x[tr], y[tr])
                                      .predict_proba(x[va])[:, 1]) for tr, va in folds])
    return max(C_GRID, key=cv)


def evaluate(model_names, tasks, batch_size):
    from run_benchmark import load_encoder

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    for name in model_names:
        todo = [t for t in tasks if not (OUT / name / f"{t}.json").exists()]
        if not todo:
            continue
        print(f"=== {name} on {device}", flush=True)
        encoder = load_encoder(config.MODELS[name], device)
        for task in todo:
            start = time.time()
            data = {s: load(task, s) for s in ("train", "test")}
            emb = {}
            for split, rows in data.items():
                path = EHR_DIR / "embeddings" / name / f"{task}_{split}.npy"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, encoder.encode([r["text"] for r in rows], batch_size=batch_size))
                emb[split] = np.load(path)
            y_tr = np.array([r["label"] for r in data["train"]], dtype=int)
            y_te = [r["label"] for r in data["test"]]
            scaler = StandardScaler().fit(emb["train"])
            x_tr, x_te = scaler.transform(emb["train"]), scaler.transform(emb["test"])
            c = tune_c(x_tr, y_tr)
            probs = LogisticRegression(C=c, max_iter=3000).fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
            metrics = binary_metrics(y_te, probs.tolist()) | {"C": c, "n_train": len(y_tr), "model": config.result_name(name)}
            (OUT / name).mkdir(parents=True, exist_ok=True)
            (OUT / name / f"{task}.json").write_text(json.dumps(metrics, indent=1))
            print(f"  {task:22s} AUROC {metrics['auroc']:.3f}  AUPRC {metrics['auprc']:.3f}  C={c}  "
                  f"{time.time() - start:5.0f}s", flush=True)
        del encoder
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "evaluate"])
    parser.add_argument("--task", nargs="*", choices=TASKS, default=TASKS)
    parser.add_argument("--model", nargs="*", choices=config.MODELS, default=list(config.MODELS))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        evaluate(args.model, args.task, args.batch_size)
