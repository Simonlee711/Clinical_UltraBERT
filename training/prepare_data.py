"""Build the Clinical UltraBERT contrastive fine-tuning set.

  python training/prepare_data.py

Writes data/train/<source>.parquet with columns anchor, positive, negative (one mined hard
negative where the source has them, else no negative column). Task instructions in the
F2LLM-v2 queries are stripped, because the benchmark encodes queries without prompts.

Left out on purpose because a benchmark task is built from the same data:
F2LLM-v2 medquad (MedicalQARetrieval), scifact (SciFact), medqa_en (R2MED MedQA-Diag),
medrxiv_clustering_* (MedrxivClustering), medical_instruction (ChatDoctor / HealthCareMagic).
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import config  # noqa: E402  (HF caches -> data/)

SEED = 0
RAW = config.DATA_DIR / "train_raw"
OUT = config.DATA_DIR / "train"

# F2LLM-v2 file -> number of rows to keep (None = all). All have 24 mined negatives.
F2LLM = {
    "pubmedqa": None,            # research question -> PubMed abstract
    "bioasq": 60000,             # biomedical question -> PubMed abstract
    "medical_flashcards": None,  # medical question -> answer
    "medmcqa": None,             # exam question -> answer
    "medinstruct": 30000,        # medical instruction -> response
    "mscinli": None,             # scientific NLI (entailment)
    "snli": None,                # general NLI, for sentence similarity
    "mnli": 40000,
}
PUBMED_CHUNKS = range(1000, 1166, 6)  # recent PubMed baseline files (2010s-2022)
PUBMED_ROWS = 60000


def strip_instruction(text: str) -> str:
    return re.sub(r"^Instruct:.*?\nQuery:\s*", "", text, flags=re.S).strip()


def f2llm(name: str, n: int | None) -> pd.DataFrame:
    path = hf_hub_download("codefuse-ai/F2LLM-v2", f"{name}.parquet", repo_type="dataset",
                           local_dir=RAW / "f2llm-v2")
    df = pq.read_table(path, columns=["query", "passage", "negative_1"]).to_pandas()
    df = df.dropna()
    if n and len(df) > n:
        df = df.sample(n=n, random_state=SEED)
    return pd.DataFrame({
        "anchor": df["query"].map(strip_instruction),
        "positive": df["passage"],
        "negative": df["negative_1"],
    })


def pubmed_title_abstract() -> pd.DataFrame:
    rows = []
    for i in PUBMED_CHUNKS:
        path = hf_hub_download("MedRAG/pubmed", f"chunk/pubmed23n{i:04d}.jsonl", repo_type="dataset",
                               local_dir=RAW / "medrag-pubmed")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if len(r["title"]) > 20 and len(r["content"]) > 200:
                    rows.append((r["title"].strip(), r["content"]))
    random.Random(SEED).shuffle(rows)
    return pd.DataFrame(rows[:PUBMED_ROWS], columns=["anchor", "positive"])


def clinical_notes() -> pd.DataFrame:
    # Doctor-patient conversation <-> the clinical note it was generated from.
    path = hf_hub_download("AGBonnet/augmented-clinical-notes", "augmented_notes_30K.jsonl",
                           repo_type="dataset", local_dir=RAW / "augmented-clinical-notes")
    df = pd.read_json(path, lines=True)
    df = df.dropna(subset=["conversation", "note"])
    return pd.DataFrame({"anchor": df["conversation"], "positive": df["note"]})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {name: (lambda name=name, n=n: f2llm(name, n)) for name, n in F2LLM.items()}
    sources["pubmed_title_abstract"] = pubmed_title_abstract
    sources["clinical_notes"] = clinical_notes
    total = 0
    for name, build in sources.items():
        df = build()
        df = df[(df["anchor"].str.len() > 0) & (df["positive"].str.len() > 0)].reset_index(drop=True)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        total += len(df)
        print(f"{name:24s} {len(df):7d} rows  cols={list(df.columns)}", flush=True)
    print(f"total {total} rows")
