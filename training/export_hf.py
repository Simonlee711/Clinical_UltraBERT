"""Export Clinical UltraBERT to huggingface/ for upload to the Hub.

  python training/export_hf.py            # weights + model card
  python training/export_hf.py --card     # refresh only the model card (results table)

Writes the Sentence Transformers model (weights, tokenizer, UltraBERT modelling code,
pooling config) at its full context length, and a README.md model card whose results table is
read from results/baselines/summary.csv.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
import config  # noqa: E402

import pandas as pd  # noqa: E402

OUT = ROOT / "huggingface"
REPO_ID = "Simonlee711/Clinical-UltraBERT"  # suggested Hub repo; change before uploading if needed

BASELINES = ["BERT-base", "Bio_ClinicalBERT", "Clinical_ModernBERT", "BioClinical-ModernBERT", "ModernBERT-base",
             "UltraBERT-150M"]
SHORT = {
    "MedicalQARetrieval": "MedicalQA", "ChatDoctorRetrieval": "ChatDoctor", "TRECCOVID": "TREC-COVID",
    "R2MEDPMCClinicalRetrieval": "R2MED PMC-Clinical", "R2MEDMedQADiagRetrieval": "R2MED MedQA-Diag",
    "BIRCO-ClinicalTrial": "BIRCO Clinical-Trial", "SciRepEvalDRSMClassification": "SciRepEval DRSM",
    "SciRepEvalMeSHDescriptorsClassification": "SciRepEval MeSH", "MedrxivClusteringS2S.v2": "MedrxivClusteringS2S",
    "PubMedRCT20k": "PubMed RCT 20k", "MedicalAbstracts": "Medical Abstracts",
}


def results_table() -> str:
    df = pd.read_csv(config.RESULTS_DIR / "summary.csv")
    tasks = df[~df["Task"].isin(["— mean —", "Mean(Task)", "Mean(TaskType)"])]
    head = "| Type | Task | Best baseline | Clinical UltraBERT |\n|---|---|---|---|"
    rows, wins, done = [], 0, 0
    for _, r in tasks.iterrows():
        base = r[BASELINES].astype(float)
        best = base.idxmax() if base.notna().any() else None
        ours = r[config.OURS]
        best_txt = f"{base[best]:.1f} ({best})" if best else "–"
        if pd.isna(ours):
            ours_txt = "pending"
        else:
            done += 1
            win = best is None or ours >= base[best]
            wins += win
            ours_txt = f"**{ours:.1f}**" if win else f"{ours:.1f}"
        kind = "Classification" if r["Type"] == "Clinical classification" else r["Type"]
        rows.append(f"| {kind} | {SHORT.get(r['Task'], r['Task'])} | {best_txt} | {ours_txt} |")
    summary = f"Clinical UltraBERT has the best score on **{wins} of {done}** tasks"
    summary += "." if done == len(tasks) else f" evaluated so far ({len(tasks) - done} still running)."
    return summary + "\n\n" + head + "\n" + "\n".join(rows)


CARD = """---
license: apache-2.0
language:
- en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: dmis-lab/UltraBERT-150M
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- medical
- clinical
- biomedical
- ultrabert
datasets:
- codefuse-ai/F2LLM-v2
- MedRAG/pubmed
- AGBonnet/augmented-clinical-notes
---

# Clinical UltraBERT

Clinical UltraBERT is a biomedical and clinical text embedding model. It maps text to a
768-dimensional vector for retrieval, semantic similarity, clustering and classification.
It is [UltraBERT-150M](https://huggingface.co/dmis-lab/UltraBERT-150M) fine-tuned with a
contrastive loss on about 100K biomedical and clinical text pairs.

| | |
|---|---|
| Base model | `dmis-lab/UltraBERT-150M` (151M parameters, 24 layers) |
| Embedding | 768-d, mean pooling, cosine similarity |
| Max length | 32,768 tokens (UltraBERT's full context) |
| Prompts | none; queries and documents are encoded as plain text |

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{repo_id}", trust_remote_code=True)

queries = ["What are the first-line treatments for community-acquired pneumonia?"]
documents = [
    "Empiric therapy for outpatient community-acquired pneumonia includes amoxicillin or doxycycline.",
    "Metformin is the preferred initial pharmacologic agent for type 2 diabetes.",
]
q = model.encode(queries)
d = model.encode(documents)
print(model.similarity(q, d))
```

`trust_remote_code=True` is needed because UltraBERT ships its own modelling code
(`modeling_ultrabert.py`); from `transformers >= 5.20` the architecture is supported natively.

## Training

Contrastive fine-tuning with InfoNCE (`CachedMultipleNegativesRankingLoss`, scale 20) using
in-batch negatives plus one mined hard negative where available. Every batch is drawn from a
single source, so in-batch negatives are on-topic. Batch size 64,
learning rate 5e-5 with 5% warm-up and linear decay, AdamW (β2 0.98, weight decay 0.01),
full-length sequences (no truncation), mean pooling, trained in fp32 on a single Apple M4 Max GPU.

10K pairs from each of these ten sources (100,000 pairs in all):

| Source | Pairs |
|---|---|
| PubMedQA, BioASQ (F2LLM-v2) | question → PubMed abstract |
| Medical flashcards, MedMCQA, MedInstruct (F2LLM-v2) | medical question → answer |
| MSciNLI, SNLI, MNLI (F2LLM-v2) | premise → entailed hypothesis |
| PubMed (MedRAG/pubmed) | article title → abstract |
| Augmented clinical notes | doctor–patient conversation → clinical note |

F2LLM-v2 subsets built from benchmark test data (MedQuAD, SciFact, MedQA, medRxiv
clustering) and HealthCareMagic doctor answers (the ChatDoctor retrieval corpus) were
left out of training.

## Evaluation

A 20-task medical suite: 10 retrieval tasks, 7 classification tasks, BIOSSES and 2
clustering tasks, from MTEB plus five clinical classification datasets. Every model reads inputs
up to its own maximum context. The baselines (BERT, Bio_ClinicalBERT, ModernBERT, Clinical
ModernBERT, BioClinical ModernBERT, UltraBERT) are masked-LM checkpoints used as frozen
mean-pooled encoders, so they are expected to trail a contrastively trained model on retrieval.
Classification scores come from a logistic-regression probe on frozen embeddings.

Scores ×100: retrieval nDCG@10; SciRepEval accuracy; MTSamples, Symptom2Diagnosis, Medical
Abstracts and PubMed RCT macro-F1; HoC micro-F1; BIOSSES Spearman; clustering V-measure.

{results}

## Limitations

- Most training and evaluation text is biomedical literature or consumer health text. Real
  clinical notes appear only through MTSamples transcriptions and synthetic doctor–patient
  conversations; no credentialed EHR data (e.g. MIMIC) was used for training.
- Classification of document structure (study type, sentence role) is weaker than the best
  masked-LM encoders, and fine-tuning lowered a few probe scores relative to UltraBERT-150M
  (SciRepEval DRSM, HoC, MTSamples, MedrxivClusteringS2S).
- On MIMIC-IV prediction from serialized EHR records (mortality, length of stay, readmission)
  the model does not improve over its base model.
- The model is not a medical device and its outputs must not be used for clinical decisions
  without expert review.

## License

Apache 2.0, like the base model. The training data keeps its own licenses: F2LLM-v2
(Apache 2.0), augmented clinical notes (MIT) and PubMed abstracts (NLM terms of use).
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", action="store_true", help="only rewrite README.md")
    args = parser.parse_args()

    if not args.card:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.MODELS[config.OURS], trust_remote_code=True, device="cpu")
        if config.MAX_SEQ_LENGTH is not None:
            model.max_seq_length = config.MAX_SEQ_LENGTH
        model.save(str(OUT))
    (OUT / "README.md").write_text(CARD.replace("{repo_id}", REPO_ID).replace("{results}", results_table()))
    print(f"wrote {OUT}")
