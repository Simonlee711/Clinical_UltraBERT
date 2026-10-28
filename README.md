# Clinical UltraBERT: An Open, State-of-the-Art Encoder for Classification and Retrieval

**Contrastive fine-tuning turns a general-purpose encoder into a medical text embedding model.**

Clinical UltraBERT is [UltraBERT-150M](https://huggingface.co/dmis-lab/UltraBERT-150M) after contrastive
fine-tuning on 100,000 public biomedical and clinical text pairs, trained on a single laptop GPU.
On a benchmark of 20 public medical tasks it has the best score on 15 and raises the mean score from 43.8 (the
strongest of six public encoders) to 55.6. Mean retrieval nDCG@10 doubles, from 22.0 to 43.8, and it leads on
all ten retrieval corpora. Gains on classification are mostly within statistical uncertainty, and prediction of
hospital outcomes from serialized MIMIC-IV records does not improve.

This repository holds the code to rebuild the benchmark, train the model, evaluate all encoders and analyse
the embedding space.

## Results

Mean scores (×100) per task type. Retrieval is nDCG@10; classification is accuracy or F1; similarity is
Spearman correlation; clustering is V-measure. EHR is the mean AUROC over three MIMIC-IV outcome-prediction
tasks. Every model is evaluated as a frozen, mean-pooled encoder.

| Model | Retrieval (10) | Classification (7) | Similarity (1) | Clustering (2) | **All 20 tasks** | EHR (3) |
|---|---|---|---|---|---|---|
| BERT | 14.8 | 62.9 | 54.7 | 52.0 | 37.3 | 72.5 |
| Bio_ClinicalBERT | 12.2 | 63.6 | 66.3 | 53.4 | 37.0 | 71.0 |
| Clinical ModernBERT | 11.6 | 62.9 | 70.4 | 52.3 | 36.6 | 72.1 |
| BioClinical ModernBERT | 15.0 | 66.3 | 66.4 | 54.1 | 39.4 | **73.0** |
| ModernBERT | 7.3 | 63.5 | 57.0 | 52.9 | 34.0 | 72.2 |
| UltraBERT | 22.0 | 66.9 | 75.1 | 55.9 | 43.8 | 72.3 |
| **Clinical UltraBERT** | **43.8** | **68.4** | **84.2** | **56.1** | **55.6** | 69.6 |

Per-task scores with 95% confidence intervals are in the paper.

## Using the model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Simonlee711/Clinical-UltraBERT", trust_remote_code=True)
queries = model.encode(["What are the first-line treatments for community-acquired pneumonia?"])
docs = model.encode(["Empiric therapy for outpatient pneumonia includes amoxicillin or doxycycline.",
                     "Metformin is the preferred initial agent for type 2 diabetes."])
print(model.similarity(queries, docs))
```

`trust_remote_code=True` is needed because UltraBERT ships its own modelling code.

## Training and evaluation

Python 3.12 on macOS (Apple silicon) or Linux. All commands run from the repository root.

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

**1. Benchmark data** (about 11 GB, written to `data/`):

```bash
python benchmark/download_data.py           # 15 MTEB tasks
python benchmark/clinical_cls.py download   # MTSamples, Symptom2Diagnosis, Medical Abstracts, PubMed RCT, HoC
```

**2. Train Clinical UltraBERT** (about 11.5 hours on an Apple M4 Max):

```bash
python training/prepare_data.py             # 100K pairs from 10 sources -> data/train/
python training/train.py --max_rows_per_source 10000 --batch_size 64
python training/export_hf.py                # Hugging Face-ready folder -> huggingface/
```

**3. Evaluate** all seven models (results are written to `results/baselines/`):

```bash
./benchmark/run_all.sh                      # clinical classification + MTEB tasks + summary
python benchmark/confidence.py              # 95% confidence intervals (non-retrieval tasks)
```

**4. EHR tasks (MIMIC-IV, credentialed).** Requires a MEDS extract of MIMIC-IV in
`data/MIMIC_MEDS_Extract_v0.0.3_test/` and the Open-MedJev cohort code (set `MEDJEV_ROOT` to its location):

```bash
MEDJEV_ROOT=/path/to/Open-MedJev python benchmark/ehr_meds.py prepare
MEDJEV_ROOT=/path/to/Open-MedJev python benchmark/ehr_meds.py evaluate
```

**5. Embedding-space analysis:**

```bash
python analysis/compute.py && python analysis/whitening.py && python analysis/plots.py
```

The analysis figures use the `.ttf` fonts in `fonts/`, which are not redistributed. Add your own fonts there
and set the font family in `analysis/plots.py`.

## Repository layout

```
benchmark/   task definitions, evaluation of all encoders, EHR tasks, confidence intervals, summary
training/    training data preparation, contrastive fine-tuning, Hugging Face export
analysis/    embedding-space analysis (anisotropy, whitening, lexical overlap, synonyms)
```

## Data and privacy

All benchmark datasets are public except MIMIC-IV, which is available to credentialed users from PhysioNet.
MIMIC-IV-derived records and embeddings stay in `data/`, which is git-ignored, and no results or data are
included in this repository.

## Citation

```bibtex
@article{lee2026clinicalultrabert,
  title  = {Clinical {UltraBERT}: contrastive fine-tuning turns a general-purpose encoder into a medical text
            embedding model},
  author = {Lee, Simon A.},
  year   = {2026},
  note   = {Preprint}
}
```
