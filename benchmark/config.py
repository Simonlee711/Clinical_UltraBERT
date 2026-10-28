"""Shared settings: which models and MTEB tasks the baseline benchmark covers.

Every Hugging Face download (datasets and their processed caches) goes under DATA_DIR,
so `data/` holds the whole benchmark suite and the eval can run offline afterwards.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "baselines"

# Point the hub and datasets caches at data/ before anything imports huggingface_hub.
os.environ.setdefault("HF_HUB_CACHE", str(DATA_DIR / "hf_hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATA_DIR / "hf_datasets"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Clinical classification datasets that are not in MTEB; see clinical_cls.py.
CLINICAL_CLS_DIR = DATA_DIR / "clinical_cls"

# Short name -> hub id (or local path). The baselines ship no Sentence Transformers config,
# so all of them are evaluated the same way: frozen encoder, mean pooling over the last
# layer. Clinical UltraBERT was trained with the same mean pooling.
MODELS = {
    "BERT-base": "google-bert/bert-base-uncased",
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "Clinical_ModernBERT": "Simonlee711/Clinical_ModernBERT",
    "BioClinical-ModernBERT": "thomas-sounack/BioClinical-ModernBERT-base",
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    "UltraBERT-150M": "dmis-lab/UltraBERT-150M",
    # Ours: UltraBERT-150M after contrastive fine-tuning (training/train.py).
    "Clinical-UltraBERT": str(ROOT / "runs" / "clinical-ultrabert" / "final"),
}
OURS = "Clinical-UltraBERT"


def result_name(model: str) -> str:
    """Name under which a model's results are stored: the hub id, or the short name for a local checkpoint
    (so result files do not depend on where the repository lives)."""
    hub_id = MODELS[model]
    return model if hub_id.startswith("/") else hub_id

# Well-known biomedical / clinical MTEB tasks (medical-only suite).
TASKS = {
    "Retrieval": [
        "SciFact",                     # BEIR: scientific claim -> supporting abstract
        "NFCorpus",                    # BEIR: nutrition / medical queries -> PubMed
        "TRECCOVID",                   # BEIR: COVID-19 queries -> CORD-19 (171K docs)
        "MedicalQARetrieval",          # consumer medical question -> answer
        "PublicHealthQA",              # public-health FAQ retrieval
        "CovidQA",                     # COVID-19 QA retrieval
        "ChatDoctorRetrieval",         # patient question -> doctor answer
        "R2MEDPMCClinicalRetrieval",   # clinical case report -> relevant PMC article
        "R2MEDMedQADiagRetrieval",     # USMLE-style vignette -> diagnostic evidence
        "BIRCO-ClinicalTrial",         # patient description -> matching clinical trial
    ],
    "Classification": [
        "SciRepEvalDRSMClassification",             # PubMed abstract -> DRSM study type
        "SciRepEvalMeSHDescriptorsClassification",  # PubMed abstract -> MeSH descriptor
    ],
    "STS": [
        "BIOSSES",                     # biomedical sentence similarity
    ],
    "Clustering": [
        "MedrxivClusteringS2S.v2",
        "ClusTREC-Covid",
    ],
}
TASK_NAMES = [name for names in TASKS.values() for name in names]

MAX_SEQ_LENGTH = None  # None: every model reads up to its own maximum context (512 BERT, 8K ModernBERT, 32K UltraBERT)


def get_tasks(names=None):
    import mteb

    return mteb.get_tasks(tasks=names or TASK_NAMES, languages=["eng"])
