"""Contrastive fine-tuning of UltraBERT-150M into Clinical UltraBERT.

  python training/train.py [--max_rows_per_source N] [--epochs N] [--max_steps N] [--output_dir DIR]

InfoNCE (MultipleNegativesRankingLoss) over in-batch negatives plus one mined hard
negative where the source has one, with GradCache so the effective batch can exceed what
fits in memory. Every batch is drawn from a single source (proportional sampling), so the
in-batch negatives are on-topic. Mean pooling, as the UltraBERT model card recommends for
dense retrieval, and the same pooling the benchmark uses for every baseline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import config  # noqa: E402  (HF caches -> data/)

import torch  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402
import pandas as pd  # noqa: E402
from sentence_transformers import (  # noqa: E402
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss  # noqa: E402
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers  # noqa: E402

BASE = "dmis-lab/UltraBERT-150M"
REMOTE = {"trust_remote_code": True}


def load_model(max_seq_length: int | None) -> SentenceTransformer:
    # max_seq_length None keeps UltraBERT's full 32K context; pass a value only to cap it.
    kwargs = {"max_seq_length": max_seq_length} if max_seq_length else {}
    transformer = models.Transformer(
        BASE,
        model_kwargs={**REMOTE, "attn_implementation": "sdpa"},
        config_kwargs=dict(REMOTE),
        processor_kwargs=dict(REMOTE),
        **kwargs,
    )
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(modules=[transformer, pooling], device=device)


def load_data(max_rows: int | None) -> DatasetDict:
    out = {}
    for path in sorted((config.DATA_DIR / "train").glob("*.parquet")):
        df = pd.read_parquet(path)
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=0)
        out[path.stem] = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)
    return DatasetDict(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(config.ROOT / "runs" / "clinical-ultrabert"))
    parser.add_argument("--max_rows_per_source", type=int)
    parser.add_argument("--epochs", type=float, default=None, help="default: the trainer's default (3)")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=None, help="optional cap; default: full context")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale", type=float, default=20.0, help="1 / temperature")
    parser.add_argument("--save_steps", type=int, default=250)
    args = parser.parse_args()

    model = load_model(args.max_seq_length)
    data = load_data(args.max_rows_per_source)
    print({k: len(v) for k, v in data.items()}, flush=True)

    loss = CachedMultipleNegativesRankingLoss(model, scale=args.scale, mini_batch_size=args.mini_batch_size)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        **({"num_train_epochs": args.epochs} if args.epochs is not None else {}),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.05,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        adam_beta2=0.98,
        max_grad_norm=1.0,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to=[],
        seed=0,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=data, loss=loss)
    trainer.train(resume_from_checkpoint=any(Path(args.output_dir).glob("checkpoint-*")) or None)
    model.save(str(Path(args.output_dir) / "final"))
    print(f"saved {args.output_dir}/final", flush=True)
