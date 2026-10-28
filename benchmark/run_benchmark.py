"""Run the MTEB baseline suite for each model and cache results under results/baselines.

  python benchmark/run_benchmark.py [--model NAME ...] [--task NAME ...]

Each model is a frozen encoder with mean pooling (no fine-tuning): retrieval / STS /
clustering use cosine similarity of embeddings, and classification fits mteb's logistic
regression probe on the embeddings. A cached (model, task) result is skipped.
"""

import argparse
import time
from pathlib import Path

import config

import mteb
import torch
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig


REMOTE = {"trust_remote_code": True}  # UltraBERT ships its modelling code


def load_encoder(hub_id: str, device: str) -> SentenceTransformer:
    if (Path(hub_id) / "modules.json").exists():  # a trained Sentence Transformers model
        model = SentenceTransformer(hub_id, device=device, trust_remote_code=True)
        if config.MAX_SEQ_LENGTH is not None:
            model.max_seq_length = config.MAX_SEQ_LENGTH
        return model
    # Only models that ship their own modelling code (UltraBERT) get the remote-code kwargs; passing them
    # to plain BERT checkpoints without a tokenizer config breaks model construction.
    remote = hasattr(AutoConfig.from_pretrained(hub_id, trust_remote_code=True), "auto_map")
    # Fresh dicts per load: Sentence Transformers adds keys to these in place.
    kwargs = dict(model_kwargs={**REMOTE, "attn_implementation": "sdpa"}, config_kwargs=dict(REMOTE),
                  processor_kwargs=dict(REMOTE)) if remote else {}
    if config.MAX_SEQ_LENGTH is not None:  # otherwise the model's own maximum context is used
        kwargs["max_seq_length"] = config.MAX_SEQ_LENGTH
    transformer = models.Transformer(hub_id, **kwargs)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling], device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="*", choices=config.MODELS, default=list(config.MODELS))
    parser.add_argument("--task", nargs="*")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    cache = mteb.ResultCache(cache_path=str(config.RESULTS_DIR))
    tasks = config.get_tasks(args.task)

    for name in args.model:
        hub_id = config.MODELS[name]
        print(f"=== {name} ({hub_id}) on {device}", flush=True)
        model = SentenceTransformerEncoderWrapper(load_encoder(hub_id, device))
        model.mteb_model_meta = ModelMeta.create_empty(overwrites=dict(name=config.result_name(name), revision="main"))
        for task in tasks:
            start = time.time()
            result = mteb.evaluate(
                model,
                task,
                cache=cache,
                encode_kwargs={"batch_size": args.batch_size},
                raise_error=False,
                show_progress_bar=False,
            )
            scores = [f"{100 * r.get_score():.2f}" for r in result.task_results]
            print(f"  {task.metadata.name:40s} {','.join(scores) or 'FAILED':>8s}  {time.time() - start:6.0f}s", flush=True)
        del model
        torch.mps.empty_cache() if device == "mps" else None
