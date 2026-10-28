"""Download every benchmark dataset into data/.

  python benchmark/download_data.py [--task NAME ...]

Calls each MTEB task's own loader, so exactly the pinned revision and splits that the
evaluation reads end up in data/hf_hub (raw files) and data/hf_datasets (arrow cache).
"""

import argparse

import config  # noqa: F401  (sets HF cache env vars first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    for task in config.get_tasks(args.task):
        name = task.metadata.name
        try:
            task.load_data()
            print(f"ok    {name:40s} {task.metadata.dataset['path']}", flush=True)
        except Exception as e:  # keep going; report failures at the end of the log
            print(f"FAIL  {name:40s} {type(e).__name__}: {e}", flush=True)
