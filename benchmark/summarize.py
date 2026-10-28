"""Collect cached MTEB and clinical_cls results into one table (tasks x models) plus per-type and overall means.

  python benchmark/summarize.py   # prints markdown, writes results/baselines/summary.{md,csv}
"""

import json

import config
from clinical_cls import DATASETS as CLINICAL_CLS

import mteb
import pandas as pd


if __name__ == "__main__":
    cache = mteb.ResultCache(cache_path=str(config.RESULTS_DIR))
    task_type = {name: t for t, names in config.TASKS.items() for name in names}

    rows = []
    for name in config.TASK_NAMES:
        row = {"Type": task_type[name], "Task": name}
        for model, hub_id in config.MODELS.items():
            result = cache.load_task_result(name, config.result_name(model), "main")
            row[model] = round(100 * result.get_score(), 2) if result else None
        rows.append(row)
    for name in CLINICAL_CLS:
        row = {"Type": "Clinical classification", "Task": name}
        for model in config.MODELS:
            path = config.RESULTS_DIR / "clinical_cls" / model / f"{name}.json"
            row[model] = round(100 * json.loads(path.read_text())["main_score"], 2) if path.exists() else None
        rows.append(row)
    table = pd.DataFrame(rows)

    cols = list(config.MODELS)
    table[cols] = table[cols].astype(float)
    # A mean is reported only when the model has a score for every task it covers.
    means = table.groupby("Type", sort=False)[cols].agg(lambda x: x.mean(skipna=False)).round(2).reset_index()
    means.insert(1, "Task", "— mean —")
    overall = pd.DataFrame([{"Type": "All", "Task": "Mean(Task)", **table[cols].mean(skipna=False).round(2)}])
    by_type = pd.DataFrame([{"Type": "All", "Task": "Mean(TaskType)", **means[cols].mean(skipna=False).round(2)}])
    full = pd.concat([table, means, overall, by_type], ignore_index=True)

    md = full.to_markdown(index=False)
    print(md)
    (config.RESULTS_DIR / "summary.md").write_text(md + "\n")
    full.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
