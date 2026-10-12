"""Metrics, plots and model comparison. Framework-agnostic: works on numpy arrays."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than +-1.96*se for small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def evaluate_predictions(y_true, y_prob, class_names) -> dict:
    """Full metric set from true integer labels and predicted class probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    labels = list(range(len(class_names)))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    _, _, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    n_correct = int((y_true == y_pred).sum())
    lo, hi = wilson_ci(n_correct, len(y_true))

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci": [lo, hi],
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(w_f1),
        "per_class": {
            name: {"precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm, class_names, title="Confusion matrix", ax=None):
    """Heat-map showing counts, with row-normalised percentages (= per-class recall)."""
    cm = np.asarray(cm)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if row_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center",
                    va="center", color=colour, fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return ax


def plot_history(history: dict, title: str = "", finetune_start: int | None = None):
    """Training vs validation accuracy and loss. Dashed line marks where fine-tuning began."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        if finetune_start:
            ax.axvline(finetune_start - 0.5, ls="--", c="grey", lw=1, label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{title} {label}".strip())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def predict_probs(model, ds):
    """Run a model over an (unshuffled) tf.data pipeline -> (y_true, y_prob)."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0)
    return y_true, y_prob


def measure_latency_ms(model, input_shape=(224, 224, 3), n_runs: int = 30) -> float:
    """Median single-image inference time in milliseconds (after one warm-up call)."""
    x = np.random.default_rng(0).random((1, *input_shape), dtype=np.float32)
    model.predict(x, verbose=0)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model. `results[name]` must hold keys:
    'all' and 'leak_free' (outputs of evaluate_predictions on the whole test set and on
    its leak-free subset) plus 'params', 'size_mb', 'latency_ms'.

    Headline columns come from the leak-free evaluation. Columns for the unfiltered test
    set are only added when the filter actually removed images (official split mode);
    with the grouped split both evaluations are identical and the extra columns are omitted.
    """
    show_all = any(r["all"]["n"] != r["leak_free"]["n"] for r in results.values())
    rows = []
    for name, r in results.items():
        row = {"model": name,
               "params_M": round(r["params"] / 1e6, 2),
               "size_MB": round(r["size_mb"], 1),
               "latency_ms": round(r["latency_ms"], 1)}
        if show_all:
            row["acc_all_test"] = round(r["all"]["accuracy"], 4)
            row["f1_all_test"] = round(r["all"]["macro_f1"], 4)
        row.update({
            "accuracy": round(r["leak_free"]["accuracy"], 4),
            "acc_ci_low": round(r["leak_free"]["accuracy_ci"][0], 4),
            "acc_ci_high": round(r["leak_free"]["accuracy_ci"][1], 4),
            "precision": round(r["leak_free"]["macro_precision"], 4),
            "recall": round(r["leak_free"]["macro_recall"], 4),
            "f1": round(r["leak_free"]["macro_f1"], 4),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def select_best_model(table: pd.DataFrame, tolerance: float = 0.005) -> str:
    """Pick the deployment model.

    Rule: highest macro-F1 on the clean (leak-free) test set; among models within
    `tolerance` of that score, prefer the one with the fewest parameters
    (cheaper to serve, and the difference is inside the noise anyway).
    """
    top = table["f1"].max()
    contenders = table[table["f1"] >= top - tolerance]
    return str(contenders.sort_values("params_M").iloc[0]["model"])


def evaluate_saved_models(data_dir, models_dir, reports_dir,
                          split_mode: str = "grouped") -> pd.DataFrame:
    """Evaluate every models/<name>.h5 on the official test set AND its leak-free subset.

    Writes reports/metrics_<name>.json, reports/model_comparison.csv and
    models/best_model.json (read by the Streamlit app to pick its default model).
    """
    from tensorflow import keras  # lazy: keeps the rest of this module TF-free

    from .config import CLASS_NAMES
    from .data import load_dataframe, make_dataset
    from .inference import available_models

    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(data_dir, split_mode)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_ds = make_dataset(test_df["path"], test_df["label"], False)
    mask = test_df["leak_free"].to_numpy()

    results = {}
    for name, path in available_models(models_dir).items():
        model = keras.models.load_model(str(path), compile=False)
        y_true, y_prob = predict_probs(model, test_ds)
        results[name] = {
            "all": evaluate_predictions(y_true, y_prob, CLASS_NAMES),
            "leak_free": evaluate_predictions(y_true[mask], y_prob[mask], CLASS_NAMES),
            "params": model.count_params(),
            "size_mb": path.stat().st_size / 1e6,
            "latency_ms": measure_latency_ms(model),
        }
        (reports_dir / f"metrics_{name}.json").write_text(json.dumps(results[name], indent=2))
        print(f"{name:15s} official acc {results[name]['all']['accuracy']:.3f} | "
              f"leak-free acc {results[name]['leak_free']['accuracy']:.3f}")

    if not results:
        raise FileNotFoundError(f"No trained .h5 models found in {models_dir}")
    table = compare_models(results)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    best = select_best_model(table)
    (models_dir / "best_model.json").write_text(json.dumps({"name": best}))
    print(f"\nBest model for deployment: {best}")
    return table


if __name__ == "__main__":
    from .config import DATA_DIR, MODELS_DIR, REPORTS_DIR

    parser = argparse.ArgumentParser(description="Evaluate and compare trained models")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports_dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--split_mode", choices=["grouped", "official"], default="grouped")
    args = parser.parse_args()
    print(evaluate_saved_models(args.data_dir, args.models_dir, args.reports_dir,
                                args.split_mode).to_string())
