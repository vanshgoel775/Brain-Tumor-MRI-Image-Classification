"""Training: two-stage transfer learning, custom CNN, random search, and a CLI.

Usage (from the repository root):
    python -m src.train --data_dir data                      # train all four models
    python -m src.train --data_dir data --models mobilenetv2 # just one
    python -m src.evaluate --data_dir data                   # compare + pick the best
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .config import (BATCH_SIZE, DATA_DIR, MODEL_NAMES, MODELS_DIR, REPORTS_DIR, SEED)
from .data import get_class_weights, load_dataframe, make_dataset
from .models import (build_augmentation, build_model, compile_model,
                     unfreeze_top_layers)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def make_callbacks(weights_path: Path, patience: int, monitor: str = "val_loss"):
    """EarlyStopping + ModelCheckpoint (best weights) + ReduceLROnPlateau.

    Checkpoints store weights only, in '<name>.weights.h5' - the one file name that is
    valid in both Keras 2 and Keras 3. The full model is saved to <name>.h5 afterwards.
    """
    return [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(weights_path), monitor=monitor,
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                          patience=max(patience // 2, 1),
                                          min_lr=1e-7, verbose=1),
    ]


def fit_stage(model, train_ds, val_ds, class_weights, epochs, learning_rate,
              weights_path: Path, patience: int, verbose: int = 2) -> dict:
    """Compile and fit one training stage; returns the history dict."""
    compile_model(model, learning_rate)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weights,
                        callbacks=make_callbacks(weights_path, patience),
                        verbose=verbose)
    return history.history


def train_model(name: str, train_ds, val_ds, class_weights: dict,
                models_dir: Path = MODELS_DIR, epochs: int = 40,
                head_epochs: int = 12, finetune_epochs: int = 15,
                finetune_layers: int = 40, learning_rate: float = 1e-3,
                build_kwargs: dict | None = None, verbose: int = 2):
    """Train one model end to end and save it as models/<name>.h5.

    custom_cnn : single stage, `epochs` max (early stopping decides when to stop)
    transfer   : stage 1 trains the new head (lr 1e-3); stage 2 unfreezes the top
                 `finetune_layers` backbone layers and continues at lr 1e-5
    Returns (model, history, finetune_start_epoch or None).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    model = build_model(name, **(build_kwargs or {}))
    weights_path = models_dir / f"{name}.weights.h5"

    if name == "custom_cnn":
        history = fit_stage(model, train_ds, val_ds, class_weights, epochs,
                            learning_rate, weights_path, patience=8, verbose=verbose)
        finetune_start = None
    else:
        history = fit_stage(model, train_ds, val_ds, class_weights, head_epochs,
                            learning_rate, weights_path, patience=4, verbose=verbose)
        finetune_start = len(history["loss"])
        n_trainable = unfreeze_top_layers(model, finetune_layers)
        print(f"[{name}] fine-tuning {n_trainable} backbone layers")
        stage2 = fit_stage(model, train_ds, val_ds, class_weights, finetune_epochs,
                           1e-5, weights_path, patience=5, verbose=verbose)
        history = {k: history[k] + stage2[k] for k in history}

    # EarlyStopping(restore_best_weights=True) already restored the best weights of the
    # last stage; save the complete model in the .h5 format requested by the brief.
    model.save(str(models_dir / f"{name}.h5"))
    return model, history, finetune_start


def random_search_cnn(train_ds, val_ds, class_weights, n_trials: int = 4,
                      epochs: int = 12, verbose: int = 0) -> tuple[pd.DataFrame, dict]:
    """Small random search over custom-CNN hyper-parameters, scored on validation loss.

    k-fold cross-validation would multiply training time by k; with ~1.7k images and a
    fixed validation split a short random search is the practical alternative.
    """
    space = {"learning_rate": [1e-3, 5e-4, 2e-4],
             "dropout": [0.3, 0.4, 0.5],
             "base_filters": [16, 32]}
    rng = random.Random(SEED)
    rows = []
    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in space.items()}
        set_seed(SEED + trial)
        model = build_model("custom_cnn", base_filters=params["base_filters"],
                            dropout=params["dropout"])
        compile_model(model, params["learning_rate"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                         class_weight=class_weights, verbose=verbose,
                         callbacks=[keras.callbacks.EarlyStopping(
                             monitor="val_loss", patience=4, restore_best_weights=True)]
                         ).history
        best = int(np.argmin(hist["val_loss"]))
        rows.append({**params, "val_loss": hist["val_loss"][best],
                     "val_accuracy": hist["val_accuracy"][best], "epochs_run": len(hist["loss"])})
        print(f"trial {trial + 1}/{n_trials}: {rows[-1]}")
    table = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
    best_params = {k: table.loc[0, k] for k in space}
    best_params["base_filters"] = int(best_params["base_filters"])
    return table, best_params


def prepare_data(data_dir: Path, split_mode: str = "grouped", leak_free_validation: bool = True):
    """Build train/val/test tf.data pipelines plus the metadata frame.

    In "official" mode, validation uses only the leak-free images by default so early
    stopping is not driven by images the model has effectively seen in training.
    (In "grouped" mode every image is leak-free, so the filter changes nothing.)
    """
    df = load_dataframe(data_dir, split_mode)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "valid"]
    if leak_free_validation:
        val_df = val_df[val_df["leak_free"]]
    test_df = df[df["split"] == "test"]

    augmentation = build_augmentation()
    train_ds = make_dataset(train_df["path"], train_df["label"], True, BATCH_SIZE,
                            augmentation=augmentation)
    val_ds = make_dataset(val_df["path"], val_df["label"], False, BATCH_SIZE)
    test_ds = make_dataset(test_df["path"], test_df["label"], False, BATCH_SIZE)
    class_weights = get_class_weights(train_df["label"].values)
    return df, train_ds, val_ds, test_ds, class_weights


def main():
    parser = argparse.ArgumentParser(description="Train brain-tumour MRI classifiers")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=40, help="max epochs, custom CNN")
    parser.add_argument("--head_epochs", type=int, default=12)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--split_mode", choices=["grouped", "official"], default="grouped",
                        help="grouped = leak-free re-split by original scan (default)")
    parser.add_argument("--all_validation", action="store_true",
                        help="validate on the full valid split instead of the leak-free part")
    args = parser.parse_args()

    df, train_ds, val_ds, _, class_weights = prepare_data(
        args.data_dir, args.split_mode, leak_free_validation=not args.all_validation)
    print("class weights:", {k: round(v, 3) for k, v in class_weights.items()})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        print(f"\n===== training {name} =====")
        _, history, ft_start = train_model(
            name, train_ds, val_ds, class_weights, args.models_dir, epochs=args.epochs,
            head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs)
        payload = {"history": {k: [float(v) for v in vs] for k, vs in history.items()},
                   "finetune_start": ft_start}
        (REPORTS_DIR / f"history_{name}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
