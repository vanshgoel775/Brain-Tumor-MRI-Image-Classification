"""Inference helpers shared by the Streamlit app and the notebook.

Preprocessing here MUST mirror `data.make_dataset`: RGB, resize to 224x224 with an
anti-aliased bilinear filter, scale to [0, 1].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, DISPLAY_NAMES, IMG_SIZE, MODEL_NAMES


def preprocess_pil(img: Image.Image, img_size=IMG_SIZE) -> np.ndarray:
    """PIL image (any mode) -> float32 array (H, W, 3) in [0, 1]."""
    img = img.convert("RGB")                       # handles grayscale / RGBA / palette
    img = img.resize((img_size[1], img_size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def available_models(models_dir: str | Path) -> dict[str, Path]:
    """Map model name -> .h5 file for every trained model found in models_dir."""
    models_dir = Path(models_dir)
    found = {}
    for name in MODEL_NAMES:
        path = models_dir / f"{name}.h5"
        if path.is_file():
            found[name] = path
    return found


def best_model_name(models_dir: str | Path) -> str | None:
    """Name written by `src.evaluate` / the notebook, or None if not available."""
    path = Path(models_dir) / "best_model.json"
    if path.is_file():
        return json.loads(path.read_text()).get("name")
    return None


def predict_probs(model, batch: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) -> (N, n_classes) probabilities."""
    return np.asarray(model.predict(batch, verbose=0))


def describe_prediction(probs: np.ndarray) -> dict:
    """Turn a probability vector into something display-ready."""
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    return {
        "label": CLASS_NAMES[top],
        "display": DISPLAY_NAMES[CLASS_NAMES[top]],
        "confidence": float(probs[top]),
        "ranking": [(DISPLAY_NAMES[CLASS_NAMES[i]], float(probs[i])) for i in order],
    }
