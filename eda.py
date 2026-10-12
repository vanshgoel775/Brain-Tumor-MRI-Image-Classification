"""Hand-crafted image statistics used for EDA, hypothesis tests and the baseline.

None of these features are used by the deep-learning models (they learn their own
features from pixels). They exist to *understand* the data before modelling.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from skimage.filters import sobel


def load_gray(path: str, size: int = 128) -> np.ndarray:
    """Load an image as float32 grayscale in [0, 1], resized to size x size."""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def image_features(gray: np.ndarray, fg_threshold: float = 0.08) -> dict:
    """Summarise one grayscale image (values in [0, 1]).

    brightness       mean pixel intensity
    contrast         standard deviation of intensities
    skewness         asymmetry of the intensity histogram
    entropy          Shannon entropy of the 64-bin histogram (texture richness)
    foreground_frac  share of pixels brighter than fg_threshold (head/brain area)
    edge_density     mean Sobel gradient magnitude (structure / detail)
    lr_symmetry      correlation between the image and its left-right mirror;
                     healthy axial slices are close to symmetric, large masses
                     reduce it
    """
    hist, _ = np.histogram(gray, bins=64, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    flat = gray.ravel()
    mirrored = gray[:, ::-1].ravel()
    sym = np.corrcoef(flat, mirrored)[0, 1] if flat.std() > 0 else 0.0
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "skewness": float(stats.skew(flat)),
        "entropy": float(-(p * np.log2(p)).sum()),
        "foreground_frac": float((gray > fg_threshold).mean()),
        "edge_density": float(sobel(gray).mean()),
        "lr_symmetry": float(sym),
    }


def compute_image_features(df: pd.DataFrame, size: int = 128) -> pd.DataFrame:
    """Return df plus one column per feature in `image_features` and file_kb."""
    feats = []
    for path in df["path"]:
        feats.append(image_features(load_gray(path, size)))
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    out["file_kb"] = [round(os.path.getsize(p) / 1024, 1) for p in out["path"]]
    return out


FEATURE_COLUMNS = [
    "brightness", "contrast", "skewness", "entropy",
    "foreground_frac", "edge_density", "lr_symmetry", "file_kb",
]
