"""Dataset indexing, leakage detection and the tf.data input pipeline.

The dataset (Roboflow "Labeled MRI Brain Tumor Dataset") ships pre-split into
train/ valid/ test/, one sub-folder per class. File names look like

    Tr-gl_0352_jpg.rf.ac41....jpg
       |  |
       |  +-- id of the ORIGINAL scan
       +----- class prefix (gl, me, no, pi)

Roboflow exported several augmented variants (flips, small shifts) of the same
original scan, and the export did not keep those variants inside one split.
`flag_leakage` finds the images whose original scan also appears in a split
they should not share with.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_SOURCE_RE = re.compile(r"^Tr-([a-z]+)_(\d+)_", re.IGNORECASE)


def source_id_from_filename(filename: str) -> str:
    """Return the id of the original scan, e.g. 'gl_0352'.

    Falls back to the full file name when the pattern does not match, which
    means "treat this image as its own scan" (never creates false leakage).
    """
    match = _SOURCE_RE.match(filename)
    if not match:
        return filename
    return f"{match.group(1).lower()}_{match.group(2)}"


def build_dataframe(data_dir: str | Path) -> pd.DataFrame:
    """Index every image under data_dir/{train,valid,test}/{class}/ into a DataFrame."""
    data_dir = Path(data_dir)
    rows = []
    for split in ("train", "valid", "test"):
        for label, class_name in enumerate(CLASS_NAMES):
            folder = data_dir / split / class_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "label": label,
                            "filename": path.name,
                            "path": str(path),
                            "source_id": source_id_from_filename(path.name),
                        }
                    )
    if not rows:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Expected {data_dir}/train/<class>/*.jpg"
        )
    return pd.DataFrame(rows)


def flag_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean column `leak_free`.

    * train rows are always leak-free (they define what the model has seen);
    * a valid row is leak-free if its original scan is NOT in train;
    * a test row is leak-free if its original scan is in NEITHER train nor valid.
    """
    df = df.copy()
    train_ids = set(df.loc[df["split"] == "train", "source_id"])
    valid_ids = set(df.loc[df["split"] == "valid", "source_id"])

    in_train = df["source_id"].isin(train_ids)
    in_valid = df["source_id"].isin(valid_ids)

    df["leak_free"] = True
    df.loc[df["split"] == "valid", "leak_free"] = ~in_train[df["split"] == "valid"]
    df.loc[df["split"] == "test", "leak_free"] = ~(in_train | in_valid)[df["split"] == "test"]
    return df


def make_grouped_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Re-split ALL images ~70/15/15 so no original scan is shared between splits.

    Stratified by class and grouped by `source_id` (StratifiedGroupKFold), which keeps
    every augmented variant of a scan together. The dataset's own split is kept in
    column `official_split`; `split` holds the new assignment and `leak_free` is True
    everywhere. Note: this removes duplicate-scan leakage, but the dataset has no patient
    id, so different slices of one patient can still land in different splits.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy().reset_index(drop=True)
    df["official_split"] = df["split"]

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)   # 1/7 -> test
    rest_idx, test_idx = next(outer.split(df, df["label"], df["source_id"]))
    rest = df.iloc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)   # 1/6 of rest -> valid
    _, val_pos = next(inner.split(rest, rest["label"], rest["source_id"]))

    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"
    df.loc[rest.index[val_pos], "split"] = "valid"
    df["leak_free"] = True
    return df


def load_dataframe(data_dir, split_mode: str = "grouped") -> pd.DataFrame:
    """Index the dataset and apply the chosen evaluation protocol.

    split_mode = "grouped"  : leak-free re-split by original scan (default, recommended)
    split_mode = "official" : the dataset's own train/valid/test, with a `leak_free` flag
                              so results can also be reported on the clean subset
    """
    df = flag_leakage(build_dataframe(data_dir))
    if split_mode == "grouped":
        return make_grouped_split(df)
    if split_mode == "official":
        return df
    raise ValueError("split_mode must be 'grouped' or 'official'")


def get_class_weights(labels: np.ndarray) -> dict[int, float]:
    """'Balanced' class weights: n_samples / (n_classes * count_of_class)."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def make_dataset(paths, labels, training: bool, batch_size: int = BATCH_SIZE,
                 img_size=IMG_SIZE, augmentation=None):
    """Build a batched tf.data pipeline: decode -> resize -> [0,1] -> (augment).

    * `antialias=True` makes tf.image.resize behave like PIL's BILINEAR resize,
      so images preprocessed in the Streamlit app match training preprocessing.
    * Decoded images are cached in RAM (about 150 KB each at 224x224x3).
    * `augmentation` (a Keras Sequential) is applied after batching and only
      when training=True, so every image in a batch gets its own random transform.
    """
    import tensorflow as tf  # imported lazily so this module is usable without TF

    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, antialias=True)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.clip_by_value(img, 0.0, 1.0)  # resize ringing can overshoot slightly
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
