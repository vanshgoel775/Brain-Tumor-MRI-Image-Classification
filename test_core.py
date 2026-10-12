"""Unit tests for the TensorFlow-free parts of the project. Run with: pytest -q"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data import (build_dataframe, flag_leakage, get_class_weights, make_grouped_split,
                      source_id_from_filename)
from src.evaluate import (compare_models, evaluate_predictions, select_best_model,
                          wilson_ci)
from src.explain import occlusion_sensitivity, overlay_heatmap
from src.inference import describe_prediction, preprocess_pil


def _touch(root: Path, split: str, cls: str, name: str):
    folder = root / split / cls
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(folder / name)


def test_source_id_parsing():
    assert source_id_from_filename("Tr-gl_0352_jpg.rf.abc123.jpg") == "gl_0352"
    assert source_id_from_filename("Tr-no_0001_jpg.rf.zzz.jpg") == "no_0001"
    assert source_id_from_filename("random_name.jpg") == "random_name.jpg"  # never a false match


def test_leakage_flags():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root, "train", "glioma", "Tr-gl_0001_jpg.rf.a.jpg")
        _touch(root, "train", "glioma", "Tr-gl_0002_jpg.rf.a.jpg")
        _touch(root, "valid", "glioma", "Tr-gl_0001_jpg.rf.b.jpg")   # twin of a train scan
        _touch(root, "valid", "glioma", "Tr-gl_0003_jpg.rf.b.jpg")   # clean
        _touch(root, "test", "glioma", "Tr-gl_0002_jpg.rf.c.jpg")    # twin of a train scan
        _touch(root, "test", "glioma", "Tr-gl_0003_jpg.rf.c.jpg")    # twin of a valid scan
        _touch(root, "test", "glioma", "Tr-gl_0004_jpg.rf.c.jpg")    # clean
        df = flag_leakage(build_dataframe(root)).set_index(["split", "source_id"])
        assert df.loc[("train", "gl_0001"), "leak_free"].all()
        assert not df.loc[("valid", "gl_0001"), "leak_free"]
        assert df.loc[("valid", "gl_0003"), "leak_free"]
        assert not df.loc[("test", "gl_0002"), "leak_free"]
        assert not df.loc[("test", "gl_0003"), "leak_free"]
        assert df.loc[("test", "gl_0004"), "leak_free"]


def test_class_weights_balanced():
    w = get_class_weights(np.array([0] * 6 + [1] * 2 + [2] * 2))
    assert w[1] > w[0] and abs(w[1] - w[2]) < 1e-9
    assert abs(sum(w[c] * n for c, n in {0: 6, 1: 2, 2: 2}.items()) - 10) < 1e-9


def test_preprocess_shape_range_and_modes():
    for mode in ("L", "RGB", "RGBA", "P"):
        arr = preprocess_pil(Image.new(mode, (640, 480)))
        assert arr.shape == (224, 224, 3) and arr.dtype == np.float32
        assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_occlusion_peaks_on_the_informative_region():
    img = np.zeros((224, 224, 3), dtype=np.float32)
    img[80:120, 100:140, :] = 1.0                      # the only informative patch

    def fake_predict(batch):                           # "class 1" = brightness of that square
        p1 = batch[:, 80:120, 100:140, :].mean(axis=(1, 2, 3))
        return np.stack([1 - p1, p1], axis=1)

    heat = occlusion_sensitivity(fake_predict, img, target_class=1, patch=32, stride=16)
    assert heat.shape == (224, 224) and heat.max() == 1.0
    y, x = np.unravel_index(heat.argmax(), heat.shape)
    assert 80 - 16 <= y <= 120 + 16 and 100 - 16 <= x <= 140 + 16
    assert heat[:40, :40].max() == 0.0                 # far-away corner carries no evidence
    assert overlay_heatmap(img, heat).dtype == np.uint8


def test_metrics_and_ci():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[[0, 0, 1, 2, 2, 2]]              # one class-1 sample mispredicted
    m = evaluate_predictions(y_true, probs, ["a", "b", "c"])
    assert abs(m["accuracy"] - 5 / 6) < 1e-9
    assert m["per_class"]["b"]["recall"] == 0.5 and m["per_class"]["c"]["precision"] == 2 / 3
    lo, hi = wilson_ci(5, 6)
    assert lo < 5 / 6 < hi and 0 <= lo and hi <= 1
    assert wilson_ci(0, 0)[0] != wilson_ci(0, 0)[0]    # NaN for n = 0


def _fake_result(f1, params):
    block = {"n": 100, "accuracy": f1, "macro_f1": f1, "macro_precision": f1, "macro_recall": f1,
             "accuracy_ci": [f1 - 0.05, f1 + 0.05]}
    return {"all": block, "leak_free": block, "params": params, "size_mb": 1.0, "latency_ms": 1.0}


def test_model_selection_prefers_smaller_model_inside_tolerance():
    table = compare_models({"big": _fake_result(0.960, 25_000_000),
                            "small": _fake_result(0.957, 2_000_000),
                            "bad": _fake_result(0.900, 1_000_000)})
    assert select_best_model(table) == "small"          # within 0.005 of the best -> cheaper wins
    table = compare_models({"big": _fake_result(0.960, 25_000_000),
                            "small": _fake_result(0.940, 2_000_000)})
    assert select_best_model(table) == "big"            # clearly better -> accuracy wins


def test_describe_prediction():
    d = describe_prediction(np.array([0.1, 0.2, 0.1, 0.6]))
    assert d["label"] == "pituitary" and abs(d["confidence"] - 0.6) < 1e-9
    assert d["ranking"][0][0] == "Pituitary tumor"


def test_grouped_split_has_no_scan_overlap_and_keeps_classes_balanced():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for cls, prefix in (("glioma", "gl"), ("pituitary", "pi")):
            for scan in range(40):
                for variant in range(1 + scan % 3):          # 1-3 augmented variants per scan
                    _touch(root, "train", cls, f"Tr-{prefix}_{scan:04d}_jpg.rf.v{variant}.jpg")
        df = make_grouped_split(flag_leakage(build_dataframe(root)))
        ids = {s: set(df.loc[df["split"] == s, "source_id"]) for s in ("train", "valid", "test")}
        assert not (ids["train"] & ids["valid"]) and not (ids["train"] & ids["test"])
        assert not (ids["valid"] & ids["test"])
        assert df["leak_free"].all() and len(df) == len(build_dataframe(root))
        assert set(df.loc[df["split"] == "test", "class_name"]) == {"glioma", "pituitary"}
        assert df["split"].value_counts(normalize=True)["train"] > 0.6


def test_compare_models_adds_unfiltered_columns_only_when_filter_removed_images():
    same = compare_models({"a": _fake_result(0.9, 1_000_000)})
    assert "acc_all_test" not in same.columns
    r = _fake_result(0.9, 1_000_000)
    r["all"] = {**r["all"], "n": 120}
    assert "acc_all_test" in compare_models({"a": r}).columns
