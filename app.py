"""Streamlit app: upload a brain MRI slice, get the predicted tumour type + confidence.

Run from the repository root:   streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets `import src...` work however the app is launched

from src.config import CLASS_NAMES, DISPLAY_NAMES, IMG_SIZE, MODELS_DIR, REPORTS_DIR  # noqa: E402
from src.explain import occlusion_sensitivity, overlay_heatmap  # noqa: E402
from src.inference import (available_models, best_model_name, describe_prediction,  # noqa: E402
                           predict_probs, preprocess_pil)

LOW_CONFIDENCE = 0.60
MIN_SIDE_PX = 64


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    """Load once per server process; compile=False because we only predict."""
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def model_card(name: str):
    """Show held-out metrics for the chosen model if the evaluation step has been run."""
    metrics_file = REPORTS_DIR / f"metrics_{name}.json"
    if not metrics_file.is_file():
        return
    m = json.loads(metrics_file.read_text())
    st.sidebar.markdown("**Held-out performance**")
    st.sidebar.caption(
        f"Leak-free test set (n={m['leak_free']['n']}): "
        f"accuracy {m['leak_free']['accuracy']:.1%}, macro-F1 {m['leak_free']['macro_f1']:.3f}")


def main():
    st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")
    st.title("🧠 Brain Tumor MRI Classifier")
    st.warning(
        "Educational project - **not a medical device**. Predictions can be wrong and must "
        "never replace review by a qualified radiologist.", icon="⚠️")

    models = available_models(MODELS_DIR)
    if not models:
        st.error(
            "No trained models found in `models/`. Train them first "
            "(`python -m src.train --data_dir data`) or copy the `.h5` files from the "
            "notebook run into the `models/` folder, then reload this page.")
        st.stop()

    # ---- sidebar -----------------------------------------------------------
    names = list(models)
    default = best_model_name(MODELS_DIR)
    index = names.index(default) if default in names else 0
    chosen = st.sidebar.selectbox("Model", names, index=index,
                                  help="The recommended model is pre-selected.")
    show_map = st.sidebar.checkbox("Show occlusion-sensitivity map", value=False,
                                   help="Highlights the image regions the model relied on. "
                                        "Takes a few extra seconds.")
    compare_all = st.sidebar.checkbox("Compare all models", value=False)
    model_card(chosen)
    st.sidebar.markdown("---")
    st.sidebar.caption("Classes: " + ", ".join(DISPLAY_NAMES[c] for c in CLASS_NAMES))

    # ---- input -------------------------------------------------------------
    upload = st.file_uploader("Upload a brain MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Upload an MRI slice to get a prediction.")
        return

    try:
        image = Image.open(upload)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image. Please upload a valid JPG or PNG.")
        return
    if min(image.size) < MIN_SIDE_PX:
        st.error(f"Image is too small ({image.size[0]}x{image.size[1]} px). "
                 f"Please upload an image at least {MIN_SIDE_PX} px on each side.")
        return

    # ---- predict -----------------------------------------------------------
    array = preprocess_pil(image)
    try:
        model = load_model(str(models[chosen]))
        probs = predict_probs(model, array[None])[0]
    except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a traceback
        st.error(f"Prediction failed: {exc}")
        return
    result = describe_prediction(probs)

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
        st.caption(f"Resized to {IMG_SIZE[0]}x{IMG_SIZE[1]} for the model.")
    with right:
        st.subheader("Prediction")
        st.metric(label=f"Predicted class ({chosen})", value=result["display"],
                  delta=f"{result['confidence']:.1%} confidence", delta_color="off")
        if result["confidence"] < LOW_CONFIDENCE:
            st.warning("Low confidence - the model is unsure about this image. "
                       "Treat the result with extra caution.")
        st.markdown("**Confidence per class**")
        for label, p in result["ranking"]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{label}: {p:.1%}")
        st.caption("This model only knows four outcomes. It was trained on axial, coronal and "
                   "sagittal slices of the dataset's scanners; images from other sources or "
                   "unrelated pictures can still receive a confident but meaningless answer.")

    if show_map:
        with st.spinner("Computing occlusion map..."):
            top = int(probs.argmax())
            heat = occlusion_sensitivity(lambda b: predict_probs(model, b), array, top,
                                         patch=32, stride=24)
            overlay = overlay_heatmap(array, heat)
        st.subheader("Where the model looked")
        c1, c2 = st.columns(2)
        c1.image((array * 255).astype("uint8"), caption="Model input", use_container_width=True)
        c2.image(overlay, caption=f"Evidence for '{result['display']}' (red = most important)",
                 use_container_width=True)

    if compare_all:
        rows = []
        for name, path in models.items():
            p = predict_probs(load_model(str(path)), array[None])[0]
            d = describe_prediction(p)
            rows.append({"model": name, "prediction": d["display"],
                         "confidence": f"{d['confidence']:.1%}"})
        st.subheader("All models")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if len({r["prediction"] for r in rows}) > 1:
            st.info("The models disagree on this image - another reason for caution.")


if __name__ == "__main__":
    main()
