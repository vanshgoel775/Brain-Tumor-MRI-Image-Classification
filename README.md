# Brain Tumor MRI Image Classification

Classify brain MRI slices into **glioma, meningioma, pituitary tumour or no tumour** with a custom CNN and
three transfer-learning models (MobileNetV2, ResNet50V2, EfficientNetB0), compare them against a classical
HOG + SVM baseline, and serve the best one through a **Streamlit** app.

> **Educational project - not a medical device.** Predictions must never replace review by a radiologist.

## Two things you should know about the dataset

I audited the data before training (`notebooks/…ipynb`, Sections 1, 4 and 5; charts in `reports/eda/`):

1. **The official train/valid/test split leaks.** The Roboflow export scattered flipped/shifted variants of the
   same original scan across splits. **97 of 502 validation and 67 of 246 test images** share their original scan
   with another split - and **all of them are gliomas** (84% of the official test gliomas). Only 13 of the 80 test
   gliomas are genuinely unseen. By default this project therefore **re-splits by original scan**
   (70/15/15, stratified by class, grouped by scan id → 1,745 / 349 / 349 images, zero shared scans).
   The official split is still available (`--split_mode official`), evaluated on the full test set *and* on its
   leak-free subset.
2. **Classes differ in how they were acquired, not only in pathology** (orientation, MRI sequence, framing,
   compression). A random forest that sees only **seven global brightness/texture numbers - no spatial
   information** - classifies ~87% of the official test set correctly (32.5% for always guessing the majority
   class), and 100% of the 67 leaked test images. Deep-model accuracy must be read against this floor, and
   external validation is needed before any clinical claim.

Remaining limitation: the dataset has no patient id, so slices of one patient can still fall in different splits.

## Verified result (reproducible without a GPU)

| Model | Test accuracy (95% CI) | Macro-F1 | Protocol |
|---|---|---|---|
| HOG + RBF-SVM baseline | 92.0% (88.6 – 94.4%) | 0.913 | grouped split, n = 349 |

The deep-model results (custom CNN, MobileNetV2, ResNet50V2, EfficientNetB0) are produced by running the notebook
or `python -m src.train` on a GPU; they are written to `reports/model_comparison.csv`. They are **not** listed here
because they have to come from your own run.

## Repository layout

```
├── notebooks/Brain_Tumor_MRI_Classification.ipynb   # full, self-contained submission notebook (template structure)
├── src/
│   ├── config.py      constants (image size, class names, paths)
│   ├── data.py        indexing, leakage detection, grouped split, class weights, tf.data pipeline
│   ├── eda.py         image-level statistics used for EDA / hypothesis tests
│   ├── models.py      custom CNN, transfer-learning builders, augmentation, fine-tuning helpers
│   ├── train.py       two-stage training, random search, CLI
│   ├── evaluate.py    metrics, plots, model comparison + selection rule, CLI
│   ├── explain.py     occlusion-sensitivity maps
│   └── inference.py   preprocessing + prediction helpers shared with the app
├── app/streamlit_app.py
├── tests/test_core.py # 10 unit tests (no TensorFlow needed)
├── reports/eda/       # 17 EDA charts
├── models/            # *.h5 and best_model.json go here (git-ignored: files can exceed 100 MB)
└── requirements.txt
```

## Quick start

### Option A - Google Colab (recommended, free GPU)
1. Open `notebooks/Brain_Tumor_MRI_Classification.ipynb` in Colab, set *Runtime → GPU*.
2. Upload the dataset zip (`drive-download-…zip`) to the Colab files panel.
3. *Run all.* Set `FAST_MODE = True` first for a ~5-minute pipeline check. A full run takes roughly 30-45 min on a T4.
4. Download `trained_models.zip` (the notebook offers it) and unzip it into `models/`.

### Option B - local
```bash
pip install -r requirements.txt
unzip drive-download-*.zip -d data            # must give data/train, data/valid, data/test
python -m src.train --data_dir data           # trains the 4 models -> models/*.h5
python -m src.evaluate --data_dir data        # comparison table + models/best_model.json
streamlit run app/streamlit_app.py
pytest -q
```
Useful flags: `--models mobilenetv2`, `--epochs 5`, `--split_mode official`.

## Method summary
* **Preprocessing:** resize to 224×224 (anti-aliased, no crop), scale to [0, 1]. Transfer models convert to their
  backbone's expected range with a `Rescaling` layer *inside* the model, so every `.h5` takes the same input.
* **Augmentation (training only):** rotation, horizontal/vertical flips, zoom, shifts, brightness, contrast; black fill.
* **Imbalance:** balanced class weights (computed from the training split only).
* **Custom CNN:** 4 × (2 × Conv-BN-ReLU) + pooling + dropout, global average pooling, dense head; tuned by random search.
* **Transfer learning:** frozen ImageNet backbone + new head, then fine-tune the top 40 layers (BatchNorm frozen) at lr 1e-5.
* **Callbacks:** EarlyStopping (best weights restored), ModelCheckpoint, ReduceLROnPlateau.
* **Evaluation:** accuracy with Wilson 95% CI, macro precision/recall/F1, per-class report, confusion matrix,
  size and latency. **Selection rule:** best macro-F1; among models within 0.005, the smallest.
* **Explainability:** occlusion-sensitivity maps (notebook Section 7 and an app checkbox).

## Streamlit app
Upload a JPG/PNG → predicted class, per-class confidence, low-confidence and model-disagreement warnings,
optional occlusion map, optional comparison of all models. The model recommended by the evaluation step is
pre-selected. The app validates file type and size and shows a friendly message if no model is found.

## Known limitations
* Single public dataset from mixed sources; scanner/sequence/plane correlate with class (see above).
* Slice-level, not patient-level, splitting.
* Occlusion maps are coarse and show *where* the model looks, not *why*.
* The model only knows four outcomes and will answer confidently for unrelated images.

Dataset: *Labeled MRI Brain Tumor Dataset* (Roboflow Universe, CC BY 4.0).
