"""Project-wide constants. Everything that other modules need to agree on lives here."""
from pathlib import Path

SEED = 42
IMG_SIZE = (224, 224)          # (height, width) fed to every model
BATCH_SIZE = 32

# Folder names in the dataset, alphabetical. Index in this list == integer label.
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Models trained by default, in the order they are reported.
MODEL_NAMES = ["custom_cnn", "mobilenetv2", "resnet50v2", "efficientnetb0"]
