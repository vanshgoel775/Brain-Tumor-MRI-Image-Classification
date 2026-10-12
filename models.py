"""Model definitions: a custom CNN and three ImageNet transfer-learning models.

Contract shared by ALL models (so the app needs no per-model special-casing):
    input  : float32 image batch (N, 224, 224, 3), pixel values in [0, 1]
    output : softmax probabilities over the 4 classes in config.CLASS_NAMES

Transfer models convert [0, 1] to whatever their backbone was pre-trained on with a
built-in Rescaling layer, so the conversion is saved inside the .h5 file.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import IMG_SIZE, SEED

INPUT_SHAPE = (*IMG_SIZE, 3)

# name -> (keras.applications class name, how to rescale [0,1] inputs for it)
#   "minus1_1": backbone expects [-1, 1]      (MobileNetV2, ResNet50V2)
#   "raw255"  : backbone expects [0, 255] and normalises internally (EfficientNetB0)
TRANSFER_BACKBONES = {
    "mobilenetv2": ("MobileNetV2", "minus1_1"),
    "resnet50v2": ("ResNet50V2", "minus1_1"),
    "efficientnetb0": ("EfficientNetB0", "raw255"),
}


def build_augmentation() -> keras.Sequential:
    """Training-time augmentation (rotation, flips, zoom, shifts, brightness, contrast).

    Fill mode is 'constant' (black) because MRI backgrounds are black; the default
    'reflect' would mirror anatomy into the background and create unrealistic images.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            layers.RandomRotation(0.06, fill_mode="constant", seed=SEED),      # about +-22 deg
            layers.RandomZoom(0.15, fill_mode="constant", seed=SEED),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", seed=SEED),
            layers.RandomBrightness(0.15, value_range=(0.0, 1.0), seed=SEED),
            layers.RandomContrast(0.15, seed=SEED),
        ],
        name="augmentation",
    )


def build_custom_cnn(num_classes: int = 4, base_filters: int = 32,
                     dropout: float = 0.4, input_shape=INPUT_SHAPE) -> keras.Model:
    """4 blocks of (Conv-BN-ReLU) x2 + MaxPool + Dropout, then GAP + dense head.

    BatchNorm stabilises learning; Dropout and global average pooling (instead of a huge
    Flatten->Dense) limit over-fitting on a ~1.7k-image training set.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for block, mult in enumerate([1, 2, 4, 8], start=1):
        for conv in (1, 2):
            x = layers.Conv2D(base_filters * mult, 3, padding="same", use_bias=False,
                              name=f"conv{block}_{conv}")(x)
            x = layers.BatchNormalization(name=f"bn{block}_{conv}")(x)
            x = layers.Activation("relu", name=f"relu{block}_{conv}")(x)
        x = layers.MaxPooling2D(name=f"pool{block}")(x)
        x = layers.Dropout(dropout * 0.5, name=f"drop{block}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, use_bias=False, name="dense")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Activation("relu", name="dense_relu")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(name: str, num_classes: int = 4, dropout: float = 0.3,
                         input_shape=INPUT_SHAPE) -> keras.Model:
    """ImageNet backbone (frozen) + new classification head."""
    if name not in TRANSFER_BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(TRANSFER_BACKBONES)}")
    class_name, rescale_mode = TRANSFER_BACKBONES[name]
    backbone_cls = getattr(keras.applications, class_name)

    inputs = keras.Input(shape=input_shape, name="image")
    if rescale_mode == "minus1_1":
        x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)
    else:
        x = layers.Rescaling(255.0, name="to_0_255")(inputs)

    backbone = backbone_cls(include_top=False, weights="imagenet", input_shape=input_shape)
    backbone.trainable = False                     # stage 1: train the new head only
    x = backbone(x, training=False)                # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop_1")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested pre-trained backbone of a transfer model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No nested backbone found - is this a transfer model?")


def unfreeze_top_layers(model: keras.Model, n_layers: int = 40) -> int:
    """Stage 2: make the last `n_layers` of the backbone trainable.

    BatchNorm layers stay frozen: with a small dataset, updating their running
    statistics tends to hurt more than it helps. Call compile() again afterwards.
    Returns the number of trainable backbone layers.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return sum(layer.trainable for layer in backbone.layers)


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(name: str, **kwargs) -> keras.Model:
    """Single entry point used by the training script and the notebook."""
    if name == "custom_cnn":
        return build_custom_cnn(**kwargs)
    return build_transfer_model(name, **kwargs)
