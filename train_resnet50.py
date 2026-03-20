from __future__ import annotations

import json
import os
import sys

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE    = Path(r"c:/Users/User/Tensorflow/workspace")
TRAIN_DIR    = WORKSPACE / "images" / "train"
VAL_DIR      = WORKSPACE / "images" / "test"
OUTPUT_DIR   = WORKSPACE / "training" / "animals_resnet50_multiclass"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH    = OUTPUT_DIR / "best.keras"
FINAL_PATH   = OUTPUT_DIR / "final.keras"
CLASSES_PATH = OUTPUT_DIR / "classes.json"
HISTORY_PATH = OUTPUT_DIR / "history.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
PHASE1_EPOCHS   = 5
PHASE2_EPOCHS   = 10
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-5
UNFREEZE_LAST_N = 30
SEED            = 42


# ---------------------------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------------------------
def make_dataset(directory: Path, shuffle: bool) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        str(directory),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


print("Loading datasets ...")
train_ds_raw = make_dataset(TRAIN_DIR, shuffle=True)
val_ds_raw   = make_dataset(VAL_DIR,   shuffle=False)

class_names: list[str] = train_ds_raw.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(images, labels):
    return preprocess_input(tf.cast(images, tf.float32)), labels

train_ds = train_ds_raw.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds_raw.map(preprocess,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# 2. Build model
# ---------------------------------------------------------------------------
print("Building model ...")
base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False  # Phase 1: freeze all base layers

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model   = Model(inputs, outputs)

model.summary(line_length=80)


# ---------------------------------------------------------------------------
# 3. Callbacks factory
# ---------------------------------------------------------------------------
def make_callbacks() -> list:
    return [
        ModelCheckpoint(
            str(BEST_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Phase 1 - train head only
# ---------------------------------------------------------------------------
print("\n-- Phase 1: Training classification head (frozen base) --")
model.compile(
    optimizer=tf.keras.optimizers.Adam(PHASE1_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=make_callbacks(),
)


# ---------------------------------------------------------------------------
# 5. Phase 2 - fine-tune last N base layers
# ---------------------------------------------------------------------------
print(f"\n-- Phase 2: Fine-tuning last {UNFREEZE_LAST_N} base layers --")
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST_N]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(PHASE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=make_callbacks(),
)


# ---------------------------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------------------------
print("\nSaving artifacts ...")

model.save(str(FINAL_PATH))
print(f"  Saved final model  -> {FINAL_PATH}")

CLASSES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
print(f"  Saved classes      -> {CLASSES_PATH}")

def merge_histories(*hists):
    merged: dict[str, list] = {}
    for h in hists:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged

full_history = merge_histories(history1, history2)
HISTORY_PATH.write_text(json.dumps(full_history, indent=2), encoding="utf-8")
print(f"  Saved history      -> {HISTORY_PATH}")

print("Evaluating on validation set ...")
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
metrics = {"val_loss": float(val_loss), "val_accuracy": float(val_acc)}
METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved metrics      -> {METRICS_PATH}")

print(f"\nTraining complete!  val_accuracy={val_acc:.4f}  val_loss={val_loss:.4f}")
print(f"Model saved to: {FINAL_PATH}")
print("Run watch_testdata.py to test live inference.")
