from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

matplotlib.use("Agg")

WORKSPACE = Path(r"c:/Users/User/Tensorflow/workspace")
MODEL_PATH = WORKSPACE / "training" / "animals_resnet50_multiclass" / "final.keras"
CLASSES_PATH = WORKSPACE / "training" / "animals_resnet50_multiclass" / "classes.json"
WATCH_DIR = WORKSPACE / "testdata"
RESULTS_DIR = WATCH_DIR / "results"
STATE_FILE = RESULTS_DIR / "processed.json"
CSV_FILE = RESULTS_DIR / "predictions.csv"
CURRENT_OUTPUT_PLOT = RESULTS_DIR / "current_output.png"
LOCK_FILE = RESULTS_DIR / ".watch_testdata.lock"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SLEEP_SECONDS = 2
AUTO_POPUP_VISUALS = True
FILE_READY_WAIT_SECONDS = 1.0


def load_processed() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_processed(processed: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")


def ensure_csv_header() -> None:
    if CSV_FILE.exists():
        return
    with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "file", "raw_output", "prediction", "confidence"])


def migrate_csv_schema_if_needed() -> None:
    if not CSV_FILE.exists():
        return

    with CSV_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        ensure_csv_header()
        return

    header = reader[0]
    if header == ["timestamp", "file", "raw_output", "prediction", "confidence"]:
        return

    if header == ["timestamp", "file", "raw_output", "prediction"]:
        migrated_rows = [["timestamp", "file", "raw_output", "prediction", "confidence"]]
        for row in reader[1:]:
            if not row:
                continue
            if len(row) >= 5:
                migrated_rows.append(row[:5])
            elif len(row) == 4:
                migrated_rows.append(row + [""])
            else:
                continue
        with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(migrated_rows)
        print("Migrated predictions.csv to 5-column schema.")
        return

    # Unknown schema: preserve old file and reset a fresh CSV.
    backup = RESULTS_DIR / "predictions_legacy_backup.csv"
    CSV_FILE.replace(backup)
    ensure_csv_header()
    print(f"Unknown CSV schema. Backed up old file to {backup.name} and created a new predictions.csv.")


def load_class_names() -> list[str]:
    if CLASSES_PATH.exists():
        return list(json.loads(CLASSES_PATH.read_text(encoding="utf-8")))
    return sorted([p.name for p in (WORKSPACE / "images" / "train").iterdir() if p.is_dir()])


def predict_image(model: tf.keras.Model, class_names: list[str], image_path: Path) -> tuple[float, str, float]:
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]

    # Multiclass path.
    if np.ndim(probs) > 0 and len(np.atleast_1d(probs)) > 1:
        idx = int(np.argmax(probs))
        pred = class_names[idx] if idx < len(class_names) else str(idx)
        confidence = float(np.max(probs))
        raw = confidence
        return raw, pred, confidence

    # Legacy binary fallback.
    raw = float(np.ravel(probs)[0])
    pred = "CAT" if raw < 0.5 else "NOT_CAT"
    confidence = float(1.0 - raw if pred == "CAT" else raw)
    return raw, pred, confidence


def append_result(image_path: Path, raw: float, pred: str, confidence: float) -> None:
    # Keep only current image output in CSV, so visuals always reset per image.
    with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "file", "raw_output", "prediction", "confidence"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), str(image_path), raw, pred, confidence])


def generate_visuals() -> None:
    if not CSV_FILE.exists():
        return

    df = pd.read_csv(CSV_FILE, engine="python")
    if df.empty:
        return
    if "prediction" not in df.columns:
        return

    # Use latest prediction only.
    df = df.dropna(subset=["prediction"]).copy()
    if df.empty:
        return
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    latest = df.iloc[-1]

    latest_file = Path(str(latest["file"]))
    latest_pred = str(latest["prediction"])
    latest_conf = float(latest["confidence"]) if "confidence" in df.columns and pd.notna(latest["confidence"]) else 0.0

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Current Prediction", fontsize=14)

    if latest_file.exists():
        img = plt.imread(str(latest_file))
        ax_img.imshow(img)
        ax_img.set_title(latest_file.name, fontsize=10)
    else:
        ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center")
    ax_img.axis("off")

    ax_bar.bar([latest_pred], [latest_conf], color="#2ca02c", width=0.25)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Confidence")
    ax_bar.set_title(f"{latest_pred} ({latest_conf:.3f})")
    ax_bar.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(CURRENT_OUTPUT_PLOT, dpi=150)
    plt.close()


def popup_visuals() -> None:
    if not AUTO_POPUP_VISUALS:
        return
    if CURRENT_OUTPUT_PLOT.exists():
        try:
            os.startfile(str(CURRENT_OUTPUT_PLOT))  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"Could not auto-open {CURRENT_OUTPUT_PLOT.name}: {exc}")


def is_file_ready(image_path: Path) -> bool:
    """Process only when file copy/write has settled."""
    try:
        s1 = image_path.stat()
        time.sleep(FILE_READY_WAIT_SECONDS)
        s2 = image_path.stat()
        return (s1.st_size == s2.st_size) and (int(s1.st_mtime) == int(s2.st_mtime))
    except FileNotFoundError:
        return False


def file_sha1(image_path: Path) -> str:
    h = hashlib.sha1()
    with image_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent multiple watcher instances running at the same time.
    lock_fd = None
    try:
        LOCK_FILE.unlink(missing_ok=True)
        lock_fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except Exception:
        pass

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    processed = load_processed()
    ensure_csv_header()
    migrate_csv_schema_if_needed()
    generate_visuals()

    print(f"Watching folder: {WATCH_DIR}")
    print(f"Using model: {MODEL_PATH}")
    print(f"Classes: {class_names}")
    print("Paste images into testdata. Press Ctrl+C to stop.")

    # Debounce repeated events from the same file path.
    last_processed_by_path: dict[str, str] = {}
    watcher_start_time = time.time()

    try:
        while True:
            image_files = [
                p for p in WATCH_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]
            # Only consider newly added/updated files after watcher start.
            image_files = [p for p in image_files if p.stat().st_mtime >= watcher_start_time - 0.5]
            if not image_files:
                time.sleep(SLEEP_SECONDS)
                continue

            # Process only the latest file event.
            image_path = max(image_files, key=lambda p: p.stat().st_mtime)
            if not is_file_ready(image_path):
                time.sleep(SLEEP_SECONDS)
                continue

            key = f"{image_path.resolve()}|{file_sha1(image_path)}"
            path_key = str(image_path.resolve())
            if last_processed_by_path.get(path_key) == key or key in processed:
                time.sleep(SLEEP_SECONDS)
                continue

            try:
                raw, pred, confidence = predict_image(model, class_names, image_path)
                append_result(image_path, raw, pred, confidence)
                processed.add(key)
                last_processed_by_path[path_key] = key
                save_processed(processed)
                generate_visuals()
                popup_visuals()
                print(f"{image_path.name} -> {pred} (score={raw:.6f}, conf={confidence:.4f})")
            except Exception as exc:
                print(f"Failed: {image_path.name} -> {exc}")

            time.sleep(SLEEP_SECONDS)
    finally:
        try:
            os.close(lock_fd)
        except Exception:
            pass
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
