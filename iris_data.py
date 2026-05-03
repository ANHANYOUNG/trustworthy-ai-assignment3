from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "iris_dataset.npz"
SAMPLE_PATH = DATA_DIR / "iris_sample.npz"
METADATA_PATH = DATA_DIR / "iris_metadata.json"
SAMPLE_INDEX = 118


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    raw_inputs = iris.data.astype(np.float32)
    labels = iris.target.astype(np.int64)

    scaler = StandardScaler()
    scaled_inputs = scaler.fit_transform(raw_inputs).astype(np.float32)

    np.savez(
        DATASET_PATH,
        raw_inputs=raw_inputs,
        scaled_inputs=scaled_inputs,
        labels=labels,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        class_names=np.array(iris.target_names),
        feature_names=np.array(iris.feature_names),
    )

    np.savez(
        SAMPLE_PATH,
        sample_index=np.array(SAMPLE_INDEX, dtype=np.int64),
        raw_input=raw_inputs[SAMPLE_INDEX],
        scaled_input=scaled_inputs[SAMPLE_INDEX],
        true_label=np.array(labels[SAMPLE_INDEX], dtype=np.int64),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        class_names=np.array(iris.target_names),
        feature_names=np.array(iris.feature_names),
    )

    metadata = {
        "dataset": "sklearn.datasets.load_iris",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "sample_index": SAMPLE_INDEX,
        "sample_true_label": int(labels[SAMPLE_INDEX]),
        "input_space": "raw Iris features in centimeters",
        "preprocessing": "StandardScaler parameters are saved for model training/export",
        "class_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Saved dataset: {DATASET_PATH.relative_to(ROOT)}")
    print(f"[INFO] Saved sample: {SAMPLE_PATH.relative_to(ROOT)}")
    print(f"[INFO] Saved metadata: {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
