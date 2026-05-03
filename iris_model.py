from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "iris_dataset.npz"
SAMPLE_PATH = ROOT / "data" / "iris_sample.npz"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "iris_mlp.onnx"
METADATA_PATH = MODEL_DIR / "iris_model_metadata.json"

INPUT_NAME = "iris_input"
OUTPUT_NAME = "iris_logits"
HIDDEN_NAME = "hidden_relu"


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

# input: 4 features (sepal length, sepal width, petal length, petal width)
# layer 1: 8 hidden units, ReLU activation
# output layer: 3 logits for 3 classes (setosa, versicolor, virginica)
def build_onnx_model(weights: list[np.ndarray], biases: list[np.ndarray]) -> onnx.ModelProto:
    w0, w1 = [w.astype(np.float32) for w in weights]
    b0, b1 = [b.astype(np.float32) for b in biases]

    initializers = [
        numpy_helper.from_array(w0, name="W0"),
        numpy_helper.from_array(b0, name="b0"),
        numpy_helper.from_array(w1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
    ]
    nodes = [
        helper.make_node("MatMul", [INPUT_NAME, "W0"], ["hidden_linear"], name="hidden_matmul"),
        helper.make_node("Add", ["hidden_linear", "b0"], ["hidden_pre_relu"], name="hidden_add"),
        helper.make_node("Relu", ["hidden_pre_relu"], [HIDDEN_NAME], name="hidden_relu"),
        helper.make_node("MatMul", [HIDDEN_NAME, "W1"], ["output_linear"], name="output_matmul"),
        helper.make_node("Add", ["output_linear", "b1"], [OUTPUT_NAME], name="output_add"),
    ]
    graph = helper.make_graph(
        nodes,
        "iris_mlp",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 3])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="trustworthy-ai-assignment3",
        opset_imports=[helper.make_operatorsetid("", 9)],
    )
    model.ir_version = 4
    onnx.checker.check_model(model)
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = np.load(DATASET_PATH)
    sample = np.load(SAMPLE_PATH)
    raw_inputs = dataset["raw_inputs"].astype(np.float32)
    scaled_inputs = dataset["scaled_inputs"].astype(np.float32)
    labels = dataset["labels"].astype(np.int64)
    scaler_mean = dataset["scaler_mean"].astype(np.float64)
    scaler_scale = dataset["scaler_scale"].astype(np.float64)

    classifier = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=1e-4,
        max_iter=5000,
        random_state=7,
    )
    classifier.fit(scaled_inputs, labels)

    raw_w0 = classifier.coefs_[0] / scaler_scale.reshape(-1, 1)
    raw_b0 = classifier.intercepts_[0] - (scaler_mean / scaler_scale) @ classifier.coefs_[0]
    raw_weights = [raw_w0, classifier.coefs_[1]]
    raw_biases = [raw_b0, classifier.intercepts_[1]]

    logits = relu(raw_inputs @ raw_weights[0] + raw_biases[0]) @ raw_weights[1]
    logits = logits + raw_biases[1]
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)

    model = build_onnx_model(raw_weights, raw_biases)
    onnx.save(model, MODEL_PATH)

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    onnx_logits = np.vstack(
        [session.run([OUTPUT_NAME], {INPUT_NAME: row[None, :]})[0] for row in raw_inputs]
    )
    np.testing.assert_allclose(onnx_logits, logits, rtol=1e-4, atol=1e-4)

    sample_index = int(sample["sample_index"])
    metadata = {
        "model": "4-8-3 ReLU MLP",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "onnx_input": INPUT_NAME,
        "onnx_output": OUTPUT_NAME,
        "onnx_ops": ["MatMul", "Add", "Relu", "MatMul", "Add"],
        "input_space": "raw Iris features in centimeters; StandardScaler folded into first affine layer",
        "accuracy_on_full_iris": float(accuracy),
        "sample_index": sample_index,
        "sample_true_label": int(labels[sample_index]),
        "sample_predicted_label": int(predictions[sample_index]),
        "sample_logits": logits[sample_index].astype(float).tolist(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"[INFO] Saved metadata: {METADATA_PATH.relative_to(ROOT)}")
    print(f"[INFO] Training-set accuracy: {accuracy:.4f}")
    print(
        "[INFO] Sample index: "
        f"{sample_index} | true label: {labels[sample_index]} | predicted label: {predictions[sample_index]}"
    )


if __name__ == "__main__":
    main()
