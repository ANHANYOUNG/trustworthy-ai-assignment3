from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "mnist_mlp.onnx"
SAMPLE_PATH = ROOT / "data" / "mnist_sample.npz"
METADATA_PATH = MODEL_DIR / "mnist_model_metadata.json"

INPUT_NAME = "mnist_input"
OUTPUT_NAME = "mnist_logits"
HIDDEN_NAME = "hidden_relu"

CLASS_NAMES = [str(i) for i in range(10)]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# input: 784 raw pixel values [0, 255]  (pixel/255 normalization is folded into W0)
# layer 1: 32 hidden units, ReLU activation
# output layer: 10 logits for digits 0-9
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
        "mnist_mlp",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 784])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 10])],
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

    print("[INFO] Loading MNIST dataset (downloads on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data.astype(np.float32)   # (70000, 784), raw pixel values [0, 255]
    y = mnist.target.astype(np.int64)   # (70000,)

    # Standard 60k/10k split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0

    print("[INFO] Training MNIST MLP (784->32->10)...")
    classifier = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=30,
        random_state=42,
        verbose=False,
    )
    classifier.fit(X_train_norm, y_train)

    # Fold pixel/255 normalization into first layer:
    # hidden = (raw_pixel / 255) @ W0 + b0 = raw_pixel @ (W0 / 255) + b0
    # Bias is unchanged because the normalization has no mean offset.
    raw_W0 = classifier.coefs_[0] / 255.0  # (784, 32)
    raw_b0 = classifier.intercepts_[0]      # (32,)
    raw_weights = [raw_W0, classifier.coefs_[1]]
    raw_biases = [raw_b0, classifier.intercepts_[1]]

    logits_test = relu(X_test @ raw_weights[0] + raw_biases[0]) @ raw_weights[1] + raw_biases[1]
    predictions_test = np.argmax(logits_test, axis=1)
    accuracy = accuracy_score(y_test, predictions_test)

    model = build_onnx_model(raw_weights, raw_biases)
    onnx.save(model, MODEL_PATH)

    # Verify ONNX outputs match numpy on a subset of test samples
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    onnx_logits = np.vstack([
        session.run([OUTPUT_NAME], {INPUT_NAME: row[None, :].astype(np.float32)})[0]
        for row in X_test[:200]
    ])
    np.testing.assert_allclose(onnx_logits, logits_test[:200], rtol=1e-4, atol=1e-4)

    # Pick the first correctly classified test sample for verification
    correct_mask = predictions_test == y_test
    sample_idx = int(np.where(correct_mask)[0][0])
    sample_raw = X_test[sample_idx].astype(np.float32)   # raw pixels [0, 255]
    sample_label = int(y_test[sample_idx])
    sample_predicted = int(predictions_test[sample_idx])

    np.savez(
        SAMPLE_PATH,
        sample_index=sample_idx,
        raw_input=sample_raw,
        true_label=sample_label,
        class_names=np.array(CLASS_NAMES),
        feature_names=np.array([f"pixel_{i}" for i in range(784)]),
    )

    metadata = {
        "model": "784-32-10 ReLU MLP",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "onnx_input": INPUT_NAME,
        "onnx_output": OUTPUT_NAME,
        "onnx_ops": ["MatMul", "Add", "Relu", "MatMul", "Add"],
        "input_space": "raw MNIST pixel values [0, 255]; pixel/255 normalization folded into first affine layer",
        "test_accuracy": float(accuracy),
        "sample_index_in_test": sample_idx,
        "sample_true_label": sample_label,
        "sample_predicted_label": sample_predicted,
        "sample_logits": logits_test[sample_idx].astype(float).tolist(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Saved model : {MODEL_PATH.relative_to(ROOT)}")
    print(f"[INFO] Saved sample: {SAMPLE_PATH.relative_to(ROOT)}")
    print(f"[INFO] Test accuracy: {accuracy:.4f}")
    print(f"[INFO] Sample index (in test set): {sample_idx} | true: {sample_label} | predicted: {sample_predicted}")


if __name__ == "__main__":
    main()
