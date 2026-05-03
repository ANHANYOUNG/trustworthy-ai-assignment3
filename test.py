from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import onnxruntime as ort

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message="Tensorflow parser is unavailable.*")

from maraboupy import Marabou, MarabouCore, MarabouUtils  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "model" / "iris_mlp.onnx"
DEFAULT_SAMPLE_PATH = ROOT / "data" / "iris_sample.npz"
DEFAULT_RESULTS_ROOT = ROOT / "results"
INPUT_NAME = "iris_input"
OUTPUT_NAME = "iris_logits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local robustness of the Iris ONNX model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--violation-margin", type=float, default=1e-6)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def default_results_dir(epsilon: float) -> Path:
    epsilon_tag = str(epsilon).replace("-", "neg_").replace(".", "_")
    return DEFAULT_RESULTS_ROOT / f"eps_{epsilon_tag}"


def load_sample(sample_path: Path) -> dict[str, Any]:
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample file: {sample_path}. Run iris_data.py first.")

    sample = np.load(sample_path)
    return {
        "sample_index": int(sample["sample_index"]),
        "raw_input": sample["raw_input"].astype(np.float32),
        "true_label": int(sample["true_label"]),
        "class_names": [str(name) for name in sample["class_names"]],
        "feature_names": [str(name) for name in sample["feature_names"]],
    }


def evaluate_onnx(model_path: Path, raw_input: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logits = session.run([OUTPUT_NAME], {INPUT_NAME: raw_input[None, :]})[0]
    return logits.reshape(-1).astype(np.float64)


# calculate bounds
def set_linf_input_bounds(network: Any, raw_input: np.ndarray, epsilon: float) -> list[tuple[float, float]]:
    # input idx
    input_vars = np.array(network.inputVars[0]).flatten()
    bounds = []
    for value, var in zip(raw_input, input_vars):
        # feat can't be negative
        # lower = value - eps
        lower = max(0.0, float(value) - epsilon)
        # upper = value + eps
        upper = float(value) + epsilon
        network.setLowerBound(int(var), lower)
        network.setUpperBound(int(var), upper)
        bounds.append((lower, upper))
    # ex)
    # x'_0 in [raw_0 - epsilon, raw_0 + epsilon]
    # x'_1 in [raw_1 - epsilon, raw_1 + epsilon]
    # x'_2 in [raw_2 - epsilon, raw_2 + epsilon]
    # x'_3 in [raw_3 - epsilon, raw_3 + epsilon]
    return bounds

# condition for counterexample
def add_misclassification_constraint(
    network: Any,
    predicted_label: int,
    target_label: int,
    violation_margin: float,
) -> None:
    # output = class logits
    output_vars = np.array(network.outputVars[0]).flatten()
    # LE: Less equal
    # counterexample
    # (output[target] >= output[predicted] + violation_margin) = (output[predicted] - output[target] <= -violation_margin)
    equation = MarabouUtils.Equation(MarabouCore.Equation.LE)
    # output[predicted]
    equation.addAddend(1, int(output_vars[predicted_label]))
    # -output[target]
    equation.addAddend(-1, int(output_vars[target_label]))
    # -violation_margin
    equation.setScalar(-violation_margin)
    network.addEquation(equation)


def solve_target(
    model_path: Path,
    raw_input: np.ndarray,
    epsilon: float,
    predicted_label: int,
    target_label: int,
    violation_margin: float,
    timeout: int,
) -> dict[str, Any]:
    network = Marabou.read_onnx(str(model_path), inputNames=[INPUT_NAME], outputNames=[OUTPUT_NAME])
    bounds = set_linf_input_bounds(network, raw_input, epsilon)
    add_misclassification_constraint(network, predicted_label, target_label, violation_margin)

    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)
    start = time.perf_counter()
    exit_code, vals, stats = network.solve(options=options, verbose=False)
    runtime = time.perf_counter() - start

    status = "TIMEOUT" if stats.hasTimedOut() else str(exit_code).upper()
    result: dict[str, Any] = {
        "target_label": target_label,
        "status": status,
        "runtime_seconds": runtime,
        "input_bounds": bounds,
    }

    if status == "SAT":
        input_vars = np.array(network.inputVars[0]).flatten()
        output_vars = np.array(network.outputVars[0]).flatten()
        result["counterexample_input"] = [float(vals[int(var)]) for var in input_vars]
        result["counterexample_logits"] = [float(vals[int(var)]) for var in output_vars]
        result["counterexample_predicted_label"] = int(np.argmax(result["counterexample_logits"]))

    return result


def short_feature_names(feature_names: list[str]) -> list[str]:
    return [
        name.replace(" (cm)", "").replace("sepal ", "sepal\n").replace("petal ", "petal\n")
        for name in feature_names
    ]


def save_summary_plot(results_dir: Path, summary: dict[str, Any], sample: dict[str, Any]) -> Path:
    feature_names = short_feature_names(sample["feature_names"])
    class_names = sample["class_names"]
    raw_input = np.array(sample["raw_input"], dtype=np.float64)
    initial_logits = np.array(summary["initial_logits"], dtype=np.float64)
    predicted_label = int(summary["predicted_label"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    fig.suptitle(f"Iris robustness result: {summary['status']} (epsilon={summary['epsilon']})")

    x = np.arange(len(feature_names))
    if summary["status"] == "SAT":
        counterexample = np.array(summary["counterexample_input"], dtype=np.float64)
        width = 0.36
        axes[0].bar(x - width / 2, raw_input, width, label="original")
        axes[0].bar(x + width / 2, counterexample, width, label="counterexample")
        axes[0].set_ylabel("cm")
        axes[0].set_title("Input features")
        axes[0].legend()
    else:
        bounds = np.array(summary["target_results"][0]["input_bounds"], dtype=np.float64)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        yerr = np.vstack([raw_input - lower, upper - raw_input])
        axes[0].errorbar(x, raw_input, yerr=yerr, fmt="o", capsize=5)
        axes[0].set_ylabel("cm")
        axes[0].set_title("Allowed input range")

    axes[0].set_xticks(x, feature_names)

    class_x = np.arange(len(class_names))
    colors = ["#4c78a8"] * len(class_names)
    colors[predicted_label] = "#59a14f"
    axes[1].bar(class_x, initial_logits, color=colors, alpha=0.75, label="original")
    if summary["status"] == "SAT":
        counter_logits = np.array(summary["counterexample_logits"], dtype=np.float64)
        axes[1].plot(class_x, counter_logits, marker="o", color="#e15759", label="counterexample")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(class_x, class_names, rotation=15)
    axes[1].set_ylabel("logit")
    axes[1].set_title("Class logits")
    axes[1].legend()

    plot_path = results_dir / "summary.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def save_results(results_dir: Path, summary: dict[str, Any], sample: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)

    if summary.get("status") == "SAT":
        adversarial_input = np.array(summary["counterexample_input"], dtype=np.float32)
        adversarial_logits = np.array(summary["counterexample_logits"], dtype=np.float32)
        np.save(results_dir / "adversarial_input.npy", adversarial_input)
        np.save(results_dir / "adversarial_logits.npy", adversarial_logits)

    plot_path = save_summary_plot(results_dir, summary, sample)
    summary["summary_plot"] = str(plot_path.relative_to(ROOT))
    (results_dir / "verification_result.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return plot_path


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    sample_path = args.sample.resolve()
    results_dir = (args.results_dir or default_results_dir(args.epsilon)).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}. Run iris_model.py first.")

    sample = load_sample(sample_path)
    logits = evaluate_onnx(model_path, sample["raw_input"])
    predicted_label = int(np.argmax(logits))
    target_labels = [label for label in range(logits.size) if label != predicted_label]

    print(f"[INFO] Loading model: {model_path.relative_to(ROOT)}")
    print(
        "[INFO] Input sample index: "
        f"{sample['sample_index']} | True label: {sample['true_label']} | Predicted label: {predicted_label}"
    )
    print(f"[INFO] Perturbation radius (epsilon): {args.epsilon}")
    print(f"[INFO] Violation margin: {args.violation_margin}")
    print("[INFO] Running Marabou verification...")

    start = time.perf_counter()
    target_results = [
        solve_target(
            model_path,
            sample["raw_input"],
            args.epsilon,
            predicted_label,
            target,
            args.violation_margin,
            args.timeout,
        )
        for target in target_labels
    ]
    total_runtime = time.perf_counter() - start

    sat_result = next((result for result in target_results if result["status"] == "SAT"), None)
    if sat_result:
        overall_status = "SAT"
    elif any(result["status"] == "TIMEOUT" for result in target_results):
        overall_status = "TIMEOUT"
    elif all(result["status"] == "UNSAT" for result in target_results):
        overall_status = "UNSAT"
    else:
        overall_status = "UNKNOWN"

    summary: dict[str, Any] = {
        "model": str(model_path.relative_to(ROOT)),
        "sample": str(sample_path.relative_to(ROOT)),
        "sample_index": sample["sample_index"],
        "true_label": sample["true_label"],
        "predicted_label": predicted_label,
        "epsilon": args.epsilon,
        "violation_margin": args.violation_margin,
        "status": overall_status,
        "runtime_seconds": total_runtime,
        "property": (
            f"All inputs within L-infinity epsilon={args.epsilon} ball are classified "
            f"as label {predicted_label} up to margin {args.violation_margin}"
        ),
        "initial_logits": logits.tolist(),
        "target_results": target_results,
    }

    if sat_result:
        summary["counterexample_target_label"] = sat_result["target_label"]
        summary["counterexample_input"] = sat_result["counterexample_input"]
        summary["counterexample_logits"] = sat_result["counterexample_logits"]
        summary["counterexample_predicted_label"] = sat_result["counterexample_predicted_label"]

    plot_path = save_results(results_dir, summary, sample)

    print()
    print("=== Verification Result ===")
    print(f"Status  : {overall_status}")
    print(f"Runtime : {total_runtime:.2f} seconds")
    print(f"Property: {summary['property']}")

    for result in target_results:
        print(
            "Target  : "
            f"{result['target_label']} | Status: {result['status']} | Runtime: {result['runtime_seconds']:.2f}s"
        )

    print(f"Result JSON saved to: {results_dir.relative_to(ROOT) / 'verification_result.json'}")
    print(f"Summary plot saved to: {plot_path.relative_to(ROOT)}")

    if sat_result:
        print(f"Counterexample saved to: {results_dir.relative_to(ROOT) / 'adversarial_input.npy'}")
        print(f"Predicted label on counterexample: {sat_result['counterexample_predicted_label']}")


if __name__ == "__main__":
    main()
