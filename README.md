# Trustworthy AI Assignment 3

## Environment

Use the assignment environment and install the Python dependencies. This
installs `maraboupy==2.0.0`, which is enough to run `test.py` through the
Marabou Python API.

```bash
conda activate trustworthy-ai-a3
python -m pip install -r requirements.txt
```

Check that the Marabou Python API and ONNX parser are available:

```bash
python -c "from maraboupy import Marabou; Marabou.read_onnx('model/iris_mlp.onnx'); print('Marabou ONNX parser ready')"
```

If you want to use the source-built Marabou clone instead of the pip-installed
package, build it and expose the local `maraboupy` package:

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou
mkdir -p build
cd build
cmake .. -DENABLE_OPENBLAS=OFF
cmake --build . --target MarabouCore -j 4
cmake --build . --target Marabou -j 4
cd ../..

export PYTHONPATH="$PWD/Marabou:${PYTHONPATH}"
export PATH="$PWD/Marabou/build:${PATH}"
```

Then verify the source-built binary:

```bash
Marabou --version
```

The OpenBLAS source-build issue and resolution are documented in `setup.md`.

## Run Order

The repository already includes all generated data, ONNX models, and
verification results. To reproduce them from the scripts, run:

**Iris (4-dim input):**

```bash
python iris_data.py
python iris_model.py
python test.py --epsilon 0.1
python test.py --epsilon 0.5
python test.py --epsilon 1.0
```

**MNIST narrow (784→32→10):**

```bash
python mnist_model.py   # downloads MNIST on first run, trains, exports ONNX
python test.py --model model/mnist_mlp.onnx --sample data/mnist_sample.npz --epsilon 1.0  --results-dir results/mnist_eps_1_0
python test.py --model model/mnist_mlp.onnx --sample data/mnist_sample.npz --epsilon 5.0  --results-dir results/mnist_eps_5_0
python test.py --model model/mnist_mlp.onnx --sample data/mnist_sample.npz --epsilon 20.0 --results-dir results/mnist_eps_20_0
```

**MNIST wide (784→128→10):**

```bash
python mnist_wide_model.py   # reuses cached MNIST download
python test.py --model model/mnist_wide_mlp.onnx --sample data/mnist_sample.npz --epsilon 1.0  --timeout 60 --results-dir results/mnist_wide_eps_1_0
python test.py --model model/mnist_wide_mlp.onnx --sample data/mnist_sample.npz --epsilon 5.0  --timeout 60 --results-dir results/mnist_wide_eps_5_0
python test.py --model model/mnist_wide_mlp.onnx --sample data/mnist_sample.npz --epsilon 10.0 --timeout 60 --results-dir results/mnist_wide_eps_10_0
```

`test.py` writes each run to an epsilon-specific directory under `results/`.

## File Roles

| File or directory | Role |
| --- | --- |
| `problem1.md` | Problem 1 resource-directory exploration report. |
| `iris_data.py` | Prepares the Iris dataset, scaler values, and one fixed sample input. |
| `iris_model.py` | Trains a small Iris MLP and exports it to ONNX. |
| `model/iris_mlp.onnx` | Iris ONNX model (4→8→3 ReLU MLP). |
| `data/iris_dataset.npz` | Full Iris dataset arrays and scaler parameters. |
| `data/iris_sample.npz` | Iris sample input used by `test.py`. |
| `mnist_model.py` | Downloads MNIST, trains a small MLP (784→32→10), and exports to ONNX. |
| `model/mnist_mlp.onnx` | MNIST narrow ONNX model (784→32→10 ReLU MLP). |
| `data/mnist_sample.npz` | MNIST sample input used by `test.py`. |
| `mnist_wide_model.py` | Trains a wide MNIST MLP (784→128→10) and exports to ONNX. |
| `model/mnist_wide_mlp.onnx` | MNIST wide ONNX model (784→128→10 ReLU MLP). |
| `test.py` | Runs Marabou local robustness queries on any supported ONNX model. |
| `results/` | Saved SAT/UNSAT results, counterexamples, and summary plots. |
| `setup.md` | Marabou installation issue and resolution notes. |

## Models And Queries

### Iris (4-dim)

```text
4 raw Iris features -> 8 ReLU hidden units -> 3 output logits
```

Raw centimeter values. StandardScaler preprocessing is folded into the first
affine layer. `test.py` checks local robustness around sample index `118`
(true label: `2` / virginica). Epsilon is in centimeters.

### MNIST narrow (784-dim)

```text
784 raw pixel values [0, 255] -> 32 ReLU hidden units -> 10 output logits
```

### MNIST wide (784-dim)

```text
784 raw pixel values [0, 255] -> 128 ReLU hidden units -> 10 output logits
```

Both MNIST models use raw pixel values. The `pixel / 255` normalization is
folded into the first affine layer. `test.py` checks local robustness around
sample index `0` of the test set (true label: `7`). Epsilon is in raw pixel
units.

Both models share the same ONNX graph structure:

```text
MatMul -> Add -> Relu -> MatMul -> Add
```

`test.py` auto-detects input/output names from the ONNX model, so it works
for both without any flags beyond `--model` and `--sample`.

## Result Interpretation

Marabou is asked for an unsafe counterexample:

```text
target_logit >= predicted_logit + 1e-6
```

Therefore:

| Result | Meaning in this experiment |
| --- | --- |
| `UNSAT` | No counterexample was found inside the chosen L-infinity box. The sample is locally robust for that epsilon. |
| `SAT` | A counterexample exists inside the chosen L-infinity box. The robustness property is violated. |
| `TIMEOUT` | Marabou did not finish within the timeout. This is not a proof of robustness. |

Observed results — Iris (epsilon in cm):

| Epsilon | Status | Runtime | Interpretation |
| --- | --- | --- | --- |
| `0.1` | `UNSAT` | 0.002s | No counterexample within ±0.1 cm. |
| `0.5` | `UNSAT` | 0.002s | No counterexample within ±0.5 cm. |
| `1.0` | `SAT` | 0.004s | Class `1` (versicolor) counterexample found within ±1.0 cm. |

Observed results — MNIST narrow 784→32→10 (epsilon in raw pixel units):

| Epsilon | Status | Runtime | Interpretation |
| --- | --- | --- | --- |
| `1.0` | `UNSAT` | 0.33s | No counterexample within ±1 pixel. |
| `5.0` | `UNSAT` | 0.54s | No counterexample within ±5 pixels. |
| `20.0` | `SAT` | 1.01s | Class `8` counterexample found within ±20 pixels. |

Observed results — MNIST wide 784→128→10 (epsilon in raw pixel units):

| Epsilon | Status | Runtime | Interpretation |
| --- | --- | --- | --- |
| `1.0` | `UNSAT` | 1.66s | No counterexample within ±1 pixel. |
| `5.0` | `UNSAT` | 9.43s | No counterexample within ±5 pixels; target 3 alone took 7.57s. |
| `10.0` | `SAT + TIMEOUT` | 145.9s | Class `3` counterexample found; targets `8` and `9` timed out at 60s. |

The `SAT` result does not mean the whole model is unsafe for all inputs. It
means this specific local robustness property fails for the chosen sample and
perturbation radius.

The runtime difference reflects the verification cost of increasing ReLU node
count: Iris (8 nodes) ~0.002s → MNIST narrow (32 nodes) ~0.3–1.0s → MNIST
wide (128 nodes) ~1.7–145s with partial TIMEOUT.

## Results Structure

```text
results/
  eps_0_1/                     Iris ε=0.1  (UNSAT)
  eps_0_5/                     Iris ε=0.5  (UNSAT)
  eps_1_0/                     Iris ε=1.0  (SAT)
    adversarial_input.npy
    adversarial_logits.npy
  mnist_eps_1_0/               MNIST narrow ε=1.0   (UNSAT)
  mnist_eps_5_0/               MNIST narrow ε=5.0   (UNSAT)
  mnist_eps_20_0/              MNIST narrow ε=20.0  (SAT)
    adversarial_input.npy
    adversarial_logits.npy
  mnist_wide_eps_1_0/          MNIST wide ε=1.0     (UNSAT)
  mnist_wide_eps_5_0/          MNIST wide ε=5.0     (UNSAT)
  mnist_wide_eps_10_0/         MNIST wide ε=10.0    (SAT + TIMEOUT)
    adversarial_input.npy
    adversarial_logits.npy
```

Each directory contains `verification_result.json` and `summary.png`.
Directories with a `SAT` result also contain `adversarial_input.npy` and
`adversarial_logits.npy`.
