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

The repository already includes the generated Iris data, ONNX model, and
verification results. To reproduce them from the scripts, run:

```bash
python iris_data.py
python iris_model.py
python test.py --epsilon 0.1
python test.py --epsilon 0.5
python test.py --epsilon 1.0
```

`test.py` writes each run to an epsilon-specific directory under `results/`.

## File Roles

| File or directory | Role |
| --- | --- |
| `problem1.md` | Problem 1 resource-directory exploration report. |
| `iris_data.py` | Prepares the Iris dataset, scaler values, and one fixed sample input. |
| `iris_model.py` | Trains a small Iris MLP and exports it to ONNX. |
| `model/iris_mlp.onnx` | External ONNX model used by Marabou. It is not from Marabou `resources/`. |
| `data/iris_dataset.npz` | Full Iris dataset arrays and scaler parameters. |
| `data/iris_sample.npz` | Sample input used by `test.py`. |
| `test.py` | Runs Marabou local robustness queries on the ONNX model. |
| `results/` | Saved SAT/UNSAT results, counterexamples, and summary plots. |
| `setup.md` | Marabou installation issue and resolution notes. |

## Model And Query

The model is a small fully connected classifier for the Iris dataset:

```text
4 raw Iris features -> 8 ReLU hidden units -> 3 output logits
```

The ONNX graph uses:

```text
MatMul -> Add -> Relu -> MatMul -> Add
```

The model input is raw Iris feature values in centimeters. StandardScaler
preprocessing is folded into the first affine layer during ONNX export, so
Marabou can verify directly around the raw sample input.

`test.py` checks local robustness around sample index `118`, whose true and
predicted label is `2` (`virginica`). For each epsilon, it constrains every
input feature to an L-infinity box around the sample and asks Marabou whether a
different class logit can exceed the predicted class logit by at least `1e-6`.

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

Observed results:

| Epsilon | Status | Interpretation |
| --- | --- | --- |
| `0.1` | `UNSAT` | No class `0` or `1` counterexample exists within +/- `0.1` cm of the sample. |
| `0.5` | `UNSAT` | No class `0` or `1` counterexample exists within +/- `0.5` cm of the sample. |
| `1.0` | `SAT` | A class `1` (`versicolor`) counterexample exists within +/- `1.0` cm of the sample. |

The `SAT` result at epsilon `1.0` does not mean the whole model is unsafe for
all inputs. It means this specific local robustness property fails for the
chosen sample and perturbation radius.

## Results Structure

```text
results/
  eps_0_1/
    verification_result.json
    summary.png
  eps_0_5/
    verification_result.json
    summary.png
  eps_1_0/
    verification_result.json
    summary.png
    adversarial_input.npy
    adversarial_logits.npy
```

`summary.png` visualizes the input bounds or counterexample and the class logits.
For `eps_1_0`, `adversarial_input.npy` stores the counterexample found by
Marabou.
