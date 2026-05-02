# Trustworthy AI Assignment 3

## Environment

```bash
conda activate trustworthy-ai-a3
python -m pip install -r requirements.txt
```

## Marabou Setup Note

I followed the source build instructions in `Marabou/README.md`:

```bash
cd Marabou
mkdir -p build
cd build
cmake ../
cmake --build ./
```

`cmake ../` completed, but the build failed with OpenBLAS enabled:

```text
Marabou/src/engine/DnCManager.cpp:134:5: error: 'openblas_set_num_threads' was not declared in this scope
```

The local OpenBLAS install directory had been created at
`Marabou/tools/OpenBLAS-0.3.19/installed/`, but it did not contain the expected
headers and libraries. Since Problem 2 uses a small fully connected ONNX model,
OpenBLAS acceleration is not required for the planned verification query.

I rebuilt Marabou with OpenBLAS disabled:

```bash
cd Marabou/build
cmake .. -DENABLE_OPENBLAS=OFF
cmake --build . --target MarabouCore -j 4
cmake --build . --target Marabou -j 4
```

The rebuilt binary was checked with:

```bash
cd Marabou
./build/Marabou --version
./build/Marabou resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet \
  resources/properties/acas_property_3.txt --timeout=10 --verbosity=0
```

The README example returned:

```text
unsat
```
