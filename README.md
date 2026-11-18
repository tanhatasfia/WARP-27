

Official implementation of **WARP**


## 1. Repository Structure

```text
WARP/
├── quantization/
│   ├── warp_kernel/
│   │   ├── affine.py
│   │   ├── packing.py
│   │   ├── trps.py
│   │   ├── reconstruction.py
│   │   ├── energy.py
│   │   ├── workload_surrogate.py
│   │   ├── pulse.py
│   │   └── model_utils.py
│   ├── scripts/
│   │   ├── calibrate_trps.py
│   │   ├── pack_all_precisions.py
│   │   ├── workload_design.py
│   │   ├── sample_diverse_bitmaps.py
│   │   ├── measure_workloads.py
│   │   ├── fit_workload_surrogate.py
│   │   └── allocate_workload.py
│   └── tests/
└── Kernel/
    ├── warp_kernels/
    ├── csrc/
    │   ├── packed_gemv.cu
    │   ├── packed_gemm.cu
    │   ├── bitpack.cuh
    │   └── bindings.cpp
    ├── scripts/
    └── tests/
```

## 2. Installation

### Quantization and Allocation

```bash
cd WARP/quantization
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

For Windows PowerShell, use:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### CUDA Kernels

```bash
cd WARP/Kernel
pip install -r requirements.txt
pip install -v .
```

Run quantization commands from `WARP/quantization`. Run CUDA kernel commands from `WARP/Kernel`.

## 3. Adjacent-Window Reconstruction and TRPS Calibration



```bash
python scripts/calibrate_trps.py \
    --model meta-llama/Llama-2-7b-hf \
    --samples 500 \
    --seq-len 2048 \
    --group-size 128 \
    --tau 0.10 \
    --lambda-tail 0.25 \
    --lambda-prop 0.25 \
    --calibration-text calibration.txt \
    --calibration-sequences 4 \
    --calibration-tokens 128 \
    --window-tokens 16 \
    --window-weight 1.0 \
    --out trps.csv
```



## 4. Pack W2/W3/W4 Weights



```bash
python scripts/pack_all_precisions.py \
    --model meta-llama/Llama-2-7b-hf \
    --group-size 128 \
    --calibration-text calibration.txt \
    --calibration-sequences 4 \
    --calibration-tokens 128 \
    --window-tokens 16 \
    --window-weight 1.0 \
    --out packed_llama2_7b
```

## 5. Generate Workload Calibration Sets



```bash
python scripts/workload_design.py \
    --out workloads.csv \
    --seed 2027
```



## 6. Sample Mixed-Precision Configurations



```bash
python scripts/sample_diverse_bitmaps.py \
    --blocks 32 \
    --count 38 \
    --seed 2027 \
    --out bitmaps_workload.npy
```

## 7. Measure GPU Energy



### Training Workloads

```bash
python scripts/measure_workloads.py \
    --model meta-llama/Llama-2-7b-hf \
    --packed-dir packed_llama2_7b \
    --bitmaps bitmaps_workload.npy \
    --workloads workloads.csv \
    --split train \
    --warmup 1 \
    --runs 3 \
    --hz 50 \
    --out energy_train.csv
```

### Validation Workloads

```bash
python scripts/measure_workloads.py \
    --model meta-llama/Llama-2-7b-hf \
    --packed-dir packed_llama2_7b \
    --bitmaps bitmaps_workload.npy \
    --workloads workloads.csv \
    --split validation \
    --warmup 1 \
    --runs 3 \
    --hz 50 \
    --out energy_validation.csv
```

### Test Workloads

```bash
python scripts/measure_workloads.py \
    --model meta-llama/Llama-2-7b-hf \
    --packed-dir packed_llama2_7b \
    --bitmaps bitmaps_workload.npy \
    --workloads workloads.csv \
    --split test \
    --warmup 1 \
    --runs 3 \
    --hz 50 \
    --out energy_test.csv
```



## 8. Fit Workload-Aware Energy Surrogate

WARP uses separate ridge-regression predictors for prefill and decode energy.

```bash
python scripts/fit_workload_surrogate.py \
    --train energy_train.csv \
    --validation energy_validation.csv \
    --test energy_test.csv \
    --out workload_surrogate.json \
    --report workload_report.json
```



## 9. PULSE Workload-Aware Precision Allocation



```bash
python scripts/allocate_workload.py \
    --surrogate workload_surrogate.json \
    --trps-csv trps.csv \
    --prompt-len 924 \
    --new-tokens 1000 \
    --avg-bit-budget 3.0 \
    --trps-tolerance <TRPS_TOLERANCE> \
    --out allocation_924_1000.json
```


```bash
python scripts/allocate_workload.py \
    --surrogate workload_surrogate.json \
    --trps-csv trps.csv \
    --prompt-len 128 \
    --new-tokens 470 \
    --avg-bit-budget 3.0 \
    --trps-tolerance <TRPS_TOLERANCE> \
    --out allocation_128_470.json
```



## 10. Packed GPU Kernel Evaluation

### Run Correctness Tests

```bash
cd WARP/Kernel
pytest -q
```

### Decode GEMV Microbenchmark

```bash
python scripts/bench_gemv.py \
    --bits 3 \
    --n 4096 \
    --k 4096 \
    --exhaustive
```

### Prefill GEMM Microbenchmark

```bash
python scripts/bench_prefill.py --m 1024
```






