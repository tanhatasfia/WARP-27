# WARP workload-aware energy experiment

This extension adds files; original TRPS, kernel, quantization and MCKP are left intact.

## 1. Environment

Install original dependencies including compiled WARP CUDA kernels.
Also install `scikit-learn scipy numpy transformers nvidia-ml-py` in the inference environment. 
Set PYTHONPATH to include `WARP/quantization` and `WARP/Kernel`.

## 2. Workloads and precision bitmaps

```bash
python scripts/workload_design.py --out workloads.csv
python scripts/sample_diverse_bitmaps.py --blocks 32 --count 38 --out bitmaps_workload.npy
```



## 3. Packed inference energy measurements (requires GPU + packed artifacts)

```bash
python scripts/measure_workloads.py --model meta-llama/Llama-2-7b-hf --packed-dir /YOUR/PACKED/DIR --bitmaps bitmaps_workload.npy --workloads workloads.csv --split train --out energy_train.csv
python scripts/measure_workloads.py --model meta-llama/Llama-2-7b-hf --packed-dir /YOUR/PACKED/DIR --bitmaps bitmaps_workload.npy --workloads workloads.csv --split validation --out energy_val.csv
python scripts/measure_workloads.py --model meta-llama/Llama-2-7b-hf --packed-dir /YOUR/PACKED/DIR --bitmaps bitmaps_workload.npy --workloads workloads.csv --split test --out energy_test.csv
```



## 4. Fit and allocate

```bash
python scripts/fit_workload_surrogate.py --train energy_train.csv --validation energy_val.csv --out workload_surrogate.json --report workload_report.json
python scripts/allocate_workload.py --surrogate workload_surrogate.json --trps-csv trps.csv --prompt-len 421 --new-tokens 637 --avg-bit-budget 3.1 --trps-tolerance 0.02 --out allocation_workload.json
```





## Optional offline reconstruction calibration



```bash
PYTHONPATH=. python scripts/pack_all_precisions.py \
  --model meta-llama/Llama-2-7b-hf --out packed_window16 \
  --calibration-text calibration.txt --calibration-device cuda \
  --window-tokens 16 --window-weight 1
```

