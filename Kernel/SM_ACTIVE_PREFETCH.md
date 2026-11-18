# Hardware SM-active-aware adaptive prefetching 



```bash
python scripts/profile_sm_prefetch.py \
  --key '0 3 0 4096 4096 4 80 4' \
  --profile /tmp/warp_sm_profiles.txt \
  -- python your_single_shape_benchmark.py
```



3. Run inference:

```bash
export WARP_MARLIN_PREFETCH=4
export WARP_SM_PROFILE=/tmp/warp_sm_profiles.txt
python your_inference.py
```




