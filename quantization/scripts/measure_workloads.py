#!/usr/bin/env python3
"""Measure physically packed WARP prefill/decode separately, 50Hz NVML.
WARNING: re-load model for each bitmap because install_bitmap mutates modules.
Measurements use fixed-length greedy token-by-token forward, not model.generate.
"""
import argparse,csv,json,gc,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from warp_kernel.energy import NVMLPowerSampler,JetsonSysfsPowerSampler,integrate_energy
from warp_kernels.integration import install_bitmap,load_tuning

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True);ap.add_argument('--packed-dir',required=True)
    ap.add_argument('--bitmaps',required=True);ap.add_argument('--workloads',required=True)
    ap.add_argument('--out',default='energy_workloads.csv');ap.add_argument('--split',choices=['train','validation','test'],required=True)
    ap.add_argument('--warmup',type=int,default=1);ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--hz',type=float,default=50);ap.add_argument('--device-index',type=int,default=0)
    ap.add_argument('--tuning-json');ap.add_argument('--prompt-file');ap.add_argument('--seed',type=int,default=2027)
    ap.add_argument('--jetson-power-path');ap.add_argument('--jetson-scale',type=float,default=1e-3)
    ap.add_argument('--limit-configs',type=int,default=None)
    a=ap.parse_args();torch.manual_seed(a.seed)
    device=f'cuda:{a.device_index}';sync=lambda:torch.cuda.synchronize(a.device_index)
    workloads=[r for r in csv.DictReader(open(a.workloads)) if r['split']==a.split]
    bitmaps=np.load(a.bitmaps)
    if a.limit_configs is not None:bitmaps=bitmaps[:a.limit_configs]
    if len(workloads)==0:raise ValueError('no matching workloads')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    src=Path(a.prompt_file).read_text() if a.prompt_file else ('Calibration sentence for energy profiling. '*800)
    seed_ids=tokenizer(src,return_tensors='pt',add_special_tokens=True).input_ids[0]
    if len(seed_ids)==0:raise ValueError('empty prompt tokens')
    tuning=load_tuning(a.tuning_json)
    sampler=JetsonSysfsPowerSampler(a.jetson_power_path,a.jetson_scale) if a.jetson_power_path else NVMLPowerSampler(a.device_index)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    fields=['split','workload_id','config_id','bitmap','prompt_tokens','generation_tokens','run_id','prefill_energy_j','decode_energy_j','total_energy_j','prefill_latency_s','decode_latency_s','total_latency_s']
    finished=set()
    if path.exists():
        with path.open(newline='') as f:
            for r in csv.DictReader(f):finished.add((r['split'],r['workload_id'],int(r['config_id']),int(r['run_id'])))
    new=not path.exists() or path.stat().st_size==0
    with path.open('a',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        if new:wr.writeheader()
        for ci,bm in enumerate(bitmaps):
            if all((a.split,r['workload_id'],ci,ri) in finished for r in workloads for ri in range(a.runs)):continue
            # Fresh model per bitmap prevents patched fusion from breaking next bitmap.
            model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,low_cpu_mem_usage=True).to(device).eval()
            install_bitmap(model,a.packed_dir,bm.tolist(),device=device,dtype=torch.float16,tuning=tuning)
            for work in workloads:
                P=int(work['prompt_tokens']);G=int(work['generation_tokens'])
                if P+G>getattr(model.config,'max_position_embeddings',P+G):raise ValueError('context capacity exceeded')
                ids=seed_ids.repeat((P+len(seed_ids)-1)//len(seed_ids))[:P].unsqueeze(0).to(device)
                def forward_prefill():
                    with torch.inference_mode():
                        return model(input_ids=ids,use_cache=True,return_dict=True)
                def forward_decode(out):
                    cache=out.past_key_values
                    token=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                    with torch.inference_mode():
                        # Exactly G generated tokens: final token is predicted by prefill,
                        # so G-1 additional forward passes are needed.
                        for _ in range(G-1):
                            out=model(input_ids=token,past_key_values=cache,use_cache=True,return_dict=True)
                            cache=out.past_key_values
                            token=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                def measure_once():
                    holder={}
                    def do_pre():holder['out']=forward_prefill()
                    pre=integrate_energy(do_pre,sampler,hz=a.hz,synchronize=sync)
                    dec=integrate_energy(lambda:forward_decode(holder['out']),sampler,hz=a.hz,synchronize=sync) if G>1 else None
                    return pre,dec
                for _ in range(a.warmup):
                    pre,dec=measure_once();del pre,dec
                for ri in range(a.runs):
                    if (a.split,work['workload_id'],ci,ri) in finished:continue
                    pre,dec=measure_once()
                    ed=dec.joules if dec else 0.0;td=dec.seconds if dec else 0.0
                    row=dict(split=a.split,workload_id=work['workload_id'],config_id=ci,bitmap=json.dumps([int(x) for x in bm]),prompt_tokens=P,generation_tokens=G,run_id=ri,prefill_energy_j=pre.joules,decode_energy_j=ed,total_energy_j=pre.joules+ed,prefill_latency_s=pre.seconds,decode_latency_s=td,total_latency_s=pre.seconds+td)
                    wr.writerow(row);f.flush()
                    print(f"{a.split} {work['workload_id']} config={ci} run={ri}: {row['total_energy_j']:.2f} J",flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    print('Saved',path)
if __name__=='__main__':main()
