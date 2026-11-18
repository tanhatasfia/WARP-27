#!/usr/bin/env python3

import argparse
import csv
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

METRIC='sm__cycles_active.avg.pct_of_peak_sustained_elapsed'

def parse_ncu_csv(output, kernel_filter):
    lines=output.splitlines()
    table=None
    for i,line in enumerate(lines):
        if line.lstrip().startswith('"ID","Process ID"') or (
                '"Kernel Name"' in line and '"Metric Name"' in line):
            table=lines[i:]
            break
    if table is None:
        raise RuntimeError('ncu did not produce a CSV metric table; check permissions and ncu version.\n'+output[-2500:])
    reader=csv.DictReader(io.StringIO('\n'.join(table)))
    values=[]
    for r in reader:
        name=next((str(v) for k,v in r.items() if k and 'Kernel Name' in k), '')
        metric=next((str(v) for k,v in r.items() if k and 'Metric Name' in k), '')
        if kernel_filter not in name or METRIC not in metric: continue
        value=next((str(v) for k,v in r.items() if k and 'Metric Value' in k),'')
        try: values.append(float(value.replace(',','').replace('%','')))
        except ValueError: continue
    if not values: raise RuntimeError('No SM-active counter samples for matching kernel.\n'+output[-2500:])
    return sum(values)/len(values),len(values)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key',required=True,help='8 integers: CUDA_DEVICE BITS MODE K V SPLITK CTAS SLICE_TILES')
    ap.add_argument('--profile',required=True,help='Selection profile path for C++ runtime')
    ap.add_argument('--kernel-filter',default='cb_mma_gemv_kernel')
    ap.add_argument('--ncu',default='ncu')
    ap.add_argument('--force',action='store_true',help='Replace existing entry for shape')
    ap.add_argument('benchmark',nargs=argparse.REMAINDER,help='-- executable and its arguments')
    args=ap.parse_args()
    key=args.key.split()
    if len(key)!=8 or any(not x.isdigit() for x in key): ap.error('--key requires 8 nonnegative integers')
    k=[int(x) for x in key]; max_d=min(3,max(0,k[7]-1))
    if max_d<1:ap.error('This K slice has no prefetch lookahead to optimize.')
    bench=args.benchmark
    if bench and bench[0]=='--':bench=bench[1:]
    if not bench:ap.error('Supply a benchmark command after --')
    if not shutil.which(args.ncu):ap.error('NVIDIA Nsight Compute (ncu) is required on the GPU host')
    results=[]
    for d in range(1,max_d+1):
        env=dict(os.environ, WARP_MARLIN_PREFETCH=str(d))
        cmd=[args.ncu,'--csv','--metrics',METRIC,'--kernel-name','regex:'+args.kernel_filter,'--launch-count','20','--target-processes','all']+bench
        proc=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        output=proc.stdout+'\n'+proc.stderr
        if proc.returncode:raise RuntimeError(f'ncu failed for prefetch={d}:\n{output[-3500:]}')
        u,n=parse_ncu_csv(output,args.kernel_filter)
        print(f'd={d}: SM-active={u:.3f}% across {n} kernel executions',flush=True)
        results.append((u,d))
    best=max(results,key=lambda v:(v[0],-v[1]))[1]
    path=Path(args.profile)
    rows=[]
    if path.exists():rows=[x for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
    for line in rows:
        if line.split()[:8]==key and not args.force:ap.error('Profile already contains this shape; use --force')
    rows=[line for line in rows if line.split()[:8]!=key]
    rows.append(' '.join(key+[str(best)]))
    path.write_text('# CUDA_DEVICE BITS MODE K V SPLITK CTAS SLICE_TILES DISTANCE\n'+'\n'.join(rows)+'\n')
    print(f'Selected d={best} using SM-active hardware counter; wrote {path}')
if __name__=='__main__':main()
