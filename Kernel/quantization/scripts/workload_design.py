#!/usr/bin/env python3
"""Generate reproducible 20/9/3 workload splits with an untouched test set."""
import argparse, csv
import numpy as np
from scipy.stats import qmc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='workloads.csv');ap.add_argument('--seed',type=int,default=2027)
    a=ap.parse_args()
   
    tests=[(200,300),(421,637),(1024,1024)]
    excluded=set(tests)

    anchors=[(64,128),(64,1024),(1024,128),(128,256),(256,128),(256,512),(512,256),(512,512),(768,768),(1024,512)]
    assert len(anchors)==10
    def lhs(n,seed,exclude):
        r=qmc.LatinHypercube(d=2,seed=seed); chosen=[];seen=set(exclude)
        while len(chosen)<n:
            for u,v in r.random(n*4):
                pair=(int(round(64+u*960)),int(round(128+v*896)))
                if pair not in seen and pair[0] != 100 and pair[1] != 384:
                    chosen.append(pair);seen.add(pair)
                if len(chosen)==n:break
        return chosen
    train=anchors+lhs(10,a.seed,excluded|set(anchors))
    valid=lhs(9,a.seed+1,excluded|set(train))
    rows=[{'workload_id':f'{split[0].upper()}{i:02}', 'split':split,'prompt_tokens':pr,'generation_tokens':ge} for split,pairs in [('train',train),('validation',valid),('test',tests)] for i,(pr,ge) in enumerate(pairs,1)]
    with open(a.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Saved {a.out}: train={len(train)} validation={len(valid)} test={len(tests)}')
    for row in rows:print(row)
if __name__=='__main__':main()
