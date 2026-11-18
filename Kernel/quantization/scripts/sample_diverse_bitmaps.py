#!/usr/bin/env python3
import argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--blocks',type=int,default=32);p.add_argument('--count',type=int,default=38)
p.add_argument('--seed',type=int,default=2027);p.add_argument('--out',default='bitmaps_workload.npy');a=p.parse_args()
if a.count<3:raise ValueError('count must be >=3 (3 uniform plus mixed configurations)')
r=np.random.default_rng(a.seed);rows=[np.full(a.blocks,b,dtype=np.int8) for b in (2,3,4)]
seen={tuple(x) for x in rows}
while len(rows)<a.count:
    probs=r.dirichlet([.6,.6,.6])
    x=r.choice([2,3,4],size=a.blocks,p=probs).astype(np.int8)
    if tuple(x) not in seen:rows.append(x);seen.add(tuple(x))
np.save(a.out,np.stack(rows));print(a.out,len(rows),'unique bitmaps (3 uniform +',len(rows)-3,'mixed)')
