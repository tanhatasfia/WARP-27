#!/usr/bin/env python3
import argparse,csv,json
from decimal import Decimal
from warp_quant.workload_surrogate import WorkloadEnergyModel
from warp_quant.pulse import solve_pulse
p=argparse.ArgumentParser();p.add_argument('--surrogate',required=True);p.add_argument('--trps-csv',required=True)
p.add_argument('--prompt-len',type=int,required=True);p.add_argument('--new-tokens',type=int,required=True)
p.add_argument('--avg-bit-budget',type=Decimal,default=Decimal('3.1'));p.add_argument('--trps-tolerance',type=Decimal,required=True);p.add_argument('--out',default='allocation_workload.json')
a=p.parse_args()
if not 64<=a.prompt_len<=1024 or not 128<=a.new_tokens<=1024:raise ValueError('outside validated length domain')
model=WorkloadEnergyModel.load(a.surrogate)
scores={(int(x['block']),int(x['bits'])):float(x['trps']) for x in csv.DictReader(open(a.trps_csv))}
sur=model.at_workload(a.prompt_len,a.new_tokens)
result=solve_pulse(scores,sur,a.trps_tolerance,a.avg_bit_budget).__dict__
result.update(prompt_tokens=a.prompt_len,generation_tokens=a.new_tokens,uncertainty_margin_j=model.delta)
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
