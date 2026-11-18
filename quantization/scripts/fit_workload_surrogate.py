#!/usr/bin/env python3
import argparse,csv,json
import numpy as np
from warp_kernel.workload_surrogate import fit_model, WorkloadEnergyModel

def load(path):
    rows=list(csv.DictReader(open(path)))
    if not rows:raise ValueError('empty energy CSV: '+path)
    grouped={}
    for r in rows:
        key=(r['workload_id'],r['config_id'])
        grouped.setdefault(key,[]).append(r)
    out=[]
    for rr in grouped.values():
        a=rr[0];x=dict(bitmap=json.loads(a['bitmap']),p=int(a['prompt_tokens']),g=int(a['generation_tokens']))
        for k in ('prefill_energy_j','decode_energy_j','total_energy_j'):
            x[k]=float(np.mean([float(r[k]) for r in rr]))
        out.append(x)
    bits=np.array([x['bitmap'] for x in out],dtype=int)
    p=np.array([x['p'] for x in out]);g=np.array([x['g'] for x in out])
    pre=np.array([x['prefill_energy_j'] for x in out]);dec=np.array([x['decode_energy_j'] for x in out])
    return bits,p,g,pre,dec

def metrics(y,pred):
    err=y-pred
    return {'n':len(y),'MAE_J':float(np.mean(abs(err))),'RMSE_J':float(np.sqrt(np.mean(err**2))),'P95_abs_J':float(np.quantile(abs(err),.95)),'MAPE_pct':float(np.mean(abs(err)/np.maximum(abs(y),1e-6))*100),'P95_underprediction_J':float(np.quantile(err,.95))}

def main():
    a=argparse.ArgumentParser()
    for x in ('train','validation'):a.add_argument('--'+x,required=True)
    a.add_argument('--test');a.add_argument('--out',default='workload_surrogate.json');a.add_argument('--report',default='workload_report.json')
    a.add_argument('--alphas',nargs='+',type=float,default=[0.001,.01,.1,1,10,100,1000])
    args=a.parse_args();tr=load(args.train);va=load(args.validation)
    if len(tr[0][0])!=len(va[0][0]):raise ValueError('mismatched n_blocks')
    candidates=[]
    for alpha in args.alphas:
        m=fit_model(*tr,alpha=alpha)
        yp=m.predict(*va[:3]);yt=va[3]+va[4]
        candidates.append((metrics(yt,yp)['MAE_J'],alpha,m))
    _,alpha,model=min(candidates,key=lambda x:x[0])
    yv=va[3]+va[4];pv=model.predict(*va[:3])
    
    model.delta=max(0.0,float(np.quantile(yv-pv,.95)))
    yt=tr[3]+tr[4];pt=model.predict(*tr[:3]);report={'alpha':alpha,'delta_J':model.delta,'train':metrics(yt,pt),'validation':metrics(yv,pv)}
    if args.test:
        te=load(args.test);report['test']=metrics(te[3]+te[4],model.predict(*te[:3]))
        report['test_budget_underprediction_rate_at_margin']=float(np.mean((te[3]+te[4])-model.predict(*te[:3])>model.delta))
    model.save(args.out)
    with open(args.report,'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
