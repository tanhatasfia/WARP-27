
import json
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from warp_quant.surrogate import EnergySurrogate, one_hot_bitmaps

def basis(p,g,phase):
    p=np.asarray(p,dtype=float).reshape(-1)/1024.0
    g=np.asarray(g,dtype=float).reshape(-1)/1024.0
    if phase=='prefill':
        return np.column_stack([np.ones(len(p)),p,p*p,np.maximum(p-.25,0),np.maximum(p-.5,0),np.maximum(p-.75,0)])
    if phase=='decode':
       
        return np.column_stack([g,g*p,g*g,np.maximum(g-.25,0),np.maximum(g-.5,0),g*np.maximum(p-.5,0)])
    raise ValueError(phase)

def features(bitmaps,p,g,phase):
    bits=np.asarray(bitmaps,dtype=int);n,l=bits.shape
    z=one_hot_bitmaps(bits,l);phi=basis(p,g,phase)

    return np.column_stack([phi,np.einsum('ni,nj->nij',z,phi).reshape(n,-1)])

@dataclass
class WorkloadEnergyModel:
    n_blocks:int
    alpha:float
    coef_pre:np.ndarray
    coef_dec:np.ndarray
    delta:float=0.0

    def _phase(self,bitmaps,p,g,phase):
        c=self.coef_pre if phase=='prefill' else self.coef_dec
        return features(bitmaps,p,g,phase)@c
    def predict(self,bitmaps,p,g):
        return self._phase(bitmaps,p,g,'prefill')+self._phase(bitmaps,p,g,'decode')
    def at_workload(self,p,g):
   
        base=[2]*self.n_blocks
        bm=[base]
        for l in range(self.n_blocks):
            for b in (3,4):
                q=base.copy();q[l]=b;bm.append(q)
        n=len(bm)
        pred=self.predict(np.asarray(bm),np.full(n,p),np.full(n,g))
        base_e=float(pred[0]);costs={(l,2):0.0 for l in range(self.n_blocks)}
        idx=1
        for l in range(self.n_blocks):
            for b in (3,4):
                costs[(l,b)]=float(pred[idx]-base_e);idx+=1
        return EnergySurrogate(base_e,costs,self.delta,self.alpha,self.n_blocks)
    def save(self,path):
        with open(path,'w') as f:json.dump({'n_blocks':self.n_blocks,'alpha':self.alpha,'coef_pre':self.coef_pre.tolist(),'coef_dec':self.coef_dec.tolist(),'delta':self.delta},f,indent=2)
    @classmethod
    def load(cls,path):
        o=json.load(open(path));return cls(o['n_blocks'],o['alpha'],np.array(o['coef_pre']),np.array(o['coef_dec']),o['delta'])

def fit_model(bitmaps,p,g,e_pre,e_dec,alpha):
    bits=np.asarray(bitmaps,dtype=int)
    xpre=features(bits,p,g,'prefill');xdec=features(bits,p,g,'decode')
    
    a=Ridge(alpha=alpha,fit_intercept=False).fit(xpre,e_pre)
    b=Ridge(alpha=alpha,fit_intercept=False).fit(xdec,e_dec)
    return WorkloadEnergyModel(bits.shape[1],alpha,a.coef_,b.coef_)
