import numpy as np, glob, os, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

import os as _os
CACHE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'data', 'session_cache')
LIM,NG,DF,SPL_C=2.0,60,4,1.0
SUPP_R,SUPP_N,MIN_SESS=0.35,5,9
grid=np.linspace(-LIM,LIM,NG); GD,GG=np.meshgrid(grid,grid)
flat=np.column_stack([GD.ravel(),GG.ravel()])
spl=SplineTransformer(n_knots=DF,degree=3,include_bias=False)

def tens(B): 
    h=B.shape[1]//2
    return np.einsum('ni,nj->nij',B[:,:h],B[:,h:]).reshape(len(B),-1)

def surface(X,y):
    m=LogisticRegression(C=SPL_C,max_iter=4000).fit(tens(spl.fit_transform(X)),y)
    return m.predict_proba(tens(spl.transform(flat)))[:,1].reshape(NG,NG)

def axes_for(z, mode):
    C=z['counts'].astype(float); ch=(z['choice']==1).astype(int)
    mu,sd=C.mean(0),C.std(0); keep=sd>0
    Z=(C[:,keep]-mu[keep])/sd[keep]; rp,lp=z['right_pref'][keep],z['left_pref'][keep]
    if mode.startswith('simple'):
        d=Z[:,rp].mean(1)-Z[:,lp].mean(1); g=Z[:,rp].mean(1)+Z[:,lp].mean(1)
    else:
        d=np.zeros(len(ch)); g=np.zeros(len(ch))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Z,ch):
            w=LogisticRegression(C=0.05,max_iter=2000).fit(Z[tr],ch[tr]).coef_.ravel()
            d[te]=Z[te]@w; g[te]=Z[te]@np.abs(w)
    if mode.endswith('orth'):
        g=g-np.polyfit(d,g,1)[0]*d
    return (d-d.mean())/d.std(), (g-g.mean())/g.std()

def run(mode):
    S={k:[] for k in ('choice','conf','acc')}; sup=np.zeros((NG,NG)); n=0
    for f in sorted(glob.glob(os.path.join(CACHE,'session_*.npz'))):
        z=np.load(f)
        if z['right_pref'].sum()<3 or z['left_pref'].sum()<3: continue
        d,g=axes_for(z,mode); X=np.column_stack([d,g])
        cov=(((np.abs(d[None,:]-GD.ravel()[:,None])<SUPP_R)&
              (np.abs(g[None,:]-GG.ravel()[:,None])<SUPP_R)).sum(1)>=SUPP_N).reshape(NG,NG)
        sup+=cov
        for k,y in (('choice',(z['choice']==1).astype(int)),('conf',z['confidence'].astype(int)),
                    ('acc',z['correct'].astype(int))):
            s=surface(X,y); s[~cov]=np.nan; S[k].append(s)
        n+=1
    mask=sup>=MIN_SESS
    return {k:np.ma.masked_array(np.nanmean(v,0),~mask) for k,v in S.items()}, mask, n
