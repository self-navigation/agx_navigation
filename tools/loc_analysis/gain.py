import sys,glob,numpy as np
sys.path.insert(0,'tools'); from departure_series import series
for cfg in sys.argv[1:]:
    K=[];TAU=[];R2=[];Kturn=[];Kstr=[]
    for p in glob.glob(f"{cfg}/s*/track_*ours*.npz"):
        try: s=series(p)
        except Exception: continue
        if s is None: continue
        m=s['t']>0; eb=s['e_cross_believed'][m]; et=s['e_cross_true'][m]; t=s['t'][m]
        if len(t)<50 or np.ptp(et)<0.15: continue
        best=None
        for tau in np.arange(0,3.01,0.1):
            x=np.interp(t-tau,t,et); k=(x@eb)/(x@x); r=1-np.sum((eb-k*x)**2)/np.sum((eb-eb.mean())**2)
            if best is None or r>best[2]: best=(k,tau,r)
        K.append(best[0]);TAU.append(best[1]);R2.append(best[2])
        w=np.abs(np.gradient(np.unwrap(s['truth'][m,2]),t)); big=np.abs(et)>0.1
        for sel,L in ((big&(w>0.3),Kturn),(big&(w<0.1),Kstr)):
            if sel.sum()>10: L.append(np.median(eb[sel]/et[sel]))
    print(cfg.split('/')[-1],f"n={len(K)} gain k med={np.median(K):.2f} lag med={np.median(TAU):.1f}s (IQR {np.percentile(TAU,25):.1f}-{np.percentile(TAU,75):.1f}) R2 med={np.median(R2):.2f} | believed/true while turning={np.median(Kturn):.2f} straight={np.median(Kstr):.2f}")
