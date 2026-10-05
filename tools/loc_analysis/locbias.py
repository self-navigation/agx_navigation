import sys,glob,collections,numpy as np
sys.path.insert(0,'tools'); from departure_series import series
for cfg in sys.argv[1:]:
    cells=collections.defaultdict(list); lags=[]; errs=[]; bias=[]
    for p in sorted(glob.glob(f"{cfg}/s*/track_*ours*.npz")):
        try: s=series(p)
        except Exception as e: continue
        if s is None: continue
        b,T,t=s['believed'],s['truth'],s['t']
        m=t>0
        if m.sum()<30: continue
        dp=b[m,:2]-T[m,:2]; errs.append(np.median(np.hypot(*dp.T))); bias.append(dp.mean(0))
        for (x,y),e in zip(T[m,:2],dp): cells[(round(x*2),round(y*2))].append(e)
        # lag: shift truth by tau, find tau minimising believed-vs-truth error
        d=np.load(p,allow_pickle=True); TR=d['track']
        best=min(((np.median(np.hypot(b[m,0]-np.interp(t[m]+float(d['goal_t'])-tau,TR[:,0],TR[:,1]),b[m,1]-np.interp(t[m]+float(d['goal_t'])-tau,TR[:,0],TR[:,2]))),tau) for tau in np.arange(0,2.01,0.05)))
        lags.append((best[1],best[0],errs[-1]))
    L=np.array(lags); B=np.array(bias)
    # spatial consistency: in cells visited by >=3 runs' samples, |mean vector| / mean |e|
    cons=[np.linalg.norm(np.mean(v,0))/np.mean(np.linalg.norm(v,axis=1)) for v in cells.values() if len(v)>=20]
    print(cfg,f"runs={len(L)} med|loc_err|={np.median(L[:,2]):.3f} best_lag med={np.median(L[:,0]):.2f}s err_at_lag={np.median(L[:,1]):.3f} frac_lag>0.2={np.mean(L[:,0]>0.2):.2f} meanbias={B.mean(0).round(3)} spatial_coherence med={np.median(cons):.2f}")
