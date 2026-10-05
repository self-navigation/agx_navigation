import sys,glob,json,os,numpy as np
sys.path.insert(0,'tools'); from departure_series import series
for cfg in sys.argv[1:]:
    rows={os.path.basename(r['track_path']):r for r in map(json.loads,open(os.path.dirname(cfg)+'/all_rows.jsonl')) if r.get('track_path') and r['track_path'].split('/')[-3]==os.path.basename(cfg)}
    on=[];off=[];growon=[];growoff=[]
    for p in glob.glob(f"{cfg}/s*/track_*ours*.npz"):
        r=rows.get(os.path.basename(p)); s=None
        try: s=series(p)
        except Exception: pass
        if s is None or r is None: continue
        m=s['t']>0; T=s['truth'][m]; dp=s['believed'][m,:2]-T[:,:2]
        th=T[:,2]; cr=np.abs(-np.sin(th)*dp[:,0]+np.cos(th)*dp[:,1])
        P=np.array([[q['x'],q['y']] for q in r['patches'] if q.get('ok')]) if r.get('patches') else np.zeros((0,2))
        near=(np.min(np.hypot(T[:,0,None]-P[None,:,0],T[:,1,None]-P[None,:,1]),1)<1.0) if len(P) else np.zeros(len(T),bool)
        on+=list(cr[near]); off+=list(cr[~near])
    on,off=np.array(on),np.array(off)
    print(os.path.basename(cfg),f"|cross loc err| near patch: med {np.median(on):.3f} p90 {np.percentile(on,90):.3f} (n={len(on)})  away: med {np.median(off):.3f} p90 {np.percentile(off,90):.3f} (n={len(off)})")
