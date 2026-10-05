import sys,glob,numpy as np
sys.path.insert(0,'tools'); from departure_series import series
for cfg in sys.argv[1:]:
    A=[];C=[];Y=[];ratio=[]
    for p in sorted(glob.glob(f"{cfg}/s*/track_*ours*.npz")):
        try: s=series(p)
        except Exception: continue
        if s is None: continue
        m=s['t']>0
        if m.sum()<30: continue
        dp=s['believed'][m,:2]-s['truth'][m,:2]; th=s['truth'][m,2]
        al=np.cos(th)*dp[:,0]+np.sin(th)*dp[:,1]; cr=-np.sin(th)*dp[:,0]+np.cos(th)*dp[:,1]
        A.append(np.median(al)); C.append(np.median(np.abs(cr))); Y.append(np.degrees(np.median(np.abs(s['believed'][m,2]-s['truth'][m,2]))))
        ratio.append(np.max(np.abs(s['e_cross_believed'][m]))/max(1e-3,np.max(np.abs(s['e_cross_true'][m]))))
    A=np.array(A)
    print(cfg.split('/')[-1],f"n={len(A)} along(believed ahead +) median={np.median(A):+.3f} frac>0={np.mean(A>0):.2f} | |cross| med={np.median(C):.3f} | |yaw| med={np.median(Y):.2f}deg | believed/true peak e_cross med={np.median(ratio):.2f}")
