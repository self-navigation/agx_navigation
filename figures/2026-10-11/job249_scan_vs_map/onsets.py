"""Excursion onsets vs skid (2026-10-11): for every amcl excursion (|amcl - truth|
crosses 0.3 m after being < 0.15 m), the 3 s BEFORE the crossing vs calm 3 s windows,
over jobs 248 + 249: wheel-odometry distance minus true distance, body contact, and
skid = mean |wheel-odometry yaw rate - true yaw rate|. Body clearance from job 248's
renderer. Run from anywhere: python onsets.py > onsets.txt
"""
import os
RR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
os.chdir(RR)
import glob, sys, numpy as np
sys.argv=[sys.argv[0]]
__file__=os.path.abspath("figures/2026-10-11/job248_sweep/render.py"); exec(open(__file__).read().split("S = json.load")[0])
files=glob.glob("run_data/2026-10-11_job248_sweep-dev/*/s*/track_*.npz")+glob.glob("run_data/2026-10-11_job249_scan-rec/*/s*/track_*.npz")
rows=[]; base=[]
def win(z,a,b):
    T,O=z["track"],z["odom"]
    m=(T[:,0]>=a)&(T[:,0]<=b)
    if m.sum()<5: return None
    tt=T[m]; dtruth=np.sum(np.hypot(np.diff(tt[:,1]),np.diff(tt[:,2])))
    mo=(O[:,0]>=a)&(O[:,0]<=b); dwheel=np.trapezoid(np.abs(O[mo,4]),O[mo,0]) if mo.sum()>2 else np.nan
    clr=body_min(tt[:,1:4])
    tw=np.gradient(np.unwrap(T[:,3]),T[:,0])[m]; ww=np.interp(tt[:,0],O[:,0],O[:,5])
    skid=np.mean(np.abs(ww-tw))
    return dwheel-dtruth, clr, skid
for f in files:
    try: z=np.load(f)
    except Exception: continue
    if "amcl" not in z.files or len(z["amcl"])<10 or len(z["track"])<20: continue
    T,AM=z["track"],z["amcl"]; st=AM[:,7]
    e=np.hypot(AM[:,1]-np.interp(st,T[:,0],T[:,1]),AM[:,2]-np.interp(st,T[:,0],T[:,2]))
    armed=True
    for t,ee in zip(st,e):
        if ee<0.15: armed=True
        elif ee>0.3 and armed:
            armed=False; r=win(z,t-3,t)
            if r: rows.append(r)
    # baseline: windows where error stays <0.15 over next 3 s
    for t in np.arange(st[0]+3,st[-1]-3,3.0):
        if np.all(e[(st>=t-3)&(st<=t+3)]<0.15):
            r=win(z,t-3,t)
            if r: base.append(r)
for name,R in (("excursion onset (3 s before)",rows),("calm windows",base)):
    R=np.array(R); print(f"{name}: n {len(R)}; wheel over-report median {np.median(R[:,0]):.3f} m (p90 {np.percentile(R[:,0],90):.3f}); "
      f"contact in window {np.mean(R[:,1]<=0)*100:.0f}%; skid |w_wheel-w_true| median {np.median(R[:,2]):.2f} rad/s; over-report>0.2 m: {np.mean(R[:,0]>0.2)*100:.0f}%")
R=np.array(rows); print("onsets with over-report>0.2 m and contact:",np.mean((R[:,0]>0.2)&(R[:,1]<=0))*100, "% ; over-report>0.2 no contact:",np.mean((R[:,0]>0.2)&(R[:,1]>0))*100,"%")
