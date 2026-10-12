import glob,json,os,sys,numpy as np
sys.argv=["x","run_data/x","/tmp","tools/jobs/c1/steerable.txt"]
src=open("figures/2026-10-11/job251_rtab/render.py").read()
exec(src[src.index("def loc_error"):src.index("runs = {}")])
for job,cell in [("run_data/2026-10-12_job256_steerable-rtab","RQ"),("run_data/2026-10-12_job255_steerable","Q0")]:
    at,clean=[],[]
    for f in glob.glob(f"{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r=json.loads(l); tp=os.path.join(job,cell,f"s{r['seed']}",os.path.basename(r.get("track_path") or ""))
            if not os.path.isfile(tp): continue
            z=np.load(tp); e=loc_error(z)
            if e is None: continue
            if r.get("wall_overlap_s",0)>0 and r.get("wall_first_t") is not None:
                O=z["odom_filtered"]; st=O[:,6]; T=z["track"]
                st=st[(st>=T[0,0])&(st<=T[-1,0])][:len(e)]
                t=r["wall_first_t"]; 
                if not (st[0]<=t<=st[-1]): t=st[0]+t
                w=(st>=t-1.0)&(st<=t)
                if w.any(): at.append(e[w].mean())
            else: clean.append(np.median(e))
    print(cell, "contact runs: loc err in 1 s before first contact, median %.3f p90 %.3f (n %d) | clean runs median err %.3f"%(np.median(at),np.percentile(at,90),len(at),np.median(clean)))
