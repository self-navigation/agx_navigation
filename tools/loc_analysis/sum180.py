import json,collections,math,sys
from scipy.stats import binomtest
f,base=sys.argv[1],sys.argv[2]
rows=[json.loads(l) for l in open(f)]
by=collections.defaultdict(dict)
for r in rows:
    c=r['track_path'].split('/')[-3] if r.get('track_path') else 'NONE'
    by[c][(r['plan'],r['seed'])]=r
def fe(r):
    v=r.get('final_err')
    return v if (v is not None and r.get('outcome') in('arrived','failed') and math.isfinite(v)) else None
out=[]
for c,d in by.items():
    v=[fe(r) for r in d.values() if fe(r) is not None]
    oc=collections.Counter(r.get('outcome') for r in d.values())
    if not v: print(c,'NO VALID',oc); continue
    w=l=0
    for k,r in d.items():
        b=by[base].get(k)
        if b and fe(r) is not None and fe(b) is not None and fe(r)!=fe(b): w+=fe(r)<fe(b); l+=fe(r)>fe(b)
    out.append((sum(x>0.5 for x in v)/len(v),c,len(v),sorted(v)[len(v)//2],w,l,binomtest(w,w+l).pvalue if w+l else 1,dict(oc)))
for s in sorted(out): print(f"{s[1]:8s} n={s[2]:3d} miss={s[0]:.0%} med_fe={s[3]:.3f} better {s[4]}/{s[4]+s[5]} p={s[6]:.2g} {s[7]}")
