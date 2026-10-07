import json,math
from scipy.stats import binomtest
def load(c):
    return {(r:=json.loads(l))['plan']:r for l in open(f'{c}/rows.s0.jsonl')}
arms=dict(L='ours-lib',M='mppi',R='rpp',V='vpp',G='graceful',K='gmpc')
print('cfg   | arrN arrS | cleanN cleanS | medErrN medErrS | S better/worse (final_err) p | lost by S | gained by S')
for loc in 'TA':
  for a,n in arms.items():
    N=load(a+loc+'N');S=load(a+loc+'S');P=sorted(set(N)&set(S))
    arr=lambda r:r['outcome']=='arrived'
    cl=lambda r:arr(r) and (r.get('wall_overlap_s') or 0)==0
    e=lambda r:r['final_err'] if r.get('final_err') is not None and math.isfinite(r['final_err']) else 99
    import statistics as st
    b=sum(e(S[p])<e(N[p])-0.02 for p in P);w=sum(e(S[p])>e(N[p])+0.02 for p in P)
    pv=binomtest(b,b+w).pvalue if b+w else 1
    lost=sum(arr(N[p]) and not arr(S[p]) for p in P);gain=sum(arr(S[p]) and not arr(N[p]) for p in P)
    print(f'{a}{loc} n={len(P)} | {sum(map(arr,(N[p] for p in P)))} {sum(map(arr,(S[p] for p in P)))} | {sum(cl(N[p]) for p in P)} {sum(cl(S[p]) for p in P)} | {st.median(e(N[p]) for p in P):.3f} {st.median(e(S[p]) for p in P):.3f} | {b}/{w} p={pv:.3f} | {lost} | {gain}')
