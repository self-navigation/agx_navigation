"""Train the re-join actor: TD3, eps-TD3+BC, TD3+BC or BC.

    python3 -m agx_planning.rl_corrector.train_rejoin --kind eps_td3bc \\
        --labels <labels dir> --plans <plan library> \\
        --map src/rudn-ordjo-building/maps/floor_6.yaml -o <run dir>

Labels are split (by --split-seed) into a test set of whole held-out plans, a
validation set used to pick the best checkpoint, and training. Each step rolls
out student and teacher at a sampled plant chi. Writes args.json, curve.jsonl,
policy_best.npz / policy_final.npz, beta_map.npz and eval_final.npz to -o.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from ..supervisor.walls import SignedWallDistance
from . import rejoin_env as renv

# residual scale per knot: v [m/s], omega [rad/s]; the teacher's residual is
# within (0.38, 0.89) at p99 on the labels
SCALE_V, SCALE_OM = 0.6, 1.5


def _chunks(n, size):
    for a in range(0, n, size):
        yield np.arange(a, min(a + size, n))


def costs(env, ii, actions, chi, chunk=4096):
    out = {k: np.empty(len(ii)) for k in ("cost", "j_track", "j_term", "j_contact", "pin")}
    for c in _chunks(len(ii), chunk):
        r = env.rollout(ii[c], actions[c], chi[c])
        for k in out:
            out[k][c] = r[k]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="eps_td3bc", choices=("td3", "td3bc", "eps_td3bc", "bc"))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--n-test-plans", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--chi", type=float, nargs=2, default=(1.3, 2.0))
    ap.add_argument("--chi-hat-noise", type=float, default=0.0)
    ap.add_argument("--env-samples", type=int, default=500_000)
    ap.add_argument("--env-batch", type=int, default=1024)
    ap.add_argument("--utd", type=float, default=0.25, help="gradient updates per env sample")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--expl", type=float, default=0.1, help="exploration noise sd, normalised action units")
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--eps0", type=float, default=float("nan"),
                    help="default: 0.1 x the teacher's median nominal-chi cost on the training pool")
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-labels", type=int, default=0, help="smoke tests: use only the first N labels")
    a = ap.parse_args(argv)

    import torch
    from .td3_bc import Learner, Variant
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"

    labels = renv.load_labels(a.labels)
    if a.max_labels:
        labels = {k: v[:a.max_labels] for k, v in labels.items()}
    plans = renv.load_plans(a.plans)
    walls = SignedWallDistance.from_map_yaml(a.map)
    env = renv.RejoinEnv(plans, labels, walls, renv.RejoinEnvConfig(chi_range=tuple(a.chi),
                                                                    chi_hat_noise=a.chi_hat_noise))
    n_k = env.n_knots
    scale = np.tile([SCALE_V, SCALE_OM], n_k)

    # --- splits ---------------------------------------------------------------
    srng = np.random.default_rng(a.split_seed)
    used = sorted(set(env.pid.tolist()))
    test_pids = set(srng.choice(used, a.n_test_plans, replace=False).tolist())
    is_test = np.isin(env.pid, list(test_pids))
    solved = env.solved & np.all(np.isfinite(env.a_teacher), axis=1)
    test_idx = srng.permutation(np.where(is_test & solved)[0])[:a.n_eval]
    pool = np.where(~is_test)[0]
    val_idx = srng.permutation(pool[solved[pool]])[:a.n_eval]
    train_idx = np.setdiff1d(pool, val_idx)
    chi_test = srng.uniform(*a.chi, len(test_idx))
    chi_val = srng.uniform(*a.chi, len(val_idx))
    nom = lambda n: np.full(n, renv.CHI_NOMINAL)  # noqa: E731
    evalsets = {"test": (test_idx, chi_test), "test_nominal": (test_idx, nom(len(test_idx))),
                "val": (val_idx, chi_val), "val_nominal": (val_idx, nom(len(val_idx)))}

    # --- teacher baselines (shared by all variants) ------------------------------
    teach = {k: costs(env, ii, env.a_teacher[ii], c) for k, (ii, c) in evalsets.items()}
    j0 = np.full(env.n, np.nan)  # J_chi0[a_PMP] per problem
    sol_idx = np.where(solved)[0]
    j0[sol_idx] = costs(env, sol_idx, env.a_teacher[sol_idx], nom(len(sol_idx)))["cost"]
    grng = np.random.default_rng(a.split_seed + 1)
    gi = grng.choice(train_idx[solved[train_idx]], 8192)
    gchi = grng.uniform(*a.chi, len(gi))
    gap = costs(env, gi, env.a_teacher[gi], gchi)["cost"] - j0[gi]
    # eps0: the gap at which beta has fallen to 1/e. NOT the median gap: most
    # re-joins are straight, chi barely matters there, and the median gap is
    # ~0.004 (2026-10-08), which would switch the teacher off almost everywhere
    # it is not exactly optimal. 10% of a typical re-join's cost is G3's own
    # tolerance: past it, the teacher's model error is worth correcting.
    tr_sol = train_idx[solved[train_idx]]
    eps0 = a.eps0 if np.isfinite(a.eps0) else 0.1 * float(np.median(j0[tr_sol]))
    print(f"[setup {time.time() - t_start:.0f}s] train {len(train_idx)} val {len(val_idx)} "
          f"test {len(test_idx)} (plans {sorted(env.plan_names[p] for p in test_pids)}); "
          f"eps_hat median {np.median(gap):.3f} p75 {np.percentile(gap, 75):.3f} p90 {np.percentile(gap, 90):.3f} "
          f"(frac > eps0: {np.mean(gap > eps0):.2f}); J_chi0 median {np.median(j0[tr_sol]):.3f}; eps0 {eps0:.3f}",
          flush=True)

    # state normalisation from the training pool
    s_tr = env.state(train_idx[:20000], env.sample_chi(min(20000, len(train_idx)), grng))
    s_mean, s_std = s_tr.mean(0), s_tr.std(0) + 1e-6
    variant = Variant(kind=a.kind, alpha=a.alpha, beta0=a.beta0, eps0=eps0)
    learner = Learner(env.state_dim, env.action_dim, variant, s_mean, s_std, device=dev)

    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(a), commit=commit, eps0=eps0, device=dev, n_knots=n_k,
                   test_plans=sorted(env.plan_names[p] for p in test_pids),
                   teacher={k: dict(mean=float(v["cost"].mean()), median=float(np.median(v["cost"])))
                            for k, v in teach.items()}),
              open(os.path.join(a.out, "args.json"), "w"), indent=1)

    def export(name):
        w = learner.export()
        np.savez(os.path.join(a.out, name), scale=scale, n_knots=np.array(n_k), **w)

    def evaluate(n_seen, extra):
        row = dict(env_samples=n_seen, wall_s=round(time.time() - t_start, 1), **extra)
        per = {}
        for k, (ii, c) in evalsets.items():
            s = env.state(ii, c)
            act = env.plan_knots[ii] + scale * learner.act(s)
            r = costs(env, ii, act, c)
            per[k] = r
            tc = teach[k]["cost"]
            row[k] = dict(mean=float(r["cost"].mean()), teacher_mean=float(tc.mean()),
                          ratio=float(r["cost"].mean() / tc.mean()),
                          median_ratio=float(np.median(r["cost"] / np.maximum(tc, 1e-9))),
                          win=float(np.mean(r["cost"] < tc)), j_track=float(r["j_track"].mean()),
                          contact=float(np.mean(r["j_contact"] > 0)), pin_med=float(np.median(r["pin"])))
        with open(os.path.join(a.out, "curve.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{n_seen:>7d} {row['wall_s']:6.0f}s] val ratio {row['val']['ratio']:.3f} "
              f"(nom {row['val_nominal']['ratio']:.3f})  test ratio {row['test']['ratio']:.3f} "
              f"(nom {row['test_nominal']['ratio']:.3f})  win {row['test']['win']:.2f}", flush=True)
        return row, per

    # --- the loop -----------------------------------------------------------------
    N = a.env_samples
    sd, ad = env.state_dim, env.action_dim
    buf = dict(s=np.empty((N, sd), np.float32), a=np.empty((N, ad), np.float32),
               r=np.empty(N, np.float32), a_bc=np.zeros((N, ad), np.float32),
               m=np.zeros(N, np.float32), eps_hat=np.zeros(N, np.float32))
    chi_buf = np.empty(N, np.float32)
    rng = np.random.default_rng(a.seed)
    n, best, info = 0, np.inf, {}
    row, _ = evaluate(0, {})
    next_eval = a.eval_every
    while n < N:
        B = min(a.env_batch, N - n)
        ii = rng.choice(train_idx, B)
        chi = env.sample_chi(B, rng)
        s = env.state(ii, chi, rng)
        an = np.clip(learner.act(s) + rng.normal(0, a.expl, (B, ad)), -1, 1)
        r = env.rollout(ii, env.plan_knots[ii] + scale * an, chi)
        ok = solved[ii]
        eps_hat = np.zeros(B)
        if ok.any():
            rt = env.rollout(ii[ok], env.a_teacher[ii[ok]], chi[ok])
            eps_hat[ok] = rt["cost"] - j0[ii[ok]]
        sl = slice(n, n + B)
        buf["s"][sl], buf["a"][sl], buf["r"][sl] = s, an, -r["cost"]
        buf["a_bc"][sl] = np.where(ok[:, None], np.clip(np.nan_to_num(env.teacher_residual(ii)) / scale, -1, 1), 0)
        buf["m"][sl], buf["eps_hat"][sl] = ok, eps_hat
        chi_buf[sl] = chi
        n += B
        for _ in range(int(round(B * a.utd))):
            j = rng.integers(0, n, a.batch)
            info = learner.update({k: v[j] for k, v in buf.items()})
        if n >= next_eval or n >= N:
            row, _ = evaluate(n, dict(train_cost=float(r["cost"].mean()), **info))
            if row["val"]["mean"] < best:
                best = row["val"]["mean"]
                export("policy_best.npz")
            next_eval += a.eval_every
    export("policy_final.npz")
    _, per = evaluate(n, dict(final=True))
    np.savez_compressed(os.path.join(a.out, "eval_final.npz"),
                        **{f"{k}_{q}": v[q] for k, v in per.items() for q in ("cost", "j_track", "pin")},
                        **{f"teacher_{k}_{q}": v[q] for k, v in teach.items() for q in ("cost", "j_track", "pin")},
                        test_idx=test_idx, val_idx=val_idx, chi_test=chi_test, chi_val=chi_val)
    last = slice(max(0, n - 50_000), n)
    from .td3_bc import beta_of
    import torch as _t
    beta = beta_of(variant, _t.as_tensor(buf["eps_hat"][last])).numpy() * buf["m"][last]
    np.savez_compressed(os.path.join(a.out, "beta_map.npz"), s=buf["s"][last], chi=chi_buf[last],
                        eps_hat=buf["eps_hat"][last], m=buf["m"][last], beta=beta)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
