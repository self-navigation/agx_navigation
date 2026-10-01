#!/usr/bin/env python3
"""Vitals timeline + freeze forensics for a comparison campaign (#32).

WHY. In v3/v4 a cell's whole stack falls silent mid-drive, ~1 cell in 5, and
sits until the 900 s backstop. The logs cannot say whether gz-sim hung (0%
CPU, blocked) or spun, or whether only the clock bridge died. This watcher
runs BESIDE a campaign (it never touches the stacks) and records:

  vitals.csv   every --period s: host CPU/RAM/load/PSI/GPU, plus per worker
               scope (agx-wN, #28) CPU cores, memory and pid count, and the
               age of that worker's newest stack log.
  freeze/<cell>/   once per cell whose stack log has been silent for
               --stale s while its scope is alive: gdb backtraces of every
               gz process in the scope, per-thread state/wchan/CPU, and a
               gz-side vs ROS-side /clock probe, and py-spy dumps of every
               Python process (needs ~/.pyspy).

Run on the VM:  python3 tools/freeze_watch.py --out-dir ~/cmp_v5/seed0 --workers "1 2 3"
Plot locally:   tools/plot_vitals.py <vitals.csv>
"""
import argparse
import csv
import glob
import os
import subprocess
import threading
import time

PYSPY = os.path.expanduser("~/.pyspy/bin/py-spy")  # venv; absent -> skipped
CG_APP = "/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice"


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def host_cpu():
    f = read("/proc/stat").splitlines()[0].split()[1:]
    v = list(map(int, f))
    idle = v[3] + v[4]
    return sum(v), idle


def psi(kind):
    # "some avg10=..." -> some avg10, full avg10
    out = {}
    for line in read(f"/proc/pressure/{kind}").splitlines():
        parts = line.split()
        out[parts[0]] = float(parts[1].split("=")[1])
    return out.get("some", float("nan")), out.get("full", float("nan"))


def meminfo():
    m = {}
    for line in read("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return m["MemTotal"] - m["MemAvailable"], m.get("SwapTotal", 0) - m.get("SwapFree", 0)


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        u, m = out.strip().split(",")
        return float(u), float(m)
    except Exception:
        return float("nan"), float("nan")


def scope_dir(w):
    return os.path.join(CG_APP.format(uid=os.getuid()), f"agx-w{w}.scope")


def scope_stats(w):
    d = scope_dir(w)
    if not os.path.isdir(d):
        return None
    usage = 0
    for line in read(os.path.join(d, "cpu.stat")).splitlines():
        if line.startswith("usage_usec"):
            usage = int(line.split()[1])
    mem = int(read(os.path.join(d, "memory.current"), "0") or 0)
    pids = int(read(os.path.join(d, "pids.current"), "0") or 0)
    return usage, mem, pids


def newest_log(out_dir, w):
    logs = glob.glob(os.path.join(out_dir, f"stack_w{w}_*.log"))
    if not logs:
        return None, None
    p = max(logs, key=os.path.getmtime)
    return p, time.time() - os.path.getmtime(p)


def scope_procs(w):
    pids = []
    for line in read(os.path.join(scope_dir(w), "cgroup.procs")).split():
        pids.append(int(line))
    return pids


def cmdline(pid):
    return read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()


def sh(cmd, env=None, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}"


def forensics(out_dir, w, log, age):
    cell = os.path.basename(log)[len("stack_"):-len(".log")]
    dest = os.path.join(out_dir, "freeze", cell)
    os.makedirs(dest, exist_ok=True)
    note = [f"captured {time.strftime('%F %T')} worker={w} log_silent={age:.0f}s"]
    env = dict(os.environ, GZ_PARTITION=f"agx{w}", ROS_DOMAIN_ID=str(40 + w))
    procs = scope_procs(w)

    # Per-process / per-thread state for the whole scope: R vs S/D tells
    # spinning from blocked, wchan says on what.
    lines = []
    for pid in procs:
        lines.append(f"== {pid} {cmdline(pid)[:200]}")
        lines.append(sh(f"ps -L -o tid,stat,pcpu,wchan:32,comm -p {pid}", timeout=10))
    with open(os.path.join(dest, "threads.txt"), "w") as f:
        f.write("\n".join(lines))

    # Two samples of CPU 10 s apart: hung (0) vs spinning.
    a = scope_stats(w)
    time.sleep(10)
    b = scope_stats(w)
    if a and b:
        note.append(f"scope cpu over 10 s: {(b[0] - a[0]) / 1e7:.3f} cores")

    for pid in procs:
        c = cmdline(pid)
        if "gz sim" in c or "gz-sim" in c or "gz_sim" in c:
            bt = sh(f"sudo -n gdb -p {pid} -batch -ex 'set pagination off' "
                    f"-ex 'thread apply all bt'", timeout=120)
            with open(os.path.join(dest, f"gdb_{pid}.txt"), "w") as f:
                f.write(c + "\n\n" + bt)
            note.append(f"gdb dumped pid {pid}: {c[:120]}")
        elif "python" in c and os.path.exists(PYSPY):
            out = sh(f"sudo -n {PYSPY} dump --pid {pid}", timeout=60)
            with open(os.path.join(dest, f"pyspy_{pid}.txt"), "w") as f:
                f.write(c + "\n\n" + out)

    # Does gz itself still tick, and does ROS see it?
    clock = "== gz /clock (2 samples)\n"
    clock += sh("gz topic -e -t /clock -n 2", env=env, timeout=15)
    clock += "\n== gz /stats\n" + sh("gz topic -e -t /stats -n 1", env=env, timeout=15)
    clock += "\n== gz topic -l\n" + sh("gz topic -l", env=env, timeout=15)
    clock += "\n== ros /clock\n" + sh(
        "bash -c 'set +u; source /opt/ros/jazzy/setup.bash; "
        "ros2 topic echo --once /clock'", env=env, timeout=20)
    with open(os.path.join(dest, "clock.txt"), "w") as f:
        f.write(clock)
    with open(os.path.join(dest, "README.txt"), "w") as f:
        f.write("\n".join(note) + "\n")
    print("[freeze]", " | ".join(note), flush=True)


def safe_forensics(*args):
    try:
        forensics(*args)
    except Exception as e:  # never let forensics kill the timeline
        print(f"[freeze] w{args[1]} forensics failed: {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", action="append", required=True,
                    help="campaign dir(s) holding stack_w*.log; repeatable")
    ap.add_argument("--workers", required=True, help='e.g. "1 2 3 4 5 6"')
    ap.add_argument("--period", type=float, default=5.0)
    ap.add_argument("--stale", type=float, default=150.0,
                    help="log silence (s) that counts as a freeze; healthy cells "
                         "stay under ~80 s")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    workers = [int(x) for x in a.workers.split()]
    csv_path = a.csv or os.path.join(a.out_dir[0], "vitals.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    cols = ["t", "host_cpu_cores", "load1", "mem_used", "swap_used",
            "psi_cpu_some", "psi_cpu_full", "psi_mem_some", "psi_mem_full",
            "psi_io_some", "psi_io_full", "gpu_util", "gpu_mem"]
    for w in workers:
        cols += [f"w{w}_cpu", f"w{w}_mem", f"w{w}_pids", f"w{w}_log_age"]
    new = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(cols)

    ncpu = os.cpu_count()
    prev_host = host_cpu()
    prev_scope = {w: scope_stats(w) for w in workers}
    prev_t = time.time()
    captured = set()
    while True:
        time.sleep(a.period)
        now = time.time()
        dt = now - prev_t
        tot, idle = host_cpu()
        busy = 1 - (idle - prev_host[1]) / max(1, tot - prev_host[0])
        prev_host = (tot, idle)
        mem, swap = meminfo()
        cs, cf = psi("cpu")
        ms, mf = psi("memory")
        ios, iof = psi("io")
        gu, gm = gpu()
        row = [round(now, 1), round(busy * ncpu, 2),
               float(read("/proc/loadavg").split()[0]), mem, swap,
               cs, cf, ms, mf, ios, iof, gu, gm]
        for w in workers:
            st = scope_stats(w)
            p = prev_scope.get(w)
            cores = ((st[0] - p[0]) / 1e6 / dt) if (st and p and st[0] >= p[0]) else ""
            prev_scope[w] = st
            log, age = None, None
            for d in a.out_dir:
                l2, a2 = newest_log(d, w)
                if l2 and (age is None or a2 < age):
                    log, age = l2, a2
            row += [round(cores, 3) if cores != "" else "",
                    st[1] if st else "", st[2] if st else "",
                    round(age, 1) if age is not None else ""]
            if (st and log and age is not None and age > a.stale
                    and log not in captured):
                captured.add(log)
                # In a thread: gdb takes minutes and the timeline must not gap.
                threading.Thread(target=safe_forensics, daemon=True,
                                 args=(os.path.dirname(log), w, log, age)).start()
        wr.writerow(row)
        fh.flush()
        prev_t = now


if __name__ == "__main__":
    main()
