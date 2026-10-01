#!/usr/bin/env python3
"""Plot a tools/freeze_watch.py vitals.csv: host load, per-worker CPU/RAM, and
log silence, with freezes (log age > --stale) shaded per worker.

    .venv/bin/python tools/plot_vitals.py vitals.csv -o vitals.png
"""
import argparse
import csv
import datetime as dt
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def col(rows, k):
    return [float(r[k]) if r.get(k) not in ("", None) else float("nan") for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--stale", type=float, default=150.0)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    t = [dt.datetime.fromtimestamp(float(r["t"])) for r in rows]
    workers = sorted({int(m.group(1)) for k in rows[0]
                      for m in [re.match(r"w(\d+)_cpu$", k)] if m})

    fig, ax = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    ax[0].plot(t, col(rows, "host_cpu_cores"), label="host busy cores")
    ax[0].plot(t, col(rows, "load1"), label="load1")
    ax[0].set_ylabel("cores")
    ax[0].legend(loc="upper left")

    ax[1].plot(t, [x / 2**30 for x in col(rows, "mem_used")], label="RAM used")
    ax[1].set_ylabel("GiB")
    g = ax[1].twinx()
    g.plot(t, col(rows, "gpu_util"), color="tab:green", alpha=.6, label="GPU %")
    g.set_ylabel("GPU %")
    ax[1].legend(loc="upper left")
    g.legend(loc="upper right")

    for k in ("psi_cpu_some", "psi_cpu_full", "psi_mem_some", "psi_io_some", "psi_io_full"):
        ax[2].plot(t, col(rows, k), label=k)
    ax[2].set_ylabel("PSI avg10 %")
    ax[2].legend(loc="upper left", ncol=5, fontsize=8)

    for w in workers:
        ax[3].plot(t, col(rows, f"w{w}_cpu"), label=f"w{w}", lw=.8)
        ax[4].plot(t, col(rows, f"w{w}_log_age"), label=f"w{w}", lw=.8)
        age = col(rows, f"w{w}_log_age")
        for i in range(1, len(t)):
            if age[i] > a.stale:
                for x in (ax[3], ax[4]):
                    x.axvspan(t[i - 1], t[i], color=f"C{workers.index(w)}", alpha=.08)
    ax[3].set_ylabel("stack CPU (cores)")
    ax[3].legend(loc="upper left", ncol=len(workers), fontsize=8)
    ax[4].axhline(a.stale, color="k", ls="--", lw=.8)
    ax[4].set_ylabel("stack log silence (s)")
    ax[4].set_yscale("symlog", linthresh=10)
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].set_title(f"Campaign vitals ({a.csv}); shaded = log silent > {a.stale:.0f} s")
    fig.tight_layout()
    fig.savefig(a.out or a.csv.rsplit(".", 1)[0] + ".png", dpi=110)


if __name__ == "__main__":
    main()
