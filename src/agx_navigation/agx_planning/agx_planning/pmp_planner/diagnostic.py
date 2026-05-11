import csv as _csv
import time as _time
import threading

import numpy as np


class TurnDiagnosticLogger:
    """Thread-safe CSV logger for comparing planned vs actual heading.

    Enabled by setting the ROS parameter ``diag_log_path`` to a non-empty
    file path (e.g. ``/tmp/pmp_diag.csv``).  Two row types are interleaved:

      source = "odom"    -- actual measured state, written on every /odom callback.
                           theta_deg, omega, v are EKF-fused chassis readings.
      source = "plan"    -- BVP-planned profile, written after each solve.
                           tick=0 is t=0 (anchored to actual state by BVP BC),
                           tick=k is the k-th committed/lookahead sample.
                           theta_deg is the BVP-planned heading (state).
                           omega, v are the PUBLISHED commands at this tick
                           (post chassis-model inversion), NOT the BVP state.
                           To recover the BVP state from omega_cmd, invert:
                             omega_state ~= gain_omega * omega_cmd - tau_omega * domega_cmd/dt
                           but in practice it is easier to log states
                           separately if needed.
                           lam_th, lam_om, alpha_cmd are only written on tick=0.

    Quick analysis (requires pandas + matplotlib)::

        import pandas as pd, matplotlib.pyplot as plt, numpy as np
        df = pd.read_csv('/tmp/pmp_diag.csv')
        odom = df[df.source == 'odom']
        p0   = df[(df.source == 'plan') & (df.tick == 0)]

        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].plot(odom.wall_s, odom.theta_deg, label='actual')
        axes[0].plot(p0.wall_s,   p0.theta_deg,   '.', label='plan t=0')
        axes[0].set_ylabel('heading (deg)')
        axes[0].legend()
        # lambda_omega at t=0: must be negative for a CCW turn.
        # If it hovers near 0 the cold-start costate fix isn't firing.
        axes[1].plot(p0.wall_s, p0.lam_om)
        axes[1].axhline(0, color='k', ls='--')
        axes[1].set_ylabel('lambda_omega(0)')
        # alpha command at t=0: should be near +/-alpha_max while turning.
        axes[2].plot(p0.wall_s, p0.alpha_cmd)
        axes[2].set_ylabel('alpha*(0)  [rad/s^2]')
        axes[2].set_xlabel('wall time (s)')
        plt.tight_layout(); plt.show()

    To plot the full planned heading *profile* for each solve::

        plan_all = df[df.source == 'plan']
        # group by chunk; each group is one BVP solve's committed arc
        for (chunk,), grp in plan_all.groupby(['chunk']):
            plt.plot(grp.tick, grp.theta_deg, label=f'chunk {chunk}')
        plt.xlabel('tick within chunk'); plt.ylabel('planned heading (deg)')
        plt.legend(); plt.show()
    """

    _HEADER = [
        "wall_s",
        "source",
        "traj_id",
        "chunk",
        "tick",
        "x",
        "y",
        "theta_deg",
        "omega",
        "v",
        "lam_th",
        "lam_om",
        "alpha_cmd",
    ]

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._f = open(path, "w", newline="", buffering=1)  # line-buffered
        self._w = _csv.writer(self._f)
        self._w.writerow(self._HEADER)
        self._t0 = _time.monotonic()

    def _now(self) -> float:
        return _time.monotonic() - self._t0

    def log_odom(
        self, x: float, y: float, theta: float, v: float, omega: float
    ) -> None:
        row = [
            f"{self._now():.4f}",
            "odom",
            "",
            "",
            "",
            f"{x:.3f}",
            f"{y:.3f}",
            f"{float(np.degrees(theta)):.3f}",
            f"{float(omega):.5f}",
            f"{float(v):.5f}",
            "",
            "",
            "",
        ]
        with self._lock:
            self._w.writerow(row)

    def log_plan(
        self,
        traj_id: int,
        chunk: int,
        thetas_deg: np.ndarray,
        omegas: np.ndarray,
        vs: np.ndarray,
        lam_th_0: float,
        lam_om_0: float,
        alpha_cmd_0: float,
    ) -> None:
        """Log the full planned heading profile for one BVP solve.

        The lam_th, lam_om, alpha_cmd columns are only populated on tick=0
        so the CSV stays readable without pivoting.
        """
        t0 = self._now()
        rows = []
        for i in range(len(thetas_deg)):
            rows.append(
                [
                    f"{t0:.4f}",
                    "plan",
                    traj_id,
                    chunk,
                    i,
                    "",
                    "",
                    f"{float(thetas_deg[i]):.3f}",
                    f"{float(omegas[i]):.5f}",
                    f"{float(vs[i]):.5f}",
                    f"{lam_th_0:.5f}" if i == 0 else "",
                    f"{lam_om_0:.5f}" if i == 0 else "",
                    f"{alpha_cmd_0:.5f}" if i == 0 else "",
                ]
            )
        with self._lock:
            self._w.writerows(rows)

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None
