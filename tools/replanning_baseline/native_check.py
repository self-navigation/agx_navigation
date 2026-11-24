"""Small installed-native checks outside benchmark sample timing."""

from pathlib import Path
import shlex
import sys


def check_incremental_native():
    """Require complete-state reuse from the installed Python and C++ code."""
    try:
        import numpy as np
        import skfmm

        phi = np.ones((9, 17))
        phi[4, 3] = -1.0
        speed = np.ones(phi.shape)
        mask = np.zeros(phi.shape, dtype=bool)
        mask[:, 8] = True
        state = skfmm.TravelTimeState(np.ma.array(phi, mask=mask), speed,
                                     track_dependencies=True)
        state.advance()
        before = state.stats
        expected = np.ma.filled(state.result(), np.nan)
        mask[4, 13] = True
        report = state.replay(speed, mask, [])
        after = state.stats
        if report.get("status") != "replay" or report.get("reason") != "complete_state_reused":
            raise ValueError(f"expected complete_state_reused, got {report}")
        if not before["complete"] or not after["complete"] or after["batches"] != before["batches"]:
            raise ValueError("hidden edit did not preserve the complete march")
        if not np.array_equal(np.ma.filled(state.result(), np.nan), expected, equal_nan=True):
            raise ValueError("hidden edit changed the completed field")
    except (ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        executable = shlex.quote(sys.executable)
        bindir = shlex.quote(str(Path(sys.executable).parent))
        raise ValueError(
            f"Incremental native preflight failed: {error}. "
            "Rebuild the installed scikit-fmm extension from the repository root "
            "with the same Python used by this benchmark: "
            f'PATH={bindir}:"$PATH" {executable} -m pip install '
            "--no-build-isolation --no-deps --force-reinstall ./depend/scikit-fmm"
        ) from error
    return {"complete_state_reuse": "passed", "replay": dict(report)}
