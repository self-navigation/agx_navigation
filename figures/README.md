# figures/ — dated, indexed, regenerable

Every figure lives in a **dated directory** named for the day the underlying
measurement was read, with a `README.md` index. The convention exists because
this directory was previously an unsorted pile of ad-hoc PNGs made to win one
argument each: six months later nobody could say what `corrector_compare_cross.png`
compared, at which gains, on which plant — so none of it was usable for a write-up.

## The rules

1. **One directory per date**, `figures/YYYY-MM-DD/`, holding `render.py`, a
   `README.md` index, and the PNGs. All three are **committed**.
2. **Every directory has a `README.md`** listing each figure: what it shows, what
   data it was built from, what it establishes, and the command that regenerates it.
3. **The images are committed too**, which reverses the repo's usual "version the
   tool, not its output" rule. The reason is specific: these render from
   `soak_data/`, `traj_data/`, `uturn_traces/`, `epsilon_data/` — all gitignored,
   several representing hours of machine time on a plant that will not exist
   forever. The renderer alone does not reproduce the picture, so the picture is
   the record. `tools/plot_*.py` keeps the old rule, because its inputs are
   re-fetchable with a `just fetch-*`.
4. **State the plant.** Anything measured before 2026-08-07 was measured on a
   robot with `wheel.xacro`'s `mu2=0.7`, which could not steer on its own test
   terrain. A figure that does not name its plant cannot be compared with one
   that does.
5. **Never edit a dated figure in place after the fact.** If the analysis changes,
   render a new one under today's date and say in its README what it supersedes.
   A figure is a record of what was believed on a date.

## Index

| date | what it establishes |
| --- | --- |
| [2026-08-13](2026-08-13/) | the U-turn's deviation is one corner; the `q_cross` notch; `J` vs `max\|e_cross\|` disagree |
| [archive](archive/) | everything from before this convention — provenance mostly unrecoverable |
