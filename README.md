# rlinf-robodojo-runtime

RoboDojo environment runtime for [RLinf](https://github.com/RLinf/RLinf) and
[RPent](https://github.com/RLinf/RPent).

RoboDojo (`RoboDojo-Benchmark/RoboDojo`) is an Isaac Sim benchmark for dual-arm
ARX-X5 manipulation. Its Python code is a source tree whose top-level
directories (`env/`, `src/`, `task/`, `utils/`) are meant to sit on
`PYTHONPATH`; its dependency set pins Isaac Sim, RoboDojo's IsaacLab fork and
cuRobo to versions that conflict with the agent stack of any framework that
drives it.

This distribution is the RoboTwin-pattern answer — compare
`rlinf-robotwin-runtime` — in two parts:

1. The validated RoboDojo tree is vendored under `robodojo_runtime/_source`, and
   `robodojo_runtime.source_root()` hands back the directory to put on
   `PYTHONPATH`. RoboDojo's own imports (`import env.global_configs`,
   `from src.collect_client.collect_env import ...`) are untouched.
2. The simulator pins live in this distribution's `dependencies`, so an
   integrator's environment extra only has to name this package.

## Use

```python
import os
from robodojo_runtime import assets_root, source_root

os.environ["ROBODOJO_SOURCE_ROOT"] = str(source_root())  # optional override
os.environ["ROBODOJO_ASSETS_ROOT"] = "/data/robodojo"   # where Assets/ lives
```

Or from a shell, with the runtime installed in the interpreter that runs the
environment server:

```bash
export PYTHONPATH="$(python -c 'from robodojo_runtime import source_root; print(source_root())'):$PYTHONPATH"
export ROBODOJO_ASSETS_ROOT=/data/robodojo
```

`ROBODOJO_SOURCE_ROOT` overrides the vendored tree, so a developer checkout can
be used without reinstalling.

## What is not shipped

* `Assets/**` — the robot, object, material and layout data. Download it from
  the RoboDojo dataset and point `ROBODOJO_ASSETS_ROOT` at the directory that
  contains `Assets/`.
* NVIDIA's USD/material asset pack referenced by IsaacLab; see RoboDojo's
  `utils/ensure_usd_path.py` and `ROBODOJO_USD_ASSET_PREFIX`.
* Policy checkpoints, in particular the Pi_05 release.
* `XPolicyLab` — a separate checkout with its own launcher.

## What the vendored tree contains

This is not a pristine copy of upstream RoboDojo. Upstream's public release is
eval-only: it has no steppable environment client, and the integration needs
two things it does not ship. `SOURCE_LOCK.txt` records both, file by file:

* **an environment patch** — cuRobo planner, robot/scene/seed managers, the eval
  client and one task. Without it the wrapped environment does not reproduce
  the behaviour the recipes were validated against.
* **a collection client** — `src/collect_client/` plus
  `src/eval_client/traj_recorder.py`: a steppable environment built on
  RoboDojo's own `ObsManager`/`SeedManager`, and the trajectory/video recorder
  the environment server uses.

Both are MIT-compatible derivative work of RoboDojo; they are published here
with provenance rather than silently folded into the upstream files.

## Vendored patch

`env/global_configs.py` gains two environment variables, both defaulting to
upstream behaviour:

* `ROBODOJO_ROOT` — code root (defaults to `<parent of env/>`, unchanged).
* `ROBODOJO_ASSETS_ROOT` — data root that contains `Assets/` (defaults to the
  code root, unchanged).

This is what lets the code live in an installed wheel while the multi-gigabyte
scene data lives on its own volume. The exact upstream commit and the patch are
recorded in `SOURCE_LOCK.txt`.

## Licence

MIT, inherited from RoboDojo. See `LICENSE`.
