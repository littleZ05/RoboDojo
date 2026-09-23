
# rlinf-robodojo-runtime

RoboDojo Sim environment runtime for RLinf and RPent: the upstream RoboDojo tree
plus the simulator dependency stack it needs, so an integrator installs one
distribution instead of re-declaring Isaac Sim, RoboDojo's IsaacLab fork and
cuRobo pins next to its own agent dependencies.

```bash
uv pip install rlinf-robodojo-runtime --extra-index-url https://pypi.nvidia.com
```

Then:

```python
from robodojo_runtime import source_root

print(source_root())  # put this on PYTHONPATH
```

## Why this exists

RoboDojo's Python code is a source tree, not a package: `env/`, `src/`, `task/`
and `utils/` are expected to be importable from the repository root, and the
environment server boots Isaac Sim before importing them. Shipping it as a
normal Python package would mean rewriting RoboDojo's own imports, so this
distribution vendors the tree and reports the directory instead — the same
approach `rlinf-robotwin-runtime` takes for RoboTwin.

The dependency side is the other half. RoboDojo pins Isaac Sim 5.1, a
developer-maintained IsaacLab fork, cuRobo and a specific Torch family. Those
pins conflict with the dependencies of the agent framework that drives the
simulator (uvicorn, wrapt, filelock, torch and typing-extensions all disagree),
so leaving them in the integrator's own extras forced a growing list of
overrides. They are simulator facts, so they travel with the simulator here.

## What is in the wheel

* `robodojo_runtime/_source/` — the RoboDojo tree (`env/`, `src/`, `task/`,
  `utils/`, `env_cfg/`, `scripts/`) at the upstream commit recorded in
  `SOURCE_LOCK.txt`, plus the environment patch and collection client that
  upstream's eval-only release does not ship. `SOURCE_LOCK.txt` lists every
  difference; nothing is folded into upstream files silently.
* `robodojo_runtime.source_root()` / `assets_root()` / `env_config_path()`.
* One vendored patch: `env/global_configs.py` honours `ROBODOJO_ROOT` and
  `ROBODOJO_ASSETS_ROOT`, defaulting to upstream behaviour.

## What is not in the wheel

Scene data (`Assets/**`), NVIDIA's USD asset pack, policy checkpoints and
`XPolicyLab`. These are large or separately licensed; download them and point
`ROBODOJO_ASSETS_ROOT` and `ROBODOJO_USD_ASSET_PREFIX` at the local copies.

## Build and publish

```bash
git clone https://github.com/RLinf/RoboDojo.git -b rpent
cd RoboDojo
uv build
uv publish --token "$PYPI_TOKEN"
```

The `rpent` branch is a standalone branch holding this distribution; upstream
RoboDojo's own repository is untouched. RoboTwin's runtime is published the
same way from `RLinf/RoboTwin@rpent`.
