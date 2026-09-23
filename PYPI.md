
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

## Simulator bridge (0.3.0)

`robodojo_runtime.bridge.VectorEnv(task_config, n_envs, env_seeds)` provides
`reset(env_idx=None, env_seeds=...)`, `get_obs()`, `step(actions)`,
`check_seeds(seeds)`, and `close(clear_cache=True)`. Importing the bridge
does not import Isaac Sim; construction starts AppLauncher before loading the
vendored environment. The bridge reuses the collection client and the native
observation, robot, scene and reward managers.

Set `task_config.task_name` and optionally `env_cfg_type` (default
`arx_x5`), `source_root`, `save_dir`, `cuda_device`, `headless`,
`max_episode_steps`, and `random`. Official reset seeds are layout IDs;
random reset selects a saved template and generates a fresh scene seed,
matching the RPent random-layout path. Random mode requires saved templates.

Actions have shape `(N,H,A)`, or four joint-key arrays shaped
`(N,H,width)`. The key/state order is left arm, right arm, left gripper,
right gripper; `action_dims` defaults to `[6,6,1,1]`.
Observations include `full_image`, optional wrist images, flat `state`,
`instruction`, and native `vision` including intrinsic/extrinsic metadata.
Step returns observations, chunk-total rewards, latched termination/truncation
arrays of shape `(N,)`, and per-slot info dictionaries. It never auto-resets.

One slot runs on the caller's main thread. Multiple slots use spawned processes,
one Isaac application per slot, because native reset and physics affect the
whole scene. Protect application entry points with `if __name__ == "__main__"`.
Partial resets use a full, globally indexed seed list and leave other slots
untouched. Multi-slot operation has CPU contract coverage only; GPU throughput,
resource use, rollout behavior and task success remain unverified.

The bridge is outside `_source/`; the validated vendored tree is unchanged.
`assets_root()` defaults to the code root, or `ROBODOJO_ASSETS_ROOT` when
set, and always names the directory **containing** `Assets/`.
