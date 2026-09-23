# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU bridge contracts; the simulator boundary is replaced by small fakes."""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robodojo_runtime import assets_root, bridge, source_root


def native_scene():
    events = []
    robots = [
        SimpleNamespace(
            type="target",
            arm_name=f"{arm}_arm",
            gripper_name=f"{arm}_ee",
            ee_type="gripper",
            gripper_move={"sign": sign, "mimic": [0, -1, 0]},
            gripper_scale=[0.0, 0.04],
        )
        for arm, sign in (("left", 1), ("right", -1))
    ]
    camera = {
        "color": np.zeros((2, 3, 3), dtype=np.uint8),
        "intrinsic_matrix": np.eye(3),
        "extrinsic_matrix": np.eye(4),
    }
    raw = {
        "vision": {key: dict(camera) for key in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        "state": {key: np.zeros(width) for key, width in zip(bridge.JOINT_KEYS, (6, 6, 1, 1))},
    }

    class Task:
        def reset(self, seed):
            events.append(("task_reset", seed))

    class Collect(Task):
        def reset(self, seed):
            events.append(("official", seed))
            self.take_action_cnt = [0]
            self.end_flag = [False]

    env = Collect()
    env.num_envs = 1
    env.step_lim = 3
    env.take_action_cnt = [0]
    env.end_flag = [False]
    env.seed_manager = SimpleNamespace(
        seed_info={0: {}}, get_seed_scene_info=lambda seed: {"template": seed}
    )
    env.traj_recorder = SimpleNamespace(reset_all=lambda: events.append("recorder"))
    env.scene_manager = SimpleNamespace(
        layout_manager=SimpleNamespace(
            set_saved_layout=lambda i, value: events.append(("layout", i, value))
        )
    )
    env.robot_manager = SimpleNamespace(
        robot_list=robots,
        process_name=lambda name: name + "_joint_state",
        set_origin_endpose=lambda: events.append("origin"),
        set_robot_init_state=lambda: events.append("robot"),
    )
    env.obs_manager = SimpleNamespace(
        desc_manager=SimpleNamespace(get_one_description=lambda: ["Pick the object."]),
        reset=lambda: events.append("obs_reset"),
    )
    env.reward_manager = SimpleNamespace(
        get_reward=lambda **kw: [0.25],
        init_state=lambda: events.append("reward_init"),
    )
    env.get_score = lambda: events.append("score")
    env.run_reward = lambda: events.append("reward")
    env.setup_scene = lambda: events.append("setup")
    env.get_obs = lambda env_idx: raw
    env.apply_target = lambda control, index: events.append(("action", index, control))
    env.is_success = lambda env_idx: env.take_action_cnt[0] >= 2
    return env, events


@pytest.fixture
def native():
    obj = object.__new__(bridge.NativeEnv)
    obj.env, obj.events = native_scene()
    obj.app = None
    obj.task_config = {}
    obj.action_dims = (6, 6, 1, 1)
    obj.random = False
    obj._closed = False
    obj._terminated = obj._truncated = False
    obj._last_obs = None
    return obj


def test_import_without_isaac_and_asset_root(monkeypatch, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['isaacsim']=None; sys.modules['isaaclab']=None; import robodojo_runtime.bridge",
        ],
        check=True,
    )
    monkeypatch.setenv("ROBODOJO_SOURCE_ROOT", str(tmp_path / "code"))
    monkeypatch.delenv("ROBODOJO_ASSETS_ROOT", raising=False)
    assert assets_root() == source_root()
    monkeypatch.setenv("ROBODOJO_ASSETS_ROOT", str(tmp_path / "data"))
    assert assets_root() == tmp_path / "data"


def test_reset_layout_and_random_template(native, monkeypatch):
    native.reset(0)
    assert native.events == [("official", 0), "score"]
    native.events.clear()
    native.random = True
    seeds = iter([0, 7654321])
    monkeypatch.setattr(bridge.random, "randrange", lambda *a: next(seeds))
    native.reset(99)
    assert native.env.env_seeds == [7654321]
    assert native.env.current_env_seed_map == {0: 7654321}
    assert native.env.take_action_cnt == [0]
    assert native.events == [
        "recorder",
        ("layout", 0, {"template": 0}),
        ("task_reset", [7654321]),
        "obs_reset",
        "setup",
        "origin",
        "robot",
        "reward_init",
        "reward",
        "score",
    ]


def test_observations_preserve_calibration(native):
    obs = native.get_obs()
    assert obs["full_image"].shape == (2, 3, 3)
    assert obs["state"].shape == (14,)
    assert obs["state"].dtype == np.float32
    assert obs["instruction"] == "Pick the object."
    for camera in obs["vision"].values():
        np.testing.assert_array_equal(camera["intrinsic_matrix"], np.eye(3))
        np.testing.assert_array_equal(camera["extrinsic_matrix"], np.eye(4))


def test_chunk_latches_terminal_and_scales_grippers(native):
    actions = np.zeros((50, 14))
    actions[:, 12:] = [1.0, 0.0]
    obs, reward, terminated, truncated, info = native.step(actions)
    assert reward == 0.5 and terminated and not truncated
    assert info["executed_steps"] == 2
    controls = [
        event[2] for event in native.events if isinstance(event, tuple) and event[0] == "action"
    ]
    assert len(controls) == 2
    assert controls[0]["left_ee_joint_state"]["position"] == [0.04, -0.04]
    assert controls[0]["right_ee_joint_state"]["position"] == [0.04, -0.04]
    result = native.step(actions)
    assert result[0] is obs and result[1] == 0 and result[-1]["executed_steps"] == 0
    native.reset(0)
    assert not native._terminated


def test_truncation_and_eval_fair_never_read_reward(native):
    native.env.is_success = lambda **kw: False
    assert native.step(np.zeros((50, 14)))[3]
    native.reset(0)
    native.env.reward_manager.get_reward = lambda **kw: pytest.fail("reward read")
    action = dict(zip(bridge.JOINT_KEYS, (np.zeros(6), np.zeros(6), [0], [1])))
    for _ in range(3):
        native.apply_action(action, eval_fair=True)
    with pytest.raises(RuntimeError, match="budget"):
        native.apply_action(action, eval_fair=True)


def test_close_order_even_if_capture_fails(native, monkeypatch):
    events = []
    native.app = SimpleNamespace(close=lambda: events.append("app"))
    monkeypatch.setattr(native, "_close_capture_manager", lambda: events.append("capture"))
    monkeypatch.setattr(native, "_stop_replicator", lambda: events.append("replicator"))
    native.close(False)
    native.close()
    assert events == ["capture", "replicator", "app"]


def test_launch_precedes_simulator_import(monkeypatch, tmp_path):
    events = []

    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            pass

        def __init__(self, args):
            assert args.enable_cameras and "isaacsim.sensors.camera" in args.kit_args
            events.append("launch")
            self.app = SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=Launcher))

    def create(config, app):
        assert events == ["launch", "config"]
        assert os.environ["ROBODOJO_ROOT"] == str(tmp_path)
        return native_scene()[0]

    monkeypatch.setitem(
        sys.modules, "src.collect_client.collect_env", SimpleNamespace(create_collect_env=create)
    )
    monkeypatch.setattr(bridge, "_build_env_cfg", lambda args: events.append("config"))
    monkeypatch.setenv("ROBODOJO_ASSETS_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROBODOJO_SOURCE_ROOT", str(tmp_path))
    bridge.NativeEnv({"task_name": "pick"})
    assert events == ["launch", "config"]


def test_vector_partial_reset_validation_and_global_seed_index(monkeypatch):
    class Slot:
        def __init__(self, cfg):
            self.native = object.__new__(bridge.NativeEnv)
            self.native.env, self.events = native_scene()
            self.native.task_config = {}
            self.native.action_dims = (6, 6, 1, 1)
            self.native.random = False
            self.native._terminated = self.native._truncated = False
            self.native._last_obs = None
            self.closed = []

        def call(self, method, *args):
            return getattr(self.native, method)(*args)

        def close(self, clear_cache):
            self.closed.append(clear_cache)

    monkeypatch.setattr(bridge, "_RemoteEnv", Slot)
    env = bridge.VectorEnv({"task_name": "pick"}, 2, [0, 1])
    env.reset()
    env.step(np.zeros((2, 1, 14)))
    env.reset(env_idx=[1], env_seeds=[7, 8])
    assert env.envs[0].native.env.take_action_cnt == [1]
    assert env.envs[1].native.env.take_action_cnt == [0]
    assert env.env_seeds == [0, 8]
    parts = {k: np.zeros((2, 1, w)) for k, w in zip(bridge.JOINT_KEYS, (6, 6, 1, 1))}
    obs, rewards, terminated, truncated, infos = env.step(parts)
    assert rewards.shape == terminated.shape == truncated.shape == (2,)
    assert len(obs) == len(infos) == 2
    for bad in (np.zeros((2, 0, 14)), np.zeros((1, 2, 14)), np.full((2, 1, 14), np.nan)):
        with pytest.raises(ValueError):
            env.step(bad)
    with pytest.raises(ValueError):
        env.reset(env_idx=[1], env_seeds=[9])
    with pytest.raises(IndexError):
        env.reset(env_idx=[2])
    env.close(False)
    env.close()
    assert [slot.closed for slot in env.envs] == [[False], [False]]


def test_config_keeps_native_camera_and_pipeline_settings(monkeypatch, tmp_path):
    from pathlib import Path

    import yaml

    from robodojo_runtime import VENDORED_SOURCE_ROOT

    root = VENDORED_SOURCE_ROOT
    monkeypatch.setitem(
        sys.modules,
        "env.global_configs",
        SimpleNamespace(
            BENCHMARK="RoboDojo",
            ROOT_DIR=str(root),
            ENV_CONFIG_PATH=str(root / "env_cfg"),
        ),
    )

    def load_yaml(path):
        with Path(path).open() as stream:
            return yaml.safe_load(stream) or {}

    monkeypatch.setitem(sys.modules, "utils.load_file", SimpleNamespace(load_yaml=load_yaml))
    calls = []

    def randomize(cfg):
        calls.append("randomization")
        return cfg

    def process(cfg, *, task_name):
        calls.append(task_name)
        return cfg, None

    monkeypatch.setitem(
        sys.modules,
        "utils.pipeline_utils",
        SimpleNamespace(
            process_config=process,
            process_randomization=randomize,
            resolve_random_task_num_envs=lambda task, count, sim: count,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "task.RoboDojo.task_registry",
        SimpleNamespace(
            task_config_path=lambda directory, name: Path(directory) / f"{name}.yml",
        ),
    )
    cfg = bridge._build_env_cfg(
        SimpleNamespace(
            task="fill_pen_holder",
            env_cfg_type="arx_x5",
            num_envs=1,
            cuda_device=0,
            save_dir=str(tmp_path),
        )
    )
    assert calls == ["randomization", "fill_pen_holder"]
    assert cfg.collect_cfg.save_dir == str(tmp_path)
    assert cfg.collect_cfg.num_envs == cfg.sim.scene.num_envs == 1
    assert cfg.sim.seed == [0]
    assert cfg.camera.default_frequency == 25
    assert cfg.collect_cfg.observation.vision.depth
    assert cfg.collect_cfg.observation.vision.intrinsic_matrix
    assert cfg.collect_cfg.observation.vision.extrinsic_matrix
    for name in ("common", "cam_head", "cam_left_wrist", "cam_right_wrist"):
        assert cfg.camera.annotator[name].distance_to_image_plane_capture.device == "cpu"


def test_seed_checks_and_optional_ee_grippers(native):
    def layout(seed):
        if seed != 2:
            raise ValueError("missing layout")
        return {}

    native.env.seed_manager.get_seed_scene_info = layout
    assert native.check_seeds([2, 1, -1]) == [True, False, False]
    native.random = True
    assert native.check_seeds([0, 999999999, -1, 1000000000]) == [True, True, False, False]
    poses = {f"{arm}_ee_pose": [0, 0, 1, 1, 0, 0, 0] for arm in ("left", "right")}
    native.env.robot_manager.solve_ik = lambda **kw: {"status": "Success", "joint_value": [0] * 6}
    native.apply_action(poses, "ee")
    control = native.events[-1][2]
    assert set(control) == {"left_arm_joint_state", "right_arm_joint_state"}
    with pytest.raises(ValueError, match="finite"):
        native.apply_action({**poses, "left_ee_pose": [float("nan")] * 7}, "ee")


def test_worker_reports_errors_and_closes_on_eof(monkeypatch):
    import multiprocessing
    import threading

    events = []

    class Native:
        def __init__(self, config):
            events.append(config)

        def reset(self, seed):
            raise ValueError("bad seed")

        def close(self, clear_cache):
            events.append(("close", clear_cache))

    monkeypatch.setattr(bridge, "NativeEnv", Native)
    owner, child = multiprocessing.Pipe()
    thread = threading.Thread(target=bridge._worker, args=(child, {"task_name": "pick"}))
    thread.start()
    try:
        assert owner.poll(5) and owner.recv() == (True, None)
        owner.send(("reset", (9,)))
        assert owner.poll(5)
        ok, error = owner.recv()
        assert not ok and "ValueError: bad seed" in error
    finally:
        owner.close()
        thread.join(5)
    assert not thread.is_alive()
    assert events == [{"task_name": "pick"}, ("close", False)]
