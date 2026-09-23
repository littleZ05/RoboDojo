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

"""Lazy simulator bridge for RLinf and RPent.

Each native scene is owned by one process. A single slot runs on the caller's
main thread; multiple slots use spawned workers because CollectEnv resets and
advances the whole Isaac scene. No simulator modules are imported at module load.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import random
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from robodojo_runtime import assets_root, source_root

logger = logging.getLogger(__name__)
JOINT_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
)
DEFAULT_ACTION_DIMS = (6, 6, 1, 1)


def _random_reset(env) -> None:
    """Fresh random layout: random template + large random env seed.

    Mirrors the official eval's random-layout rollout (eval_env.py): the std
    template only stores placement CONFIG and ClutteredGenerator samples actual
    object positions from the ENV SEED, so a random template + fresh seed =
    a fresh layout. The naive collect_env ``random_mode`` (saved_layouts=None)
    leaves the scene EMPTY — do not use it.
    """
    template = env.seed_manager.get_seed_scene_info(
        random.randrange(len(env.seed_manager.seed_info))
    )
    fresh = [random.randrange(0, 1_000_000_000) for _ in range(env.num_envs)]
    env.traj_recorder.reset_all()
    env.env_seeds = fresh
    env.success = [True] * env.num_envs
    env.end_flag = [False] * env.num_envs
    env.take_action_cnt = [0] * env.num_envs
    env.current_env_seed_map = dict(enumerate(fresh))
    for idx in range(env.num_envs):
        env.scene_manager.layout_manager.set_saved_layout(idx, template)
    # Bypass CollectEnv.reset (it would overwrite the template via
    # get_seed_scene_info(env_seed)); call the task-class reset directly.
    for cls in type(env).__mro__[1:]:
        if "reset" in cls.__dict__:
            cls.reset(env, seed=fresh)
            break
    env.obs_manager.reset()
    env.setup_scene()
    env.robot_manager.set_origin_endpose()
    env.robot_manager.set_robot_init_state()
    env.reward_manager.init_state()
    env.run_reward()
    if hasattr(env, "get_score"):
        try:
            env.get_score()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_score registration failed: %s", exc)


def _standard_reset(env, layout: int) -> None:
    env.reset(seed=layout)
    if hasattr(env, "get_score"):
        try:
            env.get_score()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_score registration failed: %s", exc)


def _control_info_from_action(env, action: dict, action_type: str) -> dict:
    """Convert a policy action dict to control_info (mirrors eval_env.take_action_batch)."""
    control_info: dict[str, Any] = {}
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        name = robot.arm_name.split("_")[0]
        if action_type == "joint":
            key_name = env.robot_manager.process_name(robot.arm_name)
            control_info[key_name] = {"position": list(action[key_name])}
            gripper_key = env.robot_manager.process_name(robot.gripper_name)
            if robot.ee_type == "gripper":
                val = float(np.clip(action[gripper_key][0], 0, 1))
                if robot.gripper_move["sign"] == 1:
                    val = (
                        val * (robot.gripper_scale[1] - robot.gripper_scale[0])
                        + robot.gripper_scale[0]
                    )
                else:
                    val = (1 - val) * (
                        robot.gripper_scale[1] - robot.gripper_scale[0]
                    ) + robot.gripper_scale[0]
                vals = [
                    val,
                    val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                ]
                control_info[gripper_key] = {"position": vals}
        elif action_type == "ee":
            key_name = f"{name}_ee_pose"
            obs_name = env.robot_manager.process_name(robot.arm_name)
            target_pose = action[key_name]
            ik_result = env.robot_manager.solve_ik(target_pose=target_pose, env_idx=0, robot=robot)
            if ik_result["status"] == "Success":
                control_info[obs_name] = {"position": ik_result["joint_value"]}
            # Gripper must be controlled in ee mode too (mirror
            # eval_env.take_action_batch); otherwise set_gripper / move_to's
            # gripper arg are silently ignored.
            gripper_key = env.robot_manager.process_name(robot.gripper_name)
            if robot.ee_type == "gripper" and gripper_key in action:
                val = float(np.clip(action[gripper_key][0], 0, 1))
                if robot.gripper_move["sign"] == 1:
                    val = (
                        val * (robot.gripper_scale[1] - robot.gripper_scale[0])
                        + robot.gripper_scale[0]
                    )
                else:
                    val = (1 - val) * (
                        robot.gripper_scale[1] - robot.gripper_scale[0]
                    ) + robot.gripper_scale[0]
                vals = [
                    val,
                    val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                ]
                control_info[gripper_key] = {"position": vals}
    return control_info


def _build_env_cfg(args: Any) -> Any:
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import (
        process_config,
        process_randomization,
        resolve_random_task_num_envs,
    )

    task_registry = __import__(f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"])
    collect_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg_type + ".yml"))
    vision_cfg = collect_cfg.setdefault("observation", {}).setdefault("vision", {})
    vision_cfg["depth"] = True
    vision_cfg["intrinsic_matrix"] = True
    vision_cfg["extrinsic_matrix"] = True
    collect_cfg["task_name"] = args.task
    collect_cfg["num_envs"] = args.num_envs
    collect_cfg["device_id"] = args.cuda_device
    collect_cfg["save_dir"] = args.save_dir
    camera_cfg = load_yaml(
        os.path.join(ENV_CONFIG_PATH, "camera", collect_cfg["config"]["camera"] + ".yml")
    )
    camera_cfg = OmegaConf.merge(
        camera_cfg,
        OmegaConf.create(
            {
                "annotator": {
                    "common": {
                        "distance_to_image_plane_capture": {
                            "type": "distance_to_image_plane",
                            "device": "cpu",
                        }
                    },
                    "cam_head": {
                        "distance_to_image_plane_capture": {
                            "type": "distance_to_image_plane",
                            "device": "cpu",
                        }
                    },
                    "cam_left_wrist": {
                        "distance_to_image_plane_capture": {
                            "type": "distance_to_image_plane",
                            "device": "cpu",
                        }
                    },
                    "cam_right_wrist": {
                        "distance_to_image_plane_capture": {
                            "type": "distance_to_image_plane",
                            "device": "cpu",
                        }
                    },
                }
            }
        ),
    )
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "sim",
                    collect_cfg["config"]["sim"] + ".yml",
                )
            ),
            "scene": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "scene",
                    collect_cfg["config"]["scene"] + ".yml",
                )
            ),
            "camera": camera_cfg,
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "robot",
                    collect_cfg["config"]["robot"] + ".yml",
                )
            ),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    os.path.join(ROOT_DIR, "task", BENCHMARK, "config"),
                    args.task,
                )
            ),
            "collect_cfg": collect_cfg,
            "eval_cfg": collect_cfg,
        }
    )
    capped = resolve_random_task_num_envs(args.task, args.num_envs, env_cfg.sim)
    OmegaConf.update(env_cfg, "sim.scene.num_envs", capped, force_add=True)
    OmegaConf.update(env_cfg, "collect_cfg.num_envs", capped, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=args.task)
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        collect_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )
    env_cfg.sim.seed = [0 for _ in range(capped)]
    return env_cfg


def validate_action(action: dict, action_type: str, action_dims=DEFAULT_ACTION_DIMS) -> dict:
    """Validate a native joint or end-pose target before executing any motion."""
    keys = (
        JOINT_KEYS
        if action_type == "joint"
        else ("left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state")
    )
    widths = action_dims if action_type == "joint" else (7, 7, 1, 1)
    if action_type not in ("joint", "ee"):
        raise ValueError("action_type must be joint or ee")
    result = {}
    for key, width in zip(keys, widths):
        if action_type == "ee" and key.endswith("_ee_joint_state") and key not in action:
            continue
        value = np.asarray(action[key], dtype=np.float64)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain {width} finite values")
        result[key] = value
    return result


class NativeEnv:
    """Own one steppable scene and its Isaac application on the main thread."""

    def __init__(self, task_config: dict):
        self._closed = False
        self.env = None
        self.app = None
        self.task_config = task_config
        self.action_dims = tuple(task_config.get("action_dims", DEFAULT_ACTION_DIMS))
        if len(self.action_dims) != 4 or any(
            not isinstance(d, (int, np.integer)) or d <= 0 for d in self.action_dims
        ):
            raise ValueError("action_dims must contain four positive integer widths")
        root = (
            Path(task_config["source_root"]).expanduser().resolve()
            if task_config.get("source_root")
            else source_root()
        )
        os.environ["ROBODOJO_SOURCE_ROOT"] = str(root)
        os.environ["ROBODOJO_ROOT"] = str(root)
        os.environ["ROBODOJO_ASSETS_ROOT"] = str(assets_root())
        sys.path.insert(0, str(root))
        self.random = bool(task_config.get("random", False))
        self._terminated = False
        self._truncated = False
        self._last_obs = None
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.headless = task_config.get("headless", True)
        args.enable_cameras = True
        args.device = task_config.get("device", "cuda:0")
        args.kit_args = task_config.get("kit_args", "") + (
            " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
        )
        args.task = task_config["task_name"]
        args.env_cfg_type = task_config.get("env_cfg_type", "arx_x5")
        args.num_envs = 1
        args.cuda_device = task_config.get("cuda_device", 0)
        args.save_dir = task_config.get("save_dir", os.getcwd())
        self.app = AppLauncher(args).app
        try:
            # The source tree imports Isaac modules, so import it only after launch.
            from src.collect_client.collect_env import create_collect_env

            self.env = create_collect_env(_build_env_cfg(args), self.app)
        except BaseException:
            try:
                self.close(False)
            except Exception:
                logger.exception("Failed to close the partially constructed scene")
            raise

    def reset(self, seed: int | None = None) -> None:
        """Reset using an official layout or a random template and fresh seed."""
        if self.random:
            _random_reset(self.env)
        else:
            _standard_reset(self.env, seed)
        self._terminated = False
        self._truncated = False
        self._last_obs = None

    def get_native_obs(self) -> dict:
        """Return unflattened vision (including calibration), state and language."""
        obs = self.env.get_obs(env_idx=0)
        instruction = self.env.obs_manager.desc_manager.get_one_description()[0]
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or "<" in instruction
            or ">" in instruction
        ):
            raise ValueError(
                "RoboDojo instruction is empty or contains unresolved template markers"
            )
        obs["instruction"] = instruction
        return obs

    def get_obs(self) -> dict:
        """Return a training observation while retaining native camera metadata."""
        if self._last_obs is not None and (self._terminated or self._truncated):
            return self._last_obs
        native = self.get_native_obs()
        obs = {
            "state": np.concatenate(
                [np.asarray(native["state"][k]).reshape(-1) for k in JOINT_KEYS]
            ).astype(np.float32),
            "instruction": native["instruction"],
            "vision": native["vision"],
        }
        for camera, key in (
            ("cam_head", "full_image"),
            ("cam_left_wrist", "left_wrist_image"),
            ("cam_right_wrist", "right_wrist_image"),
        ):
            color = native["vision"].get(camera, {}).get("color")
            if color is not None:
                obs[key] = np.asarray(color, dtype=np.uint8)
        if "full_image" not in obs:
            raise ValueError("RoboDojo observation requires cam_head.color")
        self._last_obs = obs
        return obs

    def apply_action(
        self, action: dict, action_type: str = "joint", *, eval_fair: bool = False
    ) -> None:
        """Apply a native target with the existing RPent control/count semantics."""
        action = validate_action(action, action_type, self.action_dims)
        env = self.env
        if eval_fair and env.take_action_cnt[0] >= env.step_lim:
            raise RuntimeError("Episode step budget exhausted")
        control = _control_info_from_action(env, action, action_type)
        if eval_fair or (env.take_action_cnt[0] < env.step_lim and not env.end_flag[0]):
            env.take_action_cnt[0] += 1
        env.apply_target(control, 0)
        self._last_obs = None

    def step(self, actions: np.ndarray) -> tuple:
        """Execute a horizon and aggregate rewards; terminal slots remain latched."""
        rewards = 0.0
        executed = 0
        for action in actions:
            if self._terminated or self._truncated:
                break
            if self.env.take_action_cnt[0] >= self.env.step_lim or self.env.end_flag[0]:
                self._truncated = True
                break
            split = np.split(action, np.cumsum(self.action_dims)[:-1])
            self.apply_action(dict(zip(JOINT_KEYS, split)))
            executed += 1
            rewards += float(self.env.reward_manager.get_reward(final_check=True)[0])
            self._terminated = bool(self.env.is_success(env_idx=0))
            limit = min(
                self.env.step_lim, self.task_config.get("max_episode_steps", self.env.step_lim)
            )
            self._truncated = bool(self.env.take_action_cnt[0] >= limit or self.env.end_flag[0])
        return (
            self.get_obs(),
            rewards,
            self._terminated,
            self._truncated,
            {
                "success": self._terminated,
                "executed_steps": executed,
            },
        )

    def check_seeds(self, seeds: list[int]) -> list[bool]:
        """Check saved-layout readability, or random seed range, without reset."""
        result = []
        for seed in seeds:
            if self.random:
                result.append(isinstance(seed, (int, np.integer)) and 0 <= seed < 1_000_000_000)
                continue
            try:
                self.env.seed_manager.get_seed_scene_info(seed)
                result.append(True)
            except (ValueError, OSError):
                result.append(False)
        return result

    def close(self, clear_cache: bool = True) -> None:
        """Release cameras, Replicator and the app, once, on their owning thread."""
        if self._closed:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Isaac teardown must run on the main thread")
        self._closed = True
        try:
            try:
                self._close_capture_manager()
            finally:
                if self.app is not None:
                    self._stop_replicator()
        finally:
            if self.app is not None:
                self.app.close()
            if clear_cache:
                import gc

                gc.collect()

    def _close_capture_manager(self) -> None:
        manager = getattr(self.env, "capture_manager", None)
        if manager is None:
            manager = getattr(getattr(self.env, "obs_manager", None), "capture_manager", None)
        destroy = getattr(manager, "destroy", None)
        if callable(destroy):
            # RoboDojo's CameraView destructor still refers to the old tiled
            # sensor fields; detach its current annotators before destroy().
            for camera in manager.tiled_cameras:
                for annotator in camera._annotators.values():
                    annotator.detach([camera._render_product_path])
                camera._annotators.clear()
            destroy()

    @staticmethod
    def _stop_replicator() -> None:
        import omni.replicator.core as rep
        from omni.syntheticdata import SyntheticData

        # Destroyed camera graphs must not be evaluated on the next app tick.
        # usd=False clears cached handles after capture.destroy removed graphs.
        SyntheticData.Get().reset(usd=False)
        rep.orchestrator.set_capture_on_play(False)
        rep.orchestrator.stop()
        rep.orchestrator.wait_until_complete()


def _worker(connection, task_config: dict) -> None:
    """Serve one native scene; EOF also releases owned simulator resources."""
    native = None
    try:
        native = NativeEnv(task_config)
        connection.send((True, None))
        while True:
            method, args = connection.recv()
            try:
                result = getattr(native, method)(*args)
                connection.send((True, result))
            except Exception:  # noqa: BLE001 - preserve native errors across the process boundary
                connection.send((False, traceback.format_exc()))
            if method == "close":
                break
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 - report startup failure to the owner
        connection.send((False, traceback.format_exc()))
    finally:
        if native is not None:
            native.close(False)
        connection.close()


class _RemoteEnv:
    def __init__(self, task_config: dict):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, task_config))
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.connection.close()
            self.process.join()
            raise

    def _receive(self):
        ok, result = self.connection.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def call(self, method: str, *args):
        self.connection.send((method, args))
        return self._receive()

    def close(self, clear_cache: bool):
        try:
            self.call("close", clear_cache)
        finally:
            self.connection.close()
            self.process.join()


class VectorEnv:
    """RLinf bridge with isolated slots and globally indexed reset seeds.

    task_config accepts task_name, env_cfg_type, source_root, save_dir,
    cuda_device, headless, random, max_episode_steps, and action_dims.
    Vector actions and state use left arm, right arm, left gripper, right
    gripper ordering; action_dims defaults to (6, 6, 1, 1) for ARX-X5.
    Dictionary chunks have those four joint keys, each shaped (N, H, width).
    """

    def __init__(self, task_config: dict, n_envs: int, env_seeds: list[int]):
        if n_envs <= 0 or len(env_seeds) != n_envs:
            raise ValueError("n_envs must be positive with one seed per slot")
        self.n_envs = n_envs
        self.env_seeds = list(env_seeds)
        self.action_dims = tuple(task_config.get("action_dims", DEFAULT_ACTION_DIMS))
        self._closed = False
        self.envs = []
        try:
            for index in range(n_envs):
                config = dict(task_config)
                if n_envs > 1:
                    config["save_dir"] = str(
                        Path(config.get("save_dir", os.getcwd())) / f"env_{index}"
                    )
                self.envs.append(NativeEnv(config) if n_envs == 1 else _RemoteEnv(config))
        except BaseException:
            self.close(False)
            raise

    def _call(self, index: int, method: str, *args):
        env = self.envs[index]
        return (
            env.call(method, *args) if isinstance(env, _RemoteEnv) else getattr(env, method)(*args)
        )

    def reset(self, env_idx=None, env_seeds=None) -> None:
        """Reset selected slots without changing the other scenes."""
        seeds = self.env_seeds if env_seeds is None else list(env_seeds)
        if len(seeds) != self.n_envs:
            raise ValueError("env_seeds must contain one seed per global slot")
        indices = (
            list(range(self.n_envs))
            if env_idx is None
            else ([env_idx] if isinstance(env_idx, (int, np.integer)) else list(env_idx))
        )
        if len(set(indices)) != len(indices) or any(i < 0 or i >= self.n_envs for i in indices):
            raise IndexError("invalid or duplicate environment index")
        for index in indices:
            self._call(index, "reset", seeds[index])
            self.env_seeds[index] = seeds[index]

    def get_obs(self) -> list[dict]:
        """Return observations for every slot in slot order."""
        return [self._call(i, "get_obs") for i in range(self.n_envs)]

    def step(self, actions) -> tuple:
        """Execute (N,H,A) vectors or four joint-key arrays, without auto-reset."""
        if isinstance(actions, dict):
            parts = [np.asarray(actions[key]) for key in JOINT_KEYS]
            for part, width in zip(parts, self.action_dims):
                if part.ndim != 3 or part.shape[2] != width:
                    raise ValueError("dictionary actions require (N,H,joint width)")
            actions = np.concatenate(parts, axis=-1)
        actions = np.asarray(actions)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.n_envs
            or actions.shape[1] < 1
            or actions.shape[2] != sum(self.action_dims)
        ):
            raise ValueError("actions must have shape (N, positive H, sum(action_dims))")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        results = [self._call(i, "step", actions[i]) for i in range(self.n_envs)]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (
            list(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def check_seeds(self, seeds: list[int]) -> list[bool]:
        """Validate seeds against the native layout inventory."""
        return self._call(0, "check_seeds", seeds)

    def close(self, clear_cache: bool = True) -> None:
        """Close every owned slot even if one slot reports a cleanup error."""
        if self._closed:
            return
        if self.n_envs == 1 and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Isaac teardown must run on the main thread")
        self._closed = True
        errors = []
        for env in self.envs:
            try:
                env.close(clear_cache)
            except Exception as error:  # noqa: BLE001 - release the remaining slots too
                errors.append(error)
        if errors:
            raise errors[0]
