"""CollectEnv: a leaner sibling of the eval env for automated data collection.

Drives a RoboDojo task environment without any policy server: skills produce
per-25Hz-frame control targets, the env interpolates each frame over the
configured number of physics substeps (8:2 ramp as in eval), and observations
are recorded per frame. Episodes are written as official-format HDF5 with the
official action convention ``action[t] == state[t+1]``.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np

from env.global_configs import BENCHMARK, ROOT_DIR
from env.observation_manager.obs_manager import ObsManager
from env.seed_manager.seed_manager import SeedManager
from src.eval_client.traj_recorder import TrajRecorder
from utils.cluttered_generator import UnStableError


def _expand_control_info(env, control_info: dict, env_idx: int):
    """Expand one 25Hz control target into per-substep control dicts using the
    same 8:2 interpolation convention as the eval client."""
    interpolation_nums = int(env.obs_manager.collect_interval)
    if interpolation_nums <= 0:
        return [deepcopy(control_info)]
    control_info_list = [deepcopy(control_info) for _ in range(interpolation_nums)]
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        key_name = env.robot_manager.process_name(robot.arm_name)
        if key_name in control_info.keys():
            position = control_info[key_name]["position"]
            current_position = env.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
            if current_position is not None:
                interp_count = int(np.floor(interpolation_nums * 0.8))
                current_arr = np.array(current_position)
                target_arr = np.array(position)
                for i in range(interp_count):
                    alpha = (i + 1) / (interp_count + 1)
                    interp_pos = (1 - alpha) * current_arr + alpha * target_arr
                    control_info_list[i][key_name]["position"] = interp_pos.tolist()
                for i in range(interp_count, interpolation_nums):
                    control_info_list[i][key_name]["position"] = target_arr.tolist()
        key_name = env.robot_manager.process_name(robot.gripper_name)
        if key_name in control_info.keys() and robot.ee_type == "gripper":
            position = control_info[key_name]["position"][0]
            current_position = env.robot_manager.get_end_effector_real_val(
                robot, env_idx_list=[env_idx]
            )[env_idx][0]
            if current_position is not None:
                interp_count = int(np.floor(interpolation_nums * 0.8))
                scale = robot.gripper_scale
                for i in range(interp_count):
                    alpha = (i + 1) / (interp_count + 1)
                    interp_pos = (1 - alpha) * current_position + alpha * position
                    interp_pos = np.clip(interp_pos, scale[0], scale[1])
                    vals = [
                        interp_pos,
                        interp_pos * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                    control_info_list[i][key_name]["position"] = vals
                for i in range(interp_count, interpolation_nums):
                    vals = [
                        position,
                        position * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                    control_info_list[i][key_name]["position"] = vals
    return control_info_list


def create_collect_env(config, app, **kwargs):
    task_name = config.collect_cfg.get("task_name", None)
    if task_name is None:
        raise ValueError("task_name must be specified in collect_cfg!")

    import importlib

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_name, task_class = task_registry.load_task_class(task_name)

    class CollectEnv(task_class):
        def __init__(self, config, app, **kwargs):
            super().__init__(config, app, **kwargs)
            self.collect_cfg = config.collect_cfg
            self.task_name = self.collect_cfg.get("task_name", None)
            self.save_dir = self.collect_cfg.get("save_dir", os.path.join(ROOT_DIR, "collect_result"))
            self.num_episodes = int(self.collect_cfg.get("num_episodes", 1))
            self.episode_index = int(self.collect_cfg.get("episode_start", 0))
            self.obs_config = deepcopy(self.collect_cfg.get("observation", {}))
            self.obs_config["save_dir"] = self.save_dir
            self.description_cfg = self.collect_cfg.get("description", dict())
            self.obs_manager = ObsManager(
                obs_config=self.obs_config,
                num_envs=self.num_envs,
                dt=self.dt,
                task_name=self.task_name,
                description_cfg=self.description_cfg,
                seeds_per_env=self.env_seed_list,
            )
            self.scene_manager.layout_manager.replay = True
            self.seed_manager = SeedManager(config.collect_cfg)
            self.seed_manager.init_eval()
            self.random_mode = bool(self.collect_cfg.get("random_mode", False))
            self.traj_recorder = TrajRecorder(active=True)
            self.current_env_seed_map: dict[int, int] = {}
            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.take_action_cnt = [0] * self.num_envs
            run_id = os.environ.get("ROBODOJO_RUN_ID")
            if not run_id:
                run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                os.environ["ROBODOJO_RUN_ID"] = run_id
            self.run_id = run_id

        # ---------------- lifecycle ----------------
        def reset(self, seed=None, options=None):
            self.traj_recorder.reset_all()
            if self.random_mode:
                # Fresh random layout + scene per episode: bypass the saved
                # eval-layout queue and let the layout manager sample object
                # poses from the task config using a fresh per-env seed.
                fresh = [random.randrange(0, 1_000_000_000) for _ in range(self.num_envs)]
                self.env_seeds = fresh
                self.current_env_seed_map = {i: s for i, s in enumerate(fresh)}
                for idx in range(self.num_envs):
                    self.scene_manager.layout_manager.saved_layouts[idx] = None
                super().reset(seed=self.env_seeds, options=options)
                self.obs_manager.reset()
                self.setup_scene()
                self.robot_manager.set_origin_endpose()
                self.robot_manager.set_robot_init_state()
                self.reward_manager.init_state()
                self.run_reward()
                return
            if not isinstance(seed, (list, tuple)):
                seed = [seed]
            seed = list(seed)
            if len(seed) < self.num_envs:
                seed = seed + [None] * (self.num_envs - len(seed))
            real_indices = [i for i, s in enumerate(seed) if s is not None]
            safe_seed = seed[real_indices[0]] if real_indices else 0
            self.env_seeds = [s if s is not None else safe_seed for s in seed]
            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.take_action_cnt = [0] * self.num_envs
            self.current_env_seed_map = {}
            for idx in range(self.num_envs):
                self.scene_manager.layout_manager.set_saved_layout(
                    idx, self.seed_manager.get_seed_scene_info(self.env_seeds[idx])
                )
                if seed[idx] is None:
                    self.success[idx] = False
                    self.end_flag[idx] = True
                else:
                    self.current_env_seed_map[idx] = seed[idx]
            super().reset(seed=self.env_seeds, options=options)
            self.obs_manager.reset()
            self.setup_scene()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            # Register the task's reward checks before any action so that
            # reward_manager.step() (called inside apply_target) evaluates and
            # pops them during the episode -- matching the eval flow. Without
            # this, get_reward() sees a non-empty check list and reports 0.
            self.run_reward()

        def setup_scene(self):
            self.scene_manager.apply_saved_poses(env_idx_list=list(range(self.num_envs)))
            success, unstable_envs = self.scene_manager.layout_manager.check_layout_stability(self)
            unstable_envs = [idx for idx in set(unstable_envs) if idx < self.num_envs]
            for idx in unstable_envs:
                self.success[idx] = False
                self.end_flag[idx] = True
            if not success or all(self.end_flag):
                raise UnStableError("All scene Unstable Error!")
            for _ in range(10):
                self.render()
            setup_steps = int(os.environ.get("ROBODOJO_SETUP_STEPS", "200"))
            for idx in range(setup_steps):
                self.sim_step()
                if idx % 5 == 0:
                    self.render()
                    self.obs_manager.get_obs()

        def close(self):
            self.obs_manager.reset()
            super().close()

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        # ---------------- stepping ----------------
        def step(self, env_idx_list, decimation=1):
            meta_control_list = self.robot_manager.control_manager.pop(env_idx_list)
            for _ in range(decimation):
                super().step(meta_control_list=meta_control_list)
                self.sim_step(render=False)

        def get_obs(self, env_idx=0):
            self.render()
            data = self.obs_manager.get_obs(env_idx_list=[env_idx])
            return deepcopy(data[env_idx])

        def apply_target(self, control_info: dict, env_idx=0):
            control_seq = _expand_control_info(self, control_info, env_idx)
            self.robot_manager.control_manager.push([env_idx], [control_seq])
            while self.robot_manager.control_manager.get_empty([env_idx]) != [env_idx]:
                self.step([env_idx])
            self.reward_manager.step([env_idx])

        # ---------------- collection ----------------
        def run_control_seq(self, seq, frame_callback=None):
            """Execute a per-frame control sequence, recording an observation
            before every frame plus one terminal observation at the end. An
            optional ``frame_callback(env, env_idx)`` is invoked after each
            frame's observation is recorded (and once after the terminal obs),
            useful for A/V validation logging."""
            env_idx = 0
            for control_info in seq:
                self.traj_recorder.record_obs(env_idx, self.get_obs())
                self.apply_target(control_info, env_idx)
                if frame_callback is not None:
                    frame_callback(self, env_idx)
            self.traj_recorder.record_obs(env_idx, self.get_obs())
            if frame_callback is not None:
                frame_callback(self, env_idx)

        def record_terminal_obs(self, env_idx=0):
            """Record the terminal observation after the last control frame so
            the HDF5 can be written with state=obs[:-1], action=obs[1:]."""
            self.traj_recorder.record_obs(env_idx, self.get_obs())

        def is_success(self, env_idx=0):
            reward_list = self.reward_manager.get_reward(final_check=True)
            return bool(reward_list[env_idx] > 1 - 1e-3)

        def finish_episode(self, episode_index=None, env_idx=0):
            """Write the recorded rollout as official-format HDF5 with
            action[t] == state[t+1] (achieved-state convention)."""
            if episode_index is None:
                episode_index = self.episode_index
            self.traj_recorder.write(
                env_idx,
                os.path.join(self.save_dir, self.task_name),
                episode_index,
                action_from_state=True,
            )
            self.episode_index = episode_index + 1
            return os.path.join(self.save_dir, self.task_name, "traj", f"episode_{episode_index:07d}.hdf5")

    return CollectEnv(config, app, **kwargs)
