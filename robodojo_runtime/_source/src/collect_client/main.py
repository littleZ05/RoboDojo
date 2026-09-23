"""Launch the RoboDojo automated data-collection client.

Example:
    python -u src/collect_client/main.py \
        --task_name general_pickup \
        --env_cfg_type arx_x5 \
        --num_envs 1 \
        --device_id 1 \
        --num_episodes 2 \
        --seed_start 0 \
        --save_dir /path/to/collect_result \
        --enable_cameras \
        --headless
"""

import argparse
from datetime import datetime
import importlib
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--env_cfg_type", type=str, required=True, help="config file name (e.g. arx_x5)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--device_id", type=int, required=True)
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--target_success", type=int, default=0, help="collect until this many successful episodes (0 = run num_episodes)")
parser.add_argument("--seeds", type=str, default="", help="comma-separated layout ids to collect (overrides seed queue order)")
parser.add_argument("--random", action="store_true", help="random scene/layout mode (bypasses Eval_Layout queue)")
parser.add_argument("--resume", action="store_true", help="skip episodes already recorded in the progress file")
parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--skip_plan", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR

task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")

if not os.environ.get("ROBODOJO_RUN_ID"):
    os.environ["ROBODOJO_RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omegaconf import OmegaConf

from collect.skill_manager import SkillError, SkillManager
from collect.task_scripts import get_task_script
from src.collect_client.collect_env import create_collect_env
from utils.cluttered_generator import UnStableError
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs

BENCHMARK_PATH = os.path.join(ROOT_DIR, "task", BENCHMARK)


def main():
    task_name = args_cli.task_name
    num_envs = args_cli.num_envs
    collect_cfg_name = args_cli.env_cfg_type
    collect_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, collect_cfg_name + ".yml"))
    collect_cfg["task_name"] = task_name
    collect_cfg["num_envs"] = num_envs
    collect_cfg["device_id"] = args_cli.device_id
    collect_cfg["num_episodes"] = args_cli.num_episodes
    collect_cfg["random_mode"] = args_cli.random
    collect_cfg["seed"] = args_cli.seed_start
    collect_cfg["episode_start"] = 0
    collect_cfg["save_dir"] = args_cli.save_dir or os.path.join(ROOT_DIR, "collect_result")
    if args_cli.random:
        # RoboTwin-style domain randomization: random background/table/ground.
        collect_cfg["domain_randomization"] = {
            "random_background": True,
            "random_ground": True,
            "random_table": True,
        }

    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", collect_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "scene", collect_cfg["config"]["scene"] + ".yml")
            ),
            "camera": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "camera", collect_cfg["config"]["camera"] + ".yml")
            ),
            "robot": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "robot", collect_cfg["config"]["robot"] + ".yml")
            ),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task_name)),
            "collect_cfg": collect_cfg,
            # pipeline_utils reads `eval_cfg` for domain-randomization flags.
            "eval_cfg": collect_cfg,
        }
    )
    capped = resolve_random_task_num_envs(task_name, num_envs, env_cfg.sim)
    if capped != num_envs:
        print(f"[collect] Random task {task_name}: num_envs capped {num_envs} -> {capped}")
    num_envs = capped
    collect_cfg["num_envs"] = num_envs
    OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
    OmegaConf.update(env_cfg, "collect_cfg.num_envs", num_envs, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task_name)
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        collect_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )
    env_cfg.sim.seed = [0 for _ in range(num_envs)]

    env = create_collect_env(env_cfg, simulation_app)
    progress_path = os.path.join(args_cli.save_dir or os.path.join(ROOT_DIR, "collect_result"), f"{task_name}_progress.json")
    progress = {"task": task_name, "random": args_cli.random, "episodes_done": 0, "seeds_done": []}
    if args_cli.resume and os.path.exists(progress_path):
        try:
            with open(progress_path, encoding="utf-8") as f:
                progress = json.load(f)
            print(f"[collect] resumed: episodes_done={progress.get('episodes_done', 0)} "
                  f"seeds_done={len(progress.get('seeds_done', []))}", flush=True)
        except Exception as e:
            print(f"[collect] failed to load progress {progress_path}: {e}; starting fresh", flush=True)
            progress = {"task": task_name, "random": args_cli.random, "episodes_done": 0, "seeds_done": []}
    if args_cli.seeds and not args_cli.random:
        # Override the seed queue with an explicit, ordered seed list.
        env.seed_manager.seed_list = [
            int(s) for s in args_cli.seeds.split(",") if s.strip()
        ]
        if args_cli.resume:
            done = set(int(s) for s in progress.get("seeds_done", []))
            env.seed_manager.seed_list = [s for s in env.seed_manager.seed_list if s not in done]
            env.seed_manager.ed_idx = len(env.seed_manager.seed_list)
        env.seed_manager.st_idx = 0
        env.seed_manager.idx = 0
        print(f"[collect] seed queue overridden with {len(env.seed_manager.seed_list)} seeds: {env.seed_manager.seed_list}", flush=True)
    script = get_task_script(task_name)

    success_nums = 0
    fail_nums = 0
    target = args_cli.target_success if args_cli.target_success > 0 else args_cli.num_episodes
    ep = progress.get("episodes_done", 0) if args_cli.random else 0
    while success_nums < target:
        if args_cli.random:
            batch = [ep]
            seed = ep
            print(f"\n[collect] random episode {ep} task={task_name} start", flush=True)
        else:
            batch = env.seed_manager.get_seeds(max_count=1)
            if not batch:
                print("[collect] no more layouts available", flush=True)
                break
            seed = batch[0]
            print(f"\n[collect] episode {ep} (seed={seed}) task={task_name} start", flush=True)
        try:
            env.reset(seed=seed)
        except UnStableError as e:
            print(f"[collect] episode {ep} seed {seed}: unstable scene, skip: {e}", flush=True)
            fail_nums += 1
            continue
        try:
            sm = SkillManager(env, env_idx=0)
            script(sm)
            env.record_terminal_obs()
            ok = env.is_success()
        except SkillError as e:
            print(f"[collect] episode {ep} seed {seed}: skill failed: {e}", flush=True)
            ok = False
        if not ok:
            print(f"[collect] episode {ep} seed {seed}: FAILED (reward not achieved)", flush=True)
            fail_nums += 1
            if args_cli.random:
                progress["episodes_done"] = ep + 1
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(progress, f, indent=2)
            continue
        path = env.finish_episode(episode_index=ep)
        success_nums += 1
        print(f"[collect] episode {ep} seed {seed}: SUCCESS -> {path}", flush=True)
        progress["episodes_done"] = ep + 1
        progress["seeds_done"].append(int(seed))
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
        ep += 1

    print(
        f"\n[collect] DONE task={task_name} target={target} "
        f"success={success_nums} fail={fail_nums}",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
