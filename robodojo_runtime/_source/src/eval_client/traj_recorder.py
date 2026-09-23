"""Record evaluation rollouts as RoboDojo-format HDF5 trajectories.

Enabled with ``ROBODOJO_SAVE_TRAJ=1``. For each evaluated episode the recorder
writes an ``.hdf5`` file under ``<save_dir>/traj/episode_<index>.hdf5`` whose
schema mirrors the released RoboDojo HDF5 demos (``state/*``, ``action/*``,
``vision/<cam>/colors`` JPEG-encoded fixed-width byte strings, ``instruction``,
``additional_info/frequency``, ``data_format_version``), so the outputs can be
consumed by the existing conversion tooling (e.g. ``process_data.py``).

Alignment convention: frame ``t`` records the observation *before* the policy
action ``t`` is applied; the terminal observation fetched by
``is_episode_end(..., last_frame=True)`` is dropped to keep ``state`` and
``action`` aligned (``len(state) == len(action)``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import h5py
import cv2


# HDF5 dataset suffix -> obs/action dict key.
# Joints use "<arm>_arm_joint_state" (arm_name is "left_arm"/"right_arm"),
# while ee keys use "<arm>_ee_joint_state" / "<arm>_ee_pose" / "<arm>_delta_ee_pose".
def _arm_key_map(arm: str) -> dict[str, str]:
    return {
        f"{arm}_arm_joint_states": f"{arm}_arm_joint_state",
        f"{arm}_ee_joint_states": f"{arm}_ee_joint_state",
        f"{arm}_ee_poses": f"{arm}_ee_pose",
        f"{arm}_delta_ee_poses": f"{arm}_delta_ee_pose",
    }


def _encode_jpeg_padded(frames: list[np.ndarray]) -> np.ndarray:
    """Encode RGB frames as JPEG bytes padded to a common fixed width."""
    encoded = []
    max_len = 0
    for frame in frames:
        success, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame[..., :3]))
        if not success:
            raise ValueError("cv2.imencode failed for a recorded frame")
        data = buf.tobytes()
        encoded.append(data)
        max_len = max(max_len, len(data))
    padded = [data.ljust(max_len, b"\0") for data in encoded]
    return np.array(padded, dtype=f"S{max_len}")


def _as_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in "SU":
        return value.item() if value.ndim == 0 else bytes(value)
    return str(value).encode("utf-8")


class TrajRecorder:
    """Per-env buffers + HDF5 writer for evaluation rollouts."""

    def __init__(self, active: bool = False):
        self.active = active
        self.reset_all()

    def reset_all(self) -> None:
        self._visions: dict[int, list[dict[str, np.ndarray]]] = {}
        self._states: dict[int, list[dict[str, np.ndarray]]] = {}
        self._actions: dict[int, list[dict[str, np.ndarray]]] = {}
        self._meta: dict[int, dict[str, Any]] = {}

    def reset_env(self, env_idx: int) -> None:
        self._visions.pop(env_idx, None)
        self._states.pop(env_idx, None)
        self._actions.pop(env_idx, None)
        self._meta.pop(env_idx, None)

    def record_obs(self, env_idx: int, obs: dict[str, Any]) -> None:
        if not self.active:
            return
        vision: dict[str, np.ndarray] = {}
        for cam, cam_data in (obs.get("vision") or {}).items():
            if isinstance(cam_data, dict) and cam_data.get("color") is not None:
                color = np.asarray(cam_data["color"])
                if color.ndim == 3 and color.shape[2] in (3, 4):
                    vision[cam] = np.ascontiguousarray(color[..., :3])
        state: dict[str, np.ndarray] = {}
        for key, value in (obs.get("state") or {}).items():
            arr = np.asarray(value)
            if arr.dtype == object or arr.ndim != 1:
                continue
            state[key] = arr.astype(np.float64)
        self._visions.setdefault(env_idx, []).append(vision)
        self._states.setdefault(env_idx, []).append(state)
        self._meta[env_idx] = {
            "instruction": obs.get("instruction"),
            "frequency": int((obs.get("additional_info") or {}).get("frequency", 25)),
            "data_format_version": obs.get("data_format_version", "v1.0"),
        }

    def record_action(self, env_idx: int, action: dict[str, Any]) -> None:
        if not self.active:
            return
        act: dict[str, np.ndarray] = {}
        for key, value in action.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.dtype == object:
                    continue
                act[key] = arr.astype(np.float64)
        self._actions.setdefault(env_idx, []).append(act)

    def write(self, env_idx: int, save_dir: str, episode_index: int, action_from_state: bool = False) -> None:
        if not self.active:
            return
        states = self._states.get(env_idx) or []
        actions = self._actions.get(env_idx) or []
        visions = self._visions.get(env_idx) or []
        if not states:
            self.reset_env(env_idx)
            return
        # Replay mode (official RoboDojo convention): only observations are
        # recorded (one per control frame plus the terminal frame); the action
        # for frame t is the *achieved* state at t+1, so we shift the state
        # buffer by one and write state=obs[:-1], action=obs[1:].
        if action_from_state:
            if len(states) < 2:
                self.reset_env(env_idx)
                return
            actions = [{"state": s} for s in states[1:]]
            states = states[:-1]
            visions = visions[:-1]
        def _action_value(a: dict, key: str):
            return a.get("state", a).get(key) if isinstance(a.get("state", a), dict) else a.get(key)
        # Drop the terminal observation so state/action lengths align.
        if len(actions) == len(states) - 1:
            states = states[: len(actions)]
            visions = visions[: len(actions)]
        n = min(len(states), len(actions))
        if n == 0:
            self.reset_env(env_idx)
            return
        states, actions, visions = states[:n], actions[:n], visions[:n]
        meta = self._meta.get(env_idx, {})

        path = os.path.join(save_dir, "traj", f"episode_{episode_index:07d}.hdf5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            for arm in ("left", "right"):
                for hdf_suffix, key in _arm_key_map(arm).items():
                    if all(key in s for s in states):
                        f.create_dataset(
                            f"state/{hdf_suffix}",
                            data=np.stack([s[key] for s in states]),
                        )
                for hdf_suffix, key in _arm_key_map(arm).items():
                    if all(_action_value(a, key) is not None for a in actions):
                        f.create_dataset(
                            f"action/{hdf_suffix}",
                            data=np.stack([_action_value(a, key) for a in actions]),
                        )
            for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                frames = [v.get(cam) for v in visions]
                if any(x is not None for x in frames):
                    f.create_dataset(
                        f"vision/{cam}/colors",
                        data=_encode_jpeg_padded([x for x in frames if x is not None]),
                    )
            f.create_dataset(
                "instruction",
                data=np.bytes_(_as_bytes(meta.get("instruction"))),
            )
            f.create_dataset(
                "additional_info/frequency",
                data=int(meta.get("frequency", 25)),
            )
            f.create_dataset(
                "data_format_version",
                data=np.bytes_(str(meta.get("data_format_version", "v1.0"))),
            )
        self.reset_env(env_idx)
        return path
