from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


DAY3_ROOT = Path(__file__).resolve().parent


# 환경변수와 기본 설치 위치에서 IsaacLab 경로를 찾습니다.
def isaaclab_path() -> str:
    env_path = os.environ.get("ISAACLAB_PATH")
    if env_path:
        return env_path
    if Path("/workspace/IsaacLab/isaaclab.sh").exists():
        return "/workspace/IsaacLab"
    return str(Path.home() / "IsaacLab")


# 현재 스크립트 밖에서 3.4 DataGenerator rollout 스크립트를 실행합니다.
def run_mimic_generation(
    input_file: str,
    source_file: str,
    output_file: str,
    subtask_mode: str,
    generation_num_trials: int,
    num_envs: int = 1,
    enable_cameras: bool = True,
    headless: bool = False,
    spawn_randomization: str = "original",
    visualize_subtasks: bool = False,
) -> None:
    cmd = [
        f"{isaaclab_path()}/isaaclab.sh",
        "-p",
        "day3/day3_3.4_mimic_datagenerator_rollout_answer.py",
        "--input_file",
        input_file,
        "--annotated_file",
        source_file,
        "--output_file",
        output_file,
        "--subtask_mode",
        subtask_mode,
        "--generation_num_trials",
        str(generation_num_trials),
        "--num_envs",
        str(num_envs),
        "--spawn_randomization",
        spawn_randomization,
    ]
    if enable_cameras:
        cmd.append("--enable_cameras")
    if headless:
        cmd.append("--headless")
    if visualize_subtasks:
        cmd.append("--visualize_subtasks")
    print(f"[GENERATE] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["TERM"] = "xterm"
    subprocess.run(cmd, cwd=DAY3_ROOT.parent, check=True, env=env)


# nested torch tensor tree를 CPU tensor tree로 복사합니다.
def to_cpu_tree(value):
    if isinstance(value, dict):
        return {key: to_cpu_tree(child) for key, child in value.items()}
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


# demo_숫자 이름을 숫자 순서대로 정렬하기 위한 key를 만듭니다.
def demo_sort_key(name: str):
    if name.startswith("demo_") and name.split("_")[-1].isdigit():
        return int(name.split("_")[-1])
    return name


# HDF5 data group에서 처리할 demo 이름 목록을 선택합니다.
def selected_demo_names(data_group, num_demos: int) -> list[str]:
    names = sorted(list(data_group.keys()), key=demo_sort_key)
    return names[:num_demos] if num_demos > 0 else names


# HDF5 attribute들을 다른 group/file로 복사합니다.
def copy_h5_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


# IsaacLab dataset data group에 필요한 기본 attribute를 채웁니다.
def ensure_h5_data_attrs(data_group, source_data_group=None, env_name: str = "") -> None:
    if source_data_group is not None:
        copy_h5_attrs(source_data_group, data_group)
    if "env_args" not in data_group.attrs:
        data_group.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})
    if "total" not in data_group.attrs:
        data_group.attrs["total"] = sum(int(demo["actions"].shape[0]) for demo in data_group.values() if "actions" in demo)


# HDF5 group 또는 dataset 하나를 같은 이름으로 복사합니다.
def copy_h5_item(src_group, dst_group, key: str) -> None:
    if key in dst_group:
        del dst_group[key]
    src_group.copy(key, dst_group, name=key)


# HDF5 group 안에 같은 이름의 item이 있으면 삭제합니다.
def delete_if_exists(group, key: str) -> None:
    if key in group:
        del group[key]


# 기존 dataset을 지우고 같은 이름으로 새 dataset을 만듭니다.
def recreate_h5_dataset(group, key: str, data) -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=data)


# 3x3 rotation matrix를 wxyz quaternion으로 변환합니다.
def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    tr = float(np.trace(rot))
    if tr > 0.0:
        scale = np.sqrt(tr + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (rot[2, 1] - rot[1, 2]) / scale,
            (rot[0, 2] - rot[2, 0]) / scale,
            (rot[1, 0] - rot[0, 1]) / scale,
        ], dtype=np.float32)
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array([(rot[2, 1] - rot[1, 2]) / scale, 0.25 * scale, (rot[0, 1] + rot[1, 0]) / scale, (rot[0, 2] + rot[2, 0]) / scale], dtype=np.float32)
        elif idx == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array([(rot[0, 2] - rot[2, 0]) / scale, (rot[0, 1] + rot[1, 0]) / scale, 0.25 * scale, (rot[1, 2] + rot[2, 1]) / scale], dtype=np.float32)
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array([(rot[1, 0] - rot[0, 1]) / scale, (rot[0, 2] + rot[2, 0]) / scale, (rot[1, 2] + rot[2, 1]) / scale, 0.25 * scale], dtype=np.float32)
    return quat / np.linalg.norm(quat).clip(1.0e-6)
