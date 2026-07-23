#!/usr/bin/env python3
"""Day3-3.6 PRACTICE: Multi-subtask source and mimic generation.

EEF-object 거리, gripper action, object 높이, object-bin 거리로 여러 subtask
boundary를 추정한 뒤 더 잘게 나눈 mimic source HDF5를 생성합니다.

수업 연결:
- 3.5와 같은 후처리 source 생성 방식이지만, subtask를 더 잘게 나눠 차이를 비교합니다.
- 학생이 작성하는 핵심은 더 잘게 나눈 subtask signal 생성입니다.
- 실행 확인은 3.4 DataGenerator rollout 스크립트를 내부에서 호출해 IsaacLab 시뮬레이터로 수행합니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 라이브러리 임포트 및 명령행 인자 설정
# ============================================================================
import argparse
import json
import sys
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

import h5py
import numpy as np

from day3_3_utils import copy_h5_attrs, ensure_h5_data_attrs, recreate_h5_dataset, run_mimic_generation, selected_demo_names

parser = argparse.ArgumentParser(description="Practice: multi-subtask source + mimic generation.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_ready_from_teleop_10.hdf5"))
parser.add_argument("--num_demos", type=int, default=0)
parser.add_argument("--generation_num_trials", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--enable_cameras", action="store_true", default=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--spawn_randomization", choices=["original", "wide"], default="original")
parser.add_argument("--visualize_subtasks", action="store_true", help="Show a color marker for each active subtask. Do not use the resulting images as training data.")
parser.add_argument("--source_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_source_multisubtask.hdf5"))
parser.add_argument("--output_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_generated_multisubtask.hdf5"))
args = parser.parse_args()

# ============================================================================
# 2. HDF5 trajectory / subtask signal 유틸
# ============================================================================
EEF_NAME = "franka"


# demo에서 T-bar/object의 xyz trajectory를 읽습니다.
def object_xyz(demo) -> np.ndarray:
    if "obs/datagen_info/object_pose/object_0" in demo:
        return demo["obs/datagen_info/object_pose/object_0"][:, :3, 3]
    return demo["states/rigid_object/object_0/root_pose"][:, :3]


# demo에서 bin의 xyz trajectory를 읽습니다.
def bin_xyz(demo) -> np.ndarray:
    if "obs/datagen_info/object_pose/bin" in demo:
        return demo["obs/datagen_info/object_pose/bin"][:, :3, 3]
    return demo["states/rigid_object/bin/root_pose"][:, :3]


# demo에서 target EEF xyz trajectory를 읽습니다.
def target_eef_xyz(demo) -> np.ndarray:
    if "obs/datagen_info/target_eef_pose/franka" in demo:
        return demo["obs/datagen_info/target_eef_pose/franka"][:, :3, 3]
    return demo["actions"][:, :3]


# demo에서 gripper action trajectory를 읽습니다.
def gripper_values(demo) -> np.ndarray:
    if "obs/datagen_info/gripper_action/franka" in demo:
        return demo["obs/datagen_info/gripper_action/franka"][:].reshape(-1)
    return demo["actions"][:, -1].reshape(-1)


# object가 충분히 들어올려진 첫 step을 lift boundary로 추정합니다.
def estimate_lift_step(demo, lift_height: float = 0.18) -> int | None:
    obj = object_xyz(demo)
    close_step = estimate_gripper_close_step(demo)
    start = 0 if close_step is None else close_step
    hits = np.flatnonzero(obj[start:, 2] > lift_height)
    return int(start + hits[0]) if hits.size > 0 else None


# EEF와 object가 가까워진 첫 step을 approach boundary로 추정합니다.
def estimate_approach_step(demo, threshold: float = 0.07) -> int | None:
    # [문제 6] EEF-object xy 거리식만 채우세요.
    dist = ____  # 빈칸 1: target EEF와 object의 xy distance
    hits = np.flatnonzero(dist < threshold)
    return int(hits[0]) if hits.size > 0 else int(np.argmin(dist))


# gripper command가 닫힘으로 바뀌는 첫 step을 close boundary로 추정합니다.
def estimate_gripper_close_step(demo) -> int | None:
    grip = gripper_values(demo)
    hits = np.flatnonzero(grip < 0.0)
    return int(hits[0]) if hits.size > 0 else None


# object와 bin이 가까워진 첫 step을 place 접근 boundary로 추정합니다.
def estimate_near_bin_step(demo, threshold: float = 0.13) -> int | None:
    obj = object_xyz(demo)
    goal = bin_xyz(demo)
    dist = np.linalg.norm(obj[:, :2] - goal[:, :2], axis=1)
    lift_step = estimate_lift_step(demo)
    start = 0 if lift_step is None else lift_step
    hits = np.flatnonzero(dist[start:] < threshold)
    return int(start + hits[0]) if hits.size > 0 else None


# 여러 subtask boundary가 시간 순서대로 증가하도록 보정합니다.
def monotonic_transition_steps(length: int, raw_steps: list[int | None], min_gap: int = 2) -> list[int]:
    # 경계 후보 생성은 제공하며, 허용 범위로 제한하는 한 줄만 채우세요.
    if length <= len(raw_steps) * min_gap + 1:
        min_gap = 1
    out = []
    prev = 0
    n = len(raw_steps)
    for i, raw in enumerate(raw_steps):
        fallback = int(round((i + 1) * length / (n + 1)))
        step = fallback if raw is None else int(raw)
        min_allowed = prev + min_gap
        max_allowed = length - 1 - (n - i - 1) * min_gap
        if max_allowed < min_allowed:
            max_allowed = min_allowed
        step = ____  # 빈칸 2: step을 min_allowed~max_allowed로 제한
        out.append(step)
        prev = step
    return out


# 지정 step부터 1이 되는 0/1 subtask termination signal을 만듭니다.
def step_signal(length: int, step: int) -> np.ndarray:
    signal = np.zeros((length, 1), dtype=np.float32)
    signal[step:, 0] = 1.0
    return signal


# HDF5 demo가 mimic source 생성에 필요한 datagen_info를 갖췄는지 확인합니다.
def assert_mimic_ready(demo, demo_name: str) -> None:
    required = [
        "actions",
        "obs/datagen_info/object_pose/object_0",
        "obs/datagen_info/object_pose/bin",
        "obs/datagen_info/target_eef_pose/franka",
        "obs/datagen_info/gripper_action/franka",
    ]
    missing = [path for path in required if path not in demo]
    if missing:
        raise RuntimeError(f"{demo_name} is not mimic-ready. Missing: {missing}")


# ============================================================================
# 3. Source HDF5 생성
# ============================================================================
# 여러 boundary로 multi-subtask source HDF5를 생성합니다.
def create_multisubtask_source(input_file: str, output_file: str, num_demos: int = 0) -> None:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with h5py.File(input_file, "r") as src, h5py.File(output_file, "w") as dst:
        copy_h5_attrs(src, dst)
        data_dst = dst.create_group("data")
        ensure_h5_data_attrs(data_dst, src["data"])
        names = selected_demo_names(src["data"], num_demos)
        for name in names:
            src_demo = src[f"data/{name}"]
            dst_demo = data_dst.create_group(name)
            copy_h5_attrs(src_demo, dst_demo)
            src_demo.copy("actions", dst_demo)
            obs_group = dst_demo.create_group("obs")
            src_demo["obs"].copy("datagen_info", obs_group)
        data_dst.attrs["total"] = sum(int(demo["actions"].shape[0]) for demo in data_dst.values())

    with h5py.File(output_file, "a") as f:
        f.attrs["subtask_mode"] = "multisubtask"
        for name in selected_demo_names(f["data"], num_demos):
            demo = f[f"data/{name}"]
            assert_mimic_ready(demo, name)
            length = int(demo["actions"].shape[0])
            raw_approach = estimate_approach_step(demo)
            raw_close = estimate_gripper_close_step(demo)
            raw_lifted = estimate_lift_step(demo)
            raw_near_bin = estimate_near_bin_step(demo)
            approach, close, lifted, near_bin = monotonic_transition_steps(
                length, [raw_approach, raw_close, raw_lifted, raw_near_bin]
            )
            signals = demo.require_group("obs/datagen_info/subtask_term_signals")
            recreate_h5_dataset(signals, "approach_done", step_signal(length, approach))
            recreate_h5_dataset(signals, "gripper_closed", step_signal(length, close))
            recreate_h5_dataset(signals, "object_lifted_from_height", step_signal(length, lifted))
            recreate_h5_dataset(signals, "object_near_bin", step_signal(length, near_bin))
            summary = {
                "mode": "multisubtask",
                "approach_done": approach,
                "gripper_closed": close,
                "object_lifted": lifted,
                "object_near_bin": near_bin,
                "raw_approach_done": raw_approach,
                "raw_gripper_closed": raw_close,
                "raw_object_lifted": raw_lifted,
                "raw_object_near_bin": raw_near_bin,
            }
            demo.attrs["subtask_summary"] = json.dumps(summary)
            print(f"[MULTISUBTASK] {name}: {summary}", flush=True)


# ============================================================================
# 4. Mimic generation 실행
# ============================================================================
# multi-subtask source 생성 실습 후 mimic generation을 실행합니다.
def main():
    create_multisubtask_source(args.input_file, args.source_file, args.num_demos)
    # 여기서 3.4 스크립트를 내부 호출하므로 IsaacLab AppLauncher가 실행되고 시뮬레이터에서 결과를 확인합니다.
    run_mimic_generation(
        input_file=args.input_file,
        source_file=args.source_file,
        output_file=args.output_file,
        subtask_mode="multisubtask",
        generation_num_trials=args.generation_num_trials,
        num_envs=args.num_envs,
        enable_cameras=args.enable_cameras,
        headless=args.headless,
        spawn_randomization=args.spawn_randomization,
        visualize_subtasks=args.visualize_subtasks,
    )
    print(f"[DONE] multi-subtask generation: {args.output_file}")


if __name__ == "__main__":
    main()
