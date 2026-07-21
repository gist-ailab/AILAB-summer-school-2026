#!/usr/bin/env python3
"""Day3-3.5 PRACTICE: 2-subtask source and mimic generation.

수집 시 저장한 signal을 쓰지 않고 object 높이 변화로 lift boundary를 추정해
2-subtask source HDF5를 만든 뒤, 3.4의 DataGenerator rollout을 재사용합니다.

수업 연결:
- 학생이 작성하는 핵심은 source HDF5의 subtask signal 생성입니다.
- 실행 확인은 3.4 DataGenerator rollout 스크립트를 내부에서 호출해 IsaacLab 시뮬레이터로 수행합니다.
- 3.2.2의 수집 signal을 쓰는 방식과 비교하기 위한 2-subtask 후처리 버전입니다.
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

parser = argparse.ArgumentParser(description="Practice: 2-subtask source + mimic generation.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_ready_from_teleop_10.hdf5"))
parser.add_argument("--num_demos", type=int, default=0)
parser.add_argument("--generation_num_trials", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--enable_cameras", action="store_true", default=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--spawn_randomization", choices=["original", "wide"], default="original")
parser.add_argument("--visualize_subtasks", action="store_true", help="Show a color marker for each active subtask. Do not use the resulting images as training data.")
parser.add_argument("--source_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_source_2subtask.hdf5"))
parser.add_argument("--output_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_generated_2subtask.hdf5"))
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


# 0에서 1로 처음 바뀌는 subtask signal index를 찾습니다.
def first_transition(signal: np.ndarray) -> int | None:
    active = (np.asarray(signal).reshape(-1) > 0.5).astype(np.int32)
    diffs = active[1:] - active[:-1]
    hits = np.flatnonzero(diffs > 0)
    if hits.size > 0:
        return int(hits[0] + 1)
    hits = np.flatnonzero(active)
    return int(hits[0]) if hits.size > 0 else None


# demo에서 gripper action trajectory를 읽습니다.
def gripper_values(demo) -> np.ndarray:
    if "obs/datagen_info/gripper_action/franka" in demo:
        return demo["obs/datagen_info/gripper_action/franka"][:].reshape(-1)
    return demo["actions"][:, -1].reshape(-1)


# gripper command가 닫힘으로 바뀌는 첫 step을 찾습니다.
def estimate_gripper_close_step(demo) -> int | None:
    grip = gripper_values(demo)
    hits = np.flatnonzero(grip < 0.0)
    return int(hits[0]) if hits.size > 0 else None


# gripper close 이후 object가 충분히 들어올려진 첫 step을 lift boundary로 추정합니다.
def estimate_lift_step(demo, lift_height: float = 0.18) -> int | None:
    # [문제 5-1] object z trajectory를 보고 물체가 들어올려진 첫 step을 추정하세요.
    # 힌트: gripper close 이후, base-frame object z가 lift_height를 넘는 첫 index를 찾습니다.
    raise NotImplementedError("문제 5-1: object height 기반 lifted boundary를 추정하세요.")


# subtask 경계 index가 episode 범위 안에 들어오도록 보정합니다.
def valid_transition_step(length: int, step: int | None) -> int:
    if length < 2:
        raise ValueError(f"Need at least 2 steps, got {length}")
    if step is None:
        step = length // 2
    return int(np.clip(step, 1, length - 1))


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
# object height으로 2-subtask source HDF5를 생성합니다.
def create_2subtask_source(input_file: str, output_file: str, num_demos: int = 0) -> None:
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
        f.attrs["subtask_mode"] = "2subtask"
        for name in selected_demo_names(f["data"], num_demos):
            demo = f[f"data/{name}"]
            assert_mimic_ready(demo, name)
            length = int(demo["actions"].shape[0])
            # [문제 5-2] 추정한 lifted step을 0->1 signal로 저장하세요.
            # 힌트: estimate_lift_step -> valid_transition_step -> step_signal -> recreate_h5_dataset.
            raise NotImplementedError("문제 5-2: object_lifted_from_height signal dataset을 생성하세요.")
            ref = first_transition(signals["object_lifted"][:]) if "object_lifted" in signals else None
            demo.attrs["subtask_summary"] = json.dumps({
                "mode": "2subtask",
                "object_lifted_from_height": step,
                "raw_object_lifted_from_height": raw_step,
                "collected_object_lifted": ref,
            })
            print(f"[2SUBTASK] {name}: object_lifted={step}, raw={raw_step}, collected_lifted={ref}", flush=True)


# ============================================================================
# 4. Mimic generation 실행
# ============================================================================
# 2-subtask source 생성 실습 후 mimic generation을 실행합니다.
def main():
    create_2subtask_source(args.input_file, args.source_file, args.num_demos)
    # 여기서 3.4 스크립트를 내부 호출하므로 IsaacLab AppLauncher가 실행되고 시뮬레이터에서 결과를 확인합니다.
    run_mimic_generation(
        input_file=args.input_file,
        source_file=args.source_file,
        output_file=args.output_file,
        subtask_mode="2subtask",
        generation_num_trials=args.generation_num_trials,
        num_envs=args.num_envs,
        enable_cameras=args.enable_cameras,
        headless=args.headless,
        spawn_randomization=args.spawn_randomization,
        visualize_subtasks=args.visualize_subtasks,
    )
    print(f"[DONE] 2-subtask generation: {args.output_file}")


if __name__ == "__main__":
    main()
