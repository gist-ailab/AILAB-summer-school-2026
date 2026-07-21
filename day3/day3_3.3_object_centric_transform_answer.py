#!/usr/bin/env python3
"""Day3-3.3 ANSWER: Object-centric approach trajectory transform from HDF5.

2.2에서 만든 mimic-ready HDF5의 첫 demo에서 T-bar 접근 구간 action을 가져오고,
T-bar 위치를 여러 번 바꿔도 같은 접근 구간이 object-centric 변환으로 따라가는지 확인합니다.

수업 연결:
- 3.2.2에서 만든 datagen_info 중 object/target_eef pose가 왜 필요한지 시각적으로 확인하는 단계입니다.
- 뒤의 3.4 DataGenerator가 내부적으로 수행하는 object-centric pose 변환을 작은 예제로 먼저 봅니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
import random
import sys
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay an HDF5 approach segment toward randomly moved T-bars.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_ready_from_teleop_10.hdf5"))
parser.add_argument("--num_replays", type=int, default=5)
parser.add_argument("--approach_extra_steps", type=int, default=0, help="Extra steps after the first gripper-close command.")
parser.add_argument("--object_x_offset_range", type=float, nargs=2, default=(-0.18, 0.18))
parser.add_argument("--object_y_offset_range", type=float, nargs=2, default=(-0.18, 0.18))
parser.add_argument("--object_yaw_range_deg", type=float, nargs=2, default=(-45.0, 135.0))
parser.add_argument("--hold_steps", type=int, default=45)
# Day2 3/4교시와 같은 AppLauncher 인자 추가 패턴입니다.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 / 데이터 처리 라이브러리 임포트
# ============================================================================
import carb  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from task.lift.config.ik_abs_env_cfg_3_1_answer import FrankaTBarPickPlaceEnvCfg  # noqa: E402
from day3_3_utils import demo_sort_key, matrix_to_quat_wxyz  # noqa: E402

# 카메라 렌더링 활성화 (--enable_cameras flag 대체)
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# ============================================================================
# 3. 핵심 함수 및 문제 코드
# ============================================================================
EEF_NAME = "franka"
TABLE_TOP_Z = 0.5


# 4x4 pose matrix의 inverse transform을 계산합니다.
def pose_inv(pose: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


# source EEF 궤적을 object 기준으로 옮긴 뒤 새 object pose 기준 world 궤적으로 변환합니다.
def transform_source_data_segment_using_object_pose(
    obj_pose: np.ndarray,
    src_eef_poses: np.ndarray,
    src_obj_pose: np.ndarray,
) -> np.ndarray:
    src_eef_poses_rel_obj = pose_inv(src_obj_pose) @ src_eef_poses
    return obj_pose @ src_eef_poses_rel_obj


# z축 yaw 회전을 4x4 pose matrix로 만듭니다.
def yaw_matrix(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return out


# HDF5 data group의 첫 demo 이름을 가져옵니다.
def first_demo_name(data_group) -> str:
    names = sorted(list(data_group.keys()), key=demo_sort_key)
    if not names:
        raise RuntimeError("No demo exists in input HDF5.")
    return names[0]


# HDF5 states tree에서 특정 step의 simulator state를 env.reset_to() 형식으로 읽습니다.
def load_state_step(group: h5py.Group, step: int):
    state = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            state[key] = torch.from_numpy(item[step]).unsqueeze(0)
        else:
            state[key] = load_state_step(item, step)
    return state


# target EEF pose가 실제로 움직이기 시작하는 지점을 찾습니다.
def first_motion_step(target_poses: np.ndarray, threshold: float = 1.0e-4) -> int:
    xyz = target_poses[:, :3, 3]
    delta = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    hits = np.flatnonzero(delta > threshold)
    if hits.size == 0:
        return 0
    return max(0, int(hits[0]) - 5)


# HDF5 첫 demo에서 object-centric 접근 segment를 잘라옵니다.
def load_first_approach_segment(hdf5_file: str) -> dict[str, np.ndarray | dict]:
    with h5py.File(hdf5_file, "r") as f:
        demo_name = first_demo_name(f["data"])
        demo = f[f"data/{demo_name}"]
        target_poses = demo[f"obs/datagen_info/target_eef_pose/{EEF_NAME}"][:].astype(np.float32)
        object_poses = demo["obs/datagen_info/object_pose/object_0"][:].astype(np.float32)
        gripper = demo[f"obs/datagen_info/gripper_action/{EEF_NAME}"][:].reshape(-1).astype(np.float32)
    start = first_motion_step(target_poses)
    close_hits = np.flatnonzero(gripper < 0.0)
    close_step = int(close_hits[0]) if close_hits.size else len(target_poses)
    end = int(np.clip(close_step + args_cli.approach_extra_steps, start + 2, len(target_poses)))
    dist = np.linalg.norm(target_poses[:, :2, 3] - object_poses[:, :2, 3], axis=1)
    with h5py.File(hdf5_file, "r") as f:
        states_start = load_state_step(f[f"data/{demo_name}/states"], start)
    print(
        f"[SEGMENT] demo={demo_name}, steps={start}:{end}, close_step={close_step}, "
        f"min_dist={dist[start:end].min():.3f}",
        flush=True,
    )
    return {
        "demo_name": demo_name,
        "initial_state": states_start,
        "src_obj_pose": object_poses[start],
        "src_eef_poses": target_poses[start:end],
        "gripper": gripper[start:end],
    }


# source object pose를 기준으로 새 T-bar 위치와 yaw를 샘플링합니다.
def make_random_object_pose(src_obj_pose: np.ndarray) -> np.ndarray:
    dst = yaw_matrix(random.uniform(*args_cli.object_yaw_range_deg)) @ src_obj_pose
    dst[0, 3] = src_obj_pose[0, 3] + random.uniform(*args_cli.object_x_offset_range)
    dst[1, 3] = src_obj_pose[1, 3] + random.uniform(*args_cli.object_y_offset_range)
    dst[2, 3] = src_obj_pose[2, 3]
    return dst.astype(np.float32)


# nested state dict의 tensor들을 현재 IsaacLab device로 옮깁니다.
def move_state_to_device(state, device: str):
    if isinstance(state, dict):
        return {key: move_state_to_device(value, device) for key, value in state.items()}
    return state.to(device)


# reset_to state 안의 T-bar root pose를 새 object pose로 교체합니다.
def set_object_pose_in_state(state: dict, obj_pose: np.ndarray) -> None:
    root_pose = state["rigid_object"]["object_0"]["root_pose"]
    tabletop_pos = obj_pose[:3, 3].copy()
    tabletop_pos[2] += TABLE_TOP_Z
    root_pose[0, :3] = torch.as_tensor(tabletop_pos, dtype=root_pose.dtype)
    root_pose[0, 3:7] = torch.as_tensor(matrix_to_quat_wxyz(obj_pose[:3, :3]), dtype=root_pose.dtype)


# 4x4 target EEF pose를 IK action 벡터로 변환합니다.
def pose_matrix_to_action(pose: np.ndarray, gripper: float = 1.0) -> np.ndarray:
    return np.concatenate([pose[:3, 3], matrix_to_quat_wxyz(pose[:3, :3]), np.array([gripper], dtype=np.float32)], axis=0).astype(np.float32)


# object-centric approach 재생에 사용할 PickPlace IK 환경을 생성합니다.
def make_env() -> ManagerBasedEnv:
    env_cfg = FrankaTBarPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
    env_cfg.recorders = None
    # Day2 3교시부터 사용한 ManagerBasedEnv 생성 방식입니다.
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.8], target=[0.4, 0.0, 0.5])
    return env


# 하나의 새 T-bar pose에 대해 접근 segment를 재생합니다.
def replay_approach_once(env: ManagerBasedEnv, segment: dict, replay_index: int) -> None:
    dst_obj_pose = make_random_object_pose(segment["src_obj_pose"])
    transformed = transform_source_data_segment_using_object_pose(
        dst_obj_pose,
        segment["src_eef_poses"],
        segment["src_obj_pose"],
    )
    state = move_state_to_device(segment["initial_state"], env.device)
    set_object_pose_in_state(state, dst_obj_pose)
    env.reset_to(state, env_ids=torch.tensor([0], dtype=torch.long, device=env.device), is_relative=True)
    env.sim.forward()
    env.scene.update(env.physics_dt)
    env.sim.render()
    path_dist = np.linalg.norm(transformed[:, :2, 3] - dst_obj_pose[None, :2, 3], axis=1)
    print(
        f"[PATH {replay_index + 1}/{args_cli.num_replays}] "
        f"start_dist={path_dist[0]:.3f}, end_dist={path_dist[-1]:.3f}, min_dist={path_dist.min():.3f}",
        flush=True,
    )

    action = None
    for idx, pose in enumerate(transformed):
        grip = float(segment["gripper"][idx]) if idx < len(segment["gripper"]) else 1.0
        action = torch.as_tensor(pose_matrix_to_action(pose, gripper=grip), dtype=torch.float32, device=env.device).unsqueeze(0)
        env.step(action)
    for _ in range(args_cli.hold_steps):
        if action is not None:
            env.step(action)
    print(
        f"[REPLAY {replay_index + 1}/{args_cli.num_replays}] "
        f"new_tbar_tabletop=({dst_obj_pose[0,3]:.3f}, {dst_obj_pose[1,3]:.3f}, {dst_obj_pose[2,3]:.3f}), "
        f"root_z={dst_obj_pose[2,3] + TABLE_TOP_Z:.3f}, waypoints={len(transformed)}",
        flush=True,
    )


# ============================================================================
# 4. 메인 함수
# ============================================================================
# HDF5에서 접근 segment를 읽고 새 T-bar pose 5개에 대해 object-centric 변환을 확인합니다.
def main() -> None:
    if not Path(args_cli.input_file).exists():
        raise FileNotFoundError(f"Input HDF5 not found: {args_cli.input_file}. Run problem 2.2 first.")
    segment = load_first_approach_segment(args_cli.input_file)
    env = make_env()
    try:
        for replay_index in range(args_cli.num_replays):
            replay_approach_once(env, segment, replay_index)
    finally:
        env.close()
    print(f"[DONE] source={args_cli.input_file}, demo={segment['demo_name']}, repeats={args_cli.num_replays}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
