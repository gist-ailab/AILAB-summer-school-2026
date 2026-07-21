#!/usr/bin/env python3
"""Day3-3.2.2 ANSWER: State replay and mimic-ready datagen_info recording.

PickPlace teleop HDF5의 simulator state를 frame별로 복원하면서 camera와
isaaclab_mimic에 필요한 eef/object/target/gripper/subtask 정보를 다시 기록합니다.

수업 연결:
- 3.1.1의 state replay 방식을 PickPlace 데이터에 적용합니다.
- Day3 1교시 데이터 수집에서 사용한 recorder 개념을 확장해 mimic용 datagen_info를 추가 저장합니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DAY3_ROOT.parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay IK pose trajectory and record isaaclab_mimic datagen_info.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/tbar_pickpalce_teleop_practice.hdf5"))
parser.add_argument("--replayed_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_ready_from_teleop_10.hdf5"))
parser.add_argument("--source_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_source_signal_from_replay.hdf5"))
parser.add_argument("--num_demos", type=int, default=10)
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
from isaaclab.managers import DatasetExportMode  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.envs.mdp.recorders import recorders as base_recorders  # noqa: E402
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from day3_3_utils import copy_h5_attrs, copy_h5_item, delete_if_exists, ensure_h5_data_attrs, recreate_h5_dataset, selected_demo_names, to_cpu_tree  # noqa: E402
from task.lift.config.ik_abs_env_cfg_3_1_answer import FrankaTBarPickPlaceEnvCfg  # noqa: E402

# 카메라 렌더링 활성화 (--enable_cameras flag 대체)
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# ============================================================================
# 3. 핵심 함수 및 문제 코드
# ============================================================================
# replay 중 mimic datagen_info를 저장하는 recorder입니다.
EEF_NAME = "franka"


# recording 값이 num_envs batch shape을 갖도록 검증하고 보정합니다.
def as_batch(tensor: torch.Tensor, num_envs: int, trailing_dims: tuple[int, ...], name: str) -> torch.Tensor:
    tensor = tensor.clone()
    if tensor.shape == trailing_dims and num_envs == 1:
        tensor = tensor.unsqueeze(0)
    expected_rank = 1 + len(trailing_dims)
    if tensor.dim() != expected_rank or tensor.shape[0] != num_envs or tuple(tensor.shape[1:]) != trailing_dims:
        raise RuntimeError(f"{name} must have shape ({num_envs}, {trailing_dims}), got {tuple(tensor.shape)}")
    return tensor.contiguous()


# position과 wxyz quaternion을 isaaclab_mimic이 쓰는 4x4 pose matrix로 바꿉니다.
def pose7_to_matrix(pos: torch.Tensor, quat_wxyz: torch.Tensor, num_envs: int, name: str) -> torch.Tensor:
    pos = as_batch(pos, num_envs, (3,), f"{name}.pos")
    quat_wxyz = as_batch(quat_wxyz, num_envs, (4,), f"{name}.quat")
    quat_wxyz = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1.0e-6)
    pose = math_utils.make_pose(pos, math_utils.matrix_from_quat(quat_wxyz))
    return as_batch(pose, num_envs, (4, 4), name)


class CpuInitialStateRecorder(base_recorders.InitialStateRecorder):
    # recorder 값을 CPU로 옮겨 GPU memory 누적을 줄입니다.
    def record_post_reset(self, env_ids: Sequence[int] | None):
        key, value = super().record_post_reset(env_ids)
        return key, to_cpu_tree(value)


class CpuPostStepStatesRecorder(base_recorders.PostStepStatesRecorder):
    # step 이후 recorder 값을 CPU로 옮겨 저장합니다.
    def record_post_step(self):
        key, value = super().record_post_step()
        return key, to_cpu_tree(value)


class CpuPreStepActionsRecorder(base_recorders.PreStepActionsRecorder):
    # step 이전 action/observation/datagen_info를 CPU recorder에 저장합니다.
    def record_pre_step(self):
        key, value = super().record_pre_step()
        return key, to_cpu_tree(value)


class CpuPreStepFlatPolicyObservationsRecorder(base_recorders.PreStepFlatPolicyObservationsRecorder):
    # step 이전 action/observation/datagen_info를 CPU recorder에 저장합니다.
    def record_pre_step(self):
        key, value = super().record_pre_step()
        return key, to_cpu_tree(value)


class CpuPostStepProcessedActionsRecorder(base_recorders.PostStepProcessedActionsRecorder):
    # step 이후 recorder 값을 CPU로 옮겨 저장합니다.
    def record_post_step(self):
        key, value = super().record_post_step()
        return key, to_cpu_tree(value)


@configclass
class CpuInitialStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CpuInitialStateRecorder


@configclass
class CpuPostStepStatesRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CpuPostStepStatesRecorder


@configclass
class CpuPreStepActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CpuPreStepActionsRecorder


@configclass
class CpuPreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CpuPreStepFlatPolicyObservationsRecorder


@configclass
class CpuPostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CpuPostStepProcessedActionsRecorder


class PreStepMimicDatagenInfoRecorder(RecorderTerm):
    """Record the nested ``obs/datagen_info`` structure used by isaaclab_mimic.

    The term is evaluated before the current action is applied. This keeps the
    source state, object pose, and action target aligned in the usual imitation
    learning convention: state_t paired with action_t.
    """

    # mimic datagen_info recorder가 참조할 scene entity 설정을 초기화합니다.
    def __init__(self, cfg: "PreStepMimicDatagenInfoRecorderCfg", env):
        super().__init__(cfg, env)
        self.eef_name = cfg.eef_name
        self.object_name = cfg.object_name
        self.bin_name = cfg.bin_name
        self.robot_base_z = cfg.robot_base_z
        self.lift_height = cfg.lift_height
        self.record_gripper_action = cfg.record_gripper_action

    # world pose를 env origin 기준 상대 pose matrix로 변환합니다.
    def _relative_pose_matrix(self, asset_name: str) -> torch.Tensor:
        asset = self._env.scene[asset_name]
        pos = asset.data.root_pos_w - self._env.scene.env_origins
        pos[:, 2] -= self.robot_base_z
        quat = asset.data.root_quat_w
        return pose7_to_matrix(pos, quat, self._env.num_envs, asset_name)

    # 현재 EEF pose를 datagen_info 형식으로 계산합니다.
    def _eef_pose_matrix(self) -> torch.Tensor:
        ee_frame = self._env.scene["ee_frame"]
        pos = ee_frame.data.target_pos_w[:, 0, :].clone() - self._env.scene.env_origins
        quat = ee_frame.data.target_quat_w[:, 0, :].clone()
        pos[:, 2] -= self.robot_base_z
        return pose7_to_matrix(pos, quat, self._env.num_envs, "eef_pose")

    # 현재 action target pose를 datagen_info 형식으로 계산합니다.
    def _target_eef_pose_matrix(self) -> torch.Tensor:
        action = self._env.action_manager.action
        pos = action[:, :3].clone()
        quat = action[:, 3:7].clone()
        return pose7_to_matrix(pos, quat, self._env.num_envs, "target_eef_pose")

    # step 이전 action/observation/datagen_info를 CPU recorder에 저장합니다.
    def record_pre_step(self):
        action = self._env.action_manager.action
        object_pos = self._env.scene[self.object_name].data.root_pos_w - self._env.scene.env_origins
        datagen_info = {
            "eef_pose": {self.eef_name: self._eef_pose_matrix()},
            "target_eef_pose": {self.eef_name: self._target_eef_pose_matrix()},
            "object_pose": {
                self.object_name: self._relative_pose_matrix(self.object_name),
                self.bin_name: self._relative_pose_matrix(self.bin_name),
            },
            "subtask_term_signals": {
                "object_lifted": as_batch((object_pos[:, 2:3] > self.lift_height).to(torch.float32), self._env.num_envs, (1,), "object_lifted"),
            },
        }
        if self.record_gripper_action:
            datagen_info["gripper_action"] = {self.eef_name: as_batch(action[:, 7:8], self._env.num_envs, (1,), "gripper_action")}
        return "obs/datagen_info", to_cpu_tree(datagen_info)


@configclass
class PreStepMimicDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepMimicDatagenInfoRecorder
    eef_name: str = EEF_NAME
    object_name: str = "object_0"
    bin_name: str = "bin"
    robot_base_z: float = 0.5
    lift_height: float = 0.68
    record_gripper_action: bool = True


@configclass
class MimicActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus mimic annotations, stored on CPU to avoid CUDA OOM at export."""

    record_initial_state = CpuInitialStateRecorderCfg()
    record_post_step_states = CpuPostStepStatesRecorderCfg()
    record_pre_step_actions = CpuPreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = CpuPreStepFlatPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = CpuPostStepProcessedActionsRecorderCfg()
    record_pre_step_mimic_datagen_info = PreStepMimicDatagenInfoRecorderCfg()

# --- End recorder definitions ---


# --- Signal source construction used after replay ---
EEF_NAME = "franka"


# 0에서 1로 처음 바뀌는 subtask signal index를 찾습니다.
def first_transition(signal: np.ndarray) -> int | None:
    flat = np.asarray(signal).reshape(-1)
    active = (flat > 0.5).astype(np.int32)
    diffs = active[1:] - active[:-1]
    hits = np.flatnonzero(diffs > 0)
    if hits.size > 0:
        return int(hits[0] + 1)
    hits = np.flatnonzero(active)
    return int(hits[0]) if hits.size > 0 else None


# subtask 경계 index가 episode 범위 안에 들어오도록 보정합니다.
def valid_transition_step(length: int, step: int | None) -> int:
    if length < 2:
        raise ValueError(f"Need at least 2 steps, got {length}")
    if step is None:
        step = length // 2
    return int(np.clip(step, 1, length - 1))


# 지정 step부터 1이 되는 0/1 subtask termination signal을 만듭니다.
def step_signal(length: int, step: int) -> np.ndarray:
    out = np.zeros((length, 1), dtype=np.float32)
    out[step:, 0] = 1.0
    return out



# HDF5 demo가 mimic source 생성에 필요한 datagen_info를 갖췄는지 확인합니다.
def assert_mimic_ready(demo, demo_name: str) -> None:
    required = [
        "actions",
        "obs/datagen_info/eef_pose/franka",
        "obs/datagen_info/object_pose/object_0",
        "obs/datagen_info/object_pose/bin",
        "obs/datagen_info/target_eef_pose/franka",
        "obs/datagen_info/gripper_action/franka",
        "obs/datagen_info/subtask_term_signals/object_lifted",
    ]
    missing = [path for path in required if path not in demo]
    if missing:
        raise RuntimeError(f"{demo_name} is not mimic-ready. Missing: {missing}")


# replay 중 기록한 signal을 사용해 2-subtask mimic source HDF5를 만듭니다.
def create_signal_2subtask_source(input_file: str, output_file: str, num_demos: int = 0) -> None:
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
        f.attrs["subtask_mode"] = "signal_2subtask"
        for name in selected_demo_names(f["data"], num_demos):
            demo = f[f"data/{name}"]
            assert_mimic_ready(demo, name)
            length = int(demo["actions"].shape[0])
            raw_step = first_transition(demo["obs/datagen_info/subtask_term_signals/object_lifted"][:])
            step = valid_transition_step(length, raw_step)
            signals = demo.require_group("obs/datagen_info/subtask_term_signals")
            recreate_h5_dataset(signals, "object_lifted", step_signal(length, step))
            demo.attrs["subtask_summary"] = json.dumps({"mode": "signal_2subtask", "object_lifted": step, "raw_object_lifted": raw_step})
            print(f"[SIGNAL-2] {name}: object_lifted={step}, raw={raw_step}", flush=True)
# --- End signal source construction ---


# 입력 HDF5 경로가 없을 때 루트 data 폴더의 기본 파일로 대체합니다.
def resolve_input_file(path: str) -> str:
    input_path = Path(path)
    fallback = DAY3_ROOT / "datasets/tbar_pickpalce_teleop_practice.hdf5"
    if input_path.exists():
        return str(input_path)
    if fallback.exists():
        print(f"[INPUT] {input_path} not found. Using fallback: {fallback}", flush=True)
        return str(fallback)
    raise FileNotFoundError(f"No input HDF5 found: {input_path}")


# HDF5에 저장된 특정 step의 simulator state를 reset_to 입력으로 읽습니다.
def load_state_step(group: h5py.Group, step: int, device: str):
    state = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            state[key] = torch.as_tensor(item[step], device=device).unsqueeze(0)
        else:
            state[key] = load_state_step(item, step, device)
    return state


# state replay와 mimic datagen_info 기록을 위한 환경과 recorder를 생성합니다.
def make_env(output_file: str) -> ManagerBasedEnv:
    env_cfg = FrankaTBarPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
    env_cfg.recorders = None
# Day2 3교시부터 사용한 ManagerBasedEnv 생성 방식입니다.
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    return env


# 저장된 simulator state를 복원한 뒤 카메라 관측을 다시 계산합니다.
def restore_state_and_render(env: ManagerBasedEnv, state, env_ids: torch.Tensor):
# 3.1.1에서 다룬 핵심 패턴: 저장된 simulator state를 그대로 복원합니다.
    env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
    env.sim.forward()
    env.scene.update(env.physics_dt)
    env.sim.render()
    return env.observation_manager.compute_group("policy", update_history=False)


# torch tensor를 recorder에 저장 가능한 CPU numpy array로 변환합니다.
def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


# 현재 end-effector pose를 4x4 matrix로 읽습니다.
def current_eef_pose_matrix(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the actual TCP pose in the same robot-root frame as arm actions."""
    arm_cfg = env.cfg.actions.arm_action
    robot = env.scene[arm_cfg.asset_name]
    body_ids, body_names = robot.find_bodies(arm_cfg.body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"body_name={arm_cfg.body_name!r} matched {body_names}")
    body_index = body_ids[0]
    pos, quat = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_index],
        robot.data.body_quat_w[:, body_index],
    )
    if arm_cfg.body_offset is not None:
        offset_pos = torch.tensor(arm_cfg.body_offset.pos, dtype=torch.float32, device=env.device).unsqueeze(0)
        offset_quat = torch.tensor(arm_cfg.body_offset.rot, dtype=torch.float32, device=env.device).unsqueeze(0)
        pos, quat = math_utils.combine_frame_transforms(pos, quat, offset_pos, offset_quat)
    return pose7_to_matrix(pos, quat, env.num_envs, "eef_pose")


# 현재 rigid object pose를 4x4 matrix로 읽습니다.
def current_object_pose_matrix(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    asset = env.scene[asset_name]
    pos = asset.data.root_pos_w - env.scene.env_origins
    pos[:, 2] -= 0.5
    quat = asset.data.root_quat_w
    return pose7_to_matrix(pos, quat, env.num_envs, asset_name)


# 저장된 IK pose action을 target_eef_pose 4x4 matrix로 변환합니다.
def target_pose_matrix_from_action(action_np: np.ndarray, device: str) -> torch.Tensor:
    action = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
    return pose7_to_matrix(action[:, :3], action[:, 3:7], 1, "target_eef_pose")


# state replay로 camera와 datagen_info가 포함된 mimic-ready dataset을 생성합니다.
def replay_to_mimic_ready(input_file: str, output_file: str, num_demos: int) -> None:
    input_file = resolve_input_file(input_file)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    env = make_env(output_file)
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)

    with h5py.File(input_file, "r") as src, h5py.File(output_file, "w") as dst:
        copy_h5_attrs(src, dst)
        src_data = src["data"]
        dst_data = dst.create_group("data")
        ensure_h5_data_attrs(dst_data, src_data)
        demo_names = selected_demo_names(src_data, num_demos)

        for demo_index, demo_name in enumerate(demo_names, start=1):
            src_demo = src_data[demo_name]
            dst_demo = dst_data.create_group(f"demo_{demo_index - 1}")
            copy_h5_attrs(src_demo, dst_demo)
            dst_demo.attrs["source_demo"] = demo_name
            dst_demo.attrs["replay_mode"] = "state_replay_mimic_ready"
            for key in src_demo.keys():
                copy_h5_item(src_demo, dst_demo, key)

            actions = src_demo["actions"][:]
            num_steps = int(actions.shape[0])
            top_frames, wrist_frames = [], []
            eef_poses, target_poses, object_poses, bin_poses = [], [], [], []
            gripper_actions, object_lifted = [], []
            t0 = time.perf_counter()

            for step, action_np in enumerate(actions):
                state = load_state_step(src_demo["states"], step, env.device)
                obs = restore_state_and_render(env, state, env_ids)
                top_frames.append(tensor_to_numpy(obs["top_cam"][0]))
                wrist_frames.append(tensor_to_numpy(obs["wrist_cam"][0]))

                obj_pos = env.scene["object_0"].data.root_pos_w - env.scene.env_origins
                eef_poses.append(tensor_to_numpy(current_eef_pose_matrix(env)[0]))
                target_poses.append(tensor_to_numpy(target_pose_matrix_from_action(action_np, env.device)[0]))
                object_poses.append(tensor_to_numpy(current_object_pose_matrix(env, "object_0")[0]))
                bin_poses.append(tensor_to_numpy(current_object_pose_matrix(env, "bin")[0]))
                gripper_actions.append(np.asarray([action_np[7]], dtype=np.float32))
                object_lifted.append(np.asarray([float(obj_pos[0, 2] > 0.68)], dtype=np.float32))

                if (step + 1) % 100 == 0:
                    print(f"[STATE-REPLAY] {demo_name}: {step + 1}/{num_steps}", flush=True)

            obs_group = dst_demo.require_group("obs")
            delete_if_exists(obs_group, "top_cam")
            delete_if_exists(obs_group, "wrist_cam")
            obs_group.create_dataset("top_cam", data=np.asarray(top_frames), compression="gzip")
            obs_group.create_dataset("wrist_cam", data=np.asarray(wrist_frames), compression="gzip")

            datagen = obs_group.require_group("datagen_info")
            for key in list(datagen.keys()):
                del datagen[key]
            eef_group = datagen.create_group("eef_pose")
            eef_group.create_dataset(EEF_NAME, data=np.asarray(eef_poses, dtype=np.float32), compression="gzip")
            target_group = datagen.create_group("target_eef_pose")
            target_group.create_dataset(EEF_NAME, data=np.asarray(target_poses, dtype=np.float32), compression="gzip")
            obj_group = datagen.create_group("object_pose")
            obj_group.create_dataset("object_0", data=np.asarray(object_poses, dtype=np.float32), compression="gzip")
            obj_group.create_dataset("bin", data=np.asarray(bin_poses, dtype=np.float32), compression="gzip")
            grip_group = datagen.create_group("gripper_action")
            grip_group.create_dataset(EEF_NAME, data=np.asarray(gripper_actions, dtype=np.float32), compression="gzip")
            sig_group = datagen.create_group("subtask_term_signals")
            sig_group.create_dataset("object_lifted", data=np.asarray(object_lifted, dtype=np.float32), compression="gzip")

            print(
                f"[EXPORT] {demo_name} -> demo_{demo_index - 1}, frames={num_steps}, dt={time.perf_counter() - t0:.2f}s",
                flush=True,
            )

        dst_data.attrs["total"] = sum(int(demo["actions"].shape[0]) for demo in dst_data.values())
        dst.flush()

    env.close()
    print(f"[DONE] state-replayed mimic-ready dataset: {output_file}", flush=True)


# ============================================================================
# 4. 메인 함수
# ============================================================================
# state replay와 mimic-ready source 생성 파이프라인을 실행합니다.
def main() -> None:
    replay_to_mimic_ready(args_cli.input_file, args_cli.replayed_file, args_cli.num_demos)
    create_signal_2subtask_source(args_cli.replayed_file, args_cli.source_file, args_cli.num_demos)
    print(f"[DONE] replay source: {args_cli.source_file}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
