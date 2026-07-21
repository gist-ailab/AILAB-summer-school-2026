#!/usr/bin/env python3
"""Day3-3.4 PRACTICE: IsaacLab Mimic DataGenerator rollout.

mimic source HDF5를 공식 isaaclab_mimic DataGenerator에 연결하고,
생성된 target EEF waypoint를 IsaacLab action rollout으로 실행해 새 HDF5를 저장합니다.

수업 연결:
- 3.2.2에서 만든 mimic source HDF5를 isaaclab_mimic 공식 DataGenerator에 연결합니다.
- 3.3에서 직접 본 object-centric 변환을 라이브러리 DataGenerator가 subtask 단위로 수행합니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
import asyncio
import contextlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Practice: roll out a mimic source HDF5 with isaaclab_mimic DataGenerator.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_ready_from_teleop_10.hdf5"))
parser.add_argument("--annotated_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_source_signal_from_replay.hdf5"))
parser.add_argument("--output_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_mimic_generated_from_example_source.hdf5"))
parser.add_argument("--subtask_mode", choices=["signal_2subtask", "2subtask", "multisubtask"], default="signal_2subtask")
parser.add_argument("--generation_num_trials", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--lift_height", type=float, default=0.68)
parser.add_argument("--spawn_randomization", choices=["original", "wide"], default="original")
parser.add_argument("--visualize_subtasks", action="store_true", help="Show a color marker when a subtask signal becomes active.")
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
import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as PoseUtils  # noqa: E402
import isaaclab_mimic.datagen.generation as mimic_generation  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLMimicEnv, mdp as base_mdp  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.envs.mdp.recorders import recorders as base_recorders  # noqa: E402
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig  # noqa: E402
from isaaclab.managers import DatasetExportMode, SceneEntityCfg, TerminationTermCfg  # noqa: E402
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_mimic.datagen.generation import setup_async_generation  # noqa: E402
from task.lift.config.ik_abs_env_cfg_3_1_answer import FrankaTBarPickPlaceEnvCfg  # noqa: E402
from task.lift.mdp_3_1.terminations import object_pickplace_goal  # noqa: E402
from day3_3_utils import to_cpu_tree  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402

# 카메라 렌더링 활성화 (--enable_cameras flag 대체)
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# ============================================================================
# 3. 핵심 함수 및 문제 코드
# ============================================================================
EEF_NAME = "franka"
TASK_ID = "Isaac-Day3-TBar-PickPlace-Mimic-v0"

# 새 trajectory 생성 시 사용할 spawn randomization 범위입니다.
SPAWN_RANDOMIZATION_PRESETS = {
    "original": {
        "object_pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi / 4, math.pi / 4)},
        "bin_pose_range": None,
    },
    "wide": {
        "object_pose_range": {"x": (-0.18, 0.18), "y": (-0.18, 0.18), "yaw": (-math.pi / 4, 3 * math.pi / 4)},
        "bin_pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "pitch": (-math.pi / 6, math.pi / 6)},
    },
}


SUBTASK_MARKER_LOCAL_OFFSET = (0.13, 0.0, 0.035)

SUBTASK_DEBUG_COLORS = {
    "reset": (1.0, 0.05, 0.05),
    "object_lifted": (0.05, 0.95, 0.95),
    "object_lifted_from_height": (0.05, 0.95, 0.95),
    "approach_done": (0.10, 0.35, 1.0),
    "gripper_closed": (1.0, 0.05, 0.80),
    "object_near_bin": (0.15, 1.0, 0.20),
}


# 현재 subtask mode에서 시각화할 signal 이름을 반환합니다.
def active_debug_signal_names() -> list[str]:
    if args_cli.subtask_mode == "signal_2subtask":
        return ["object_lifted"]
    if args_cli.subtask_mode == "2subtask":
        return ["object_lifted_from_height"]
    return ["approach_done", "gripper_closed", "object_lifted_from_height", "object_near_bin"]


# debug marker용 PreviewSurface material을 만들거나 갱신합니다.
def create_debug_material(stage, env_id: int, color: tuple[float, float, float]):
    safe = "_".join(f"{int(c * 255):03d}" for c in color)
    mat_path = f"/World/day3_debug_materials/env_{env_id}_{safe}"
    shader_path = f"{mat_path}/Shader"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


# T-bar 머리(crossbar) 끝 옆에 marker를 배치해 현재 subtask를 시각화합니다.
def set_subtask_marker(env, env_id: int, color: tuple[float, float, float]) -> None:
    stage = env.sim.stage
    marker_path = f"/World/envs/env_{env_id}/day3_subtask_marker"
    marker = UsdGeom.Sphere.Define(stage, marker_path)
    marker.CreateRadiusAttr().Set(0.025)
    translate_attr = marker.GetPrim().GetAttribute("xformOp:translate")
    if not translate_attr:
        translate_attr = marker.AddTranslateOp().GetAttr()
    object_asset = env.scene["object_0"]
    object_pos = object_asset.data.root_pos_w[env_id]
    object_quat = object_asset.data.root_quat_w[env_id]
    local_offset = torch.tensor(SUBTASK_MARKER_LOCAL_OFFSET, dtype=torch.float32, device=env.device).unsqueeze(0)
    world_offset = PoseUtils.quat_apply(object_quat.unsqueeze(0), local_offset)[0]
    marker_pos = object_pos + world_offset
    translate_attr.Set(Gf.Vec3d(float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])))
    material = create_debug_material(stage, env_id, color)
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)

# 현재 활성화된 가장 뒤 subtask에 해당하는 marker 색을 반환합니다.
def current_subtask_color(signals: dict[str, torch.Tensor], env_id: int) -> tuple[float, float, float]:
    color = SUBTASK_DEBUG_COLORS["reset"]
    for name in active_debug_signal_names():
        if name in signals and float(signals[name][env_id].reshape(-1)[0].item()) >= 0.5:
            color = SUBTASK_DEBUG_COLORS[name]
    return color


# 현재 signal 값에 맞춰 marker 색만 갱신합니다.
def update_subtask_debug_visual(env) -> None:
    signals = env.get_subtask_term_signals()
    for env_id in range(env.num_envs):
        set_subtask_marker(env, env_id, current_subtask_color(signals, env_id))

# mimic rollout에서 사용할 T-bar/bin spawn randomization 범위를 환경 설정에 반영합니다.
def apply_spawn_randomization_preset(env_cfg, preset_name: str) -> None:
    preset = SPAWN_RANDOMIZATION_PRESETS[preset_name]
    env_cfg.events.reset_object.params["pose_range"] = dict(preset["object_pose_range"])
    env_cfg.events.reset_bin = None
    print(f"[SPAWN] preset={preset_name}, object={preset['object_pose_range']}, bin={preset['bin_pose_range']}", flush=True)


# DataGenerator와 IsaacLab 환경 사이에서 EEF pose, action, subtask signal을 변환하는 어댑터입니다.
class TBarPickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    # isaaclab_mimic env API에서 env_ids=None을 전체 env 선택으로 해석합니다.
    def _ids(self, env_ids: Sequence[int] | None):
        return slice(None) if env_ids is None else env_ids

    # episode reset 시 필요하면 bin pose도 추가로 randomize합니다.
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        env_ids = env_ids.to(device=self.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, dtype=torch.long, device=self.device)
        object_asset = self.scene["object_0"]
        object_pose = object_asset.data.root_state_w[env_ids, :7].clone()
        object_pose[:, 2] = self.scene.env_origins[env_ids, 2] + 0.5
        object_asset.write_root_pose_to_sim(object_pose, env_ids=env_ids)
        object_asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self.scene.write_data_to_sim()
        bin_pose_range = SPAWN_RANDOMIZATION_PRESETS[args_cli.spawn_randomization]["bin_pose_range"]
        if bin_pose_range is None:
            return
        bin_asset = self.scene["bin"]
        n = len(env_ids)

        # 지정된 pose randomization 범위에서 값을 샘플링합니다.
        def sample(key: str):
            low, high = bin_pose_range.get(key, (0.0, 0.0))
            return torch.empty(n, device=self.device).uniform_(float(low), float(high))

        dx, dy, dz = sample("x"), sample("y"), sample("z")
        droll, dpitch, dyaw = sample("roll"), sample("pitch"), sample("yaw")
        root = bin_asset.data.default_root_state[env_ids].clone()
        pos = root[:, :3] + self.scene.env_origins[env_ids]
        pos[:, 0] += dx
        pos[:, 1] += dy
        pos[:, 2] += dz
        quat = PoseUtils.quat_mul(root[:, 3:7], PoseUtils.quat_from_euler_xyz(droll, dpitch, dyaw))
        bin_asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        bin_asset.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]), env_ids=env_ids)
        self.scene.write_data_to_sim()

    # DataGenerator가 현재 robot EEF pose를 object-centric 계산에 쓰도록 제공합니다.
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if eef_name != EEF_NAME:
            raise ValueError(f"Unsupported eef_name={eef_name}")
        ids = self._ids(env_ids)
        arm_cfg = self.cfg.actions.arm_action
        robot = self.scene[arm_cfg.asset_name]
        body_ids, body_names = robot.find_bodies(arm_cfg.body_name)
        if len(body_ids) != 1:
            raise RuntimeError(f"body_name={arm_cfg.body_name!r} matched {body_names}")
        body_index = body_ids[0]
        pos, quat = PoseUtils.subtract_frame_transforms(
            robot.data.root_pos_w[ids], robot.data.root_quat_w[ids],
            robot.data.body_pos_w[ids, body_index], robot.data.body_quat_w[ids, body_index],
        )
        if arm_cfg.body_offset is not None:
            offset_pos = torch.tensor(arm_cfg.body_offset.pos, dtype=torch.float32, device=self.device).unsqueeze(0)
            offset_quat = torch.tensor(arm_cfg.body_offset.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
            pos, quat = PoseUtils.combine_frame_transforms(pos, quat, offset_pos, offset_quat)
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

    # DataGenerator가 만든 target EEF pose를 env.step action으로 변환합니다.
    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict, action_noise_dict: dict | None = None, env_id: int = 0) -> torch.Tensor:
        pose = target_eef_pose_dict[EEF_NAME]
        pos, rot = PoseUtils.unmake_pose(pose)
        if action_noise_dict is not None:
            pos = pos + float(action_noise_dict.get(EEF_NAME, 0.0)) * torch.randn_like(pos)
        quat = PoseUtils.quat_from_matrix(rot)
        grip = gripper_action_dict.get(EEF_NAME, torch.ones(1, device=self.device)).reshape(-1)[:1]
        return torch.cat([pos, quat, grip], dim=0).unsqueeze(0)

    # 저장된 action을 source datagen_info의 target EEF pose로 변환합니다.
    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        quat = action[:, 3:7] / torch.linalg.norm(action[:, 3:7], dim=-1, keepdim=True).clamp_min(1.0e-6)
        return {EEF_NAME: PoseUtils.make_pose(action[:, :3], PoseUtils.matrix_from_quat(quat)).clone()}

    # 저장된 action에서 gripper command만 분리해 mimic source 형식으로 만듭니다.
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {EEF_NAME: actions[..., 7:8]}

    # DataGenerator가 현재 object pose를 action/EEF와 같은 robot-base action frame으로 쓰도록 제공합니다.
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        ids = self._ids(env_ids)
        object_pose_matrix = {}
        for obj_name in ("object_0", "bin"):
            asset = self.scene[obj_name]
            pos = asset.data.root_pos_w[ids].clone() - self.scene.env_origins[ids]
            pos[:, 2] -= 0.5
            quat = asset.data.root_quat_w[ids].clone()
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))
        return object_pose_matrix

    # 현재 환경에서 subtask termination signal 값을 읽어 DataGenerator에 제공합니다.
    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = self._ids(env_ids)
        obj = self.scene["object_0"]
        ee = self.scene["ee_frame"]
        bin_obj = self.scene["bin"]
        obj_pos = obj.data.root_pos_w[ids] - self.scene.env_origins[ids]
        obj_pos[:, 2] -= 0.5
        ee_pos = ee.data.target_pos_w[ids, 0, :] - self.scene.env_origins[ids]
        ee_pos[:, 2] -= 0.5
        bin_pos = bin_obj.data.root_pos_w[ids] - self.scene.env_origins[ids]
        bin_pos[:, 2] -= 0.5
        action = self.action_manager.action[ids]
        object_lifted = (obj.data.root_pos_w[ids, 2] > args_cli.lift_height).reshape(-1, 1).to(torch.float32)
        return {
            "object_lifted": object_lifted,
            "object_lifted_from_height": object_lifted,
            "approach_done": (torch.linalg.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=-1, keepdim=True) < 0.07).to(torch.float32),
            "gripper_closed": (action[:, 7:8] < 0.0).to(torch.float32),
            "object_near_bin": (torch.linalg.norm(obj_pos[:, :2] - bin_pos[:, :2], dim=-1, keepdim=True) < 0.13).to(torch.float32),
        }


@configclass
class EmptyRewardsCfg:
    pass


@configclass
class EmptyCurriculumCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=object_pickplace_goal)


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


@configclass
class CpuActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    # 카메라 관측은 크므로 긴 generation 동안 CUDA 메모리에 쌓이지 않게 CPU로 옮깁니다.
    # 이렇게 하면 40개 이상 demo를 만들 때 GPU 메모리 사용량을 줄일 수 있습니다.
    record_initial_state = CpuInitialStateRecorderCfg()
    record_post_step_states = CpuPostStepStatesRecorderCfg()
    record_pre_step_actions = CpuPreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = CpuPreStepFlatPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = CpuPostStepProcessedActionsRecorderCfg()


# isaaclab_mimic이 읽는 datagen_config와 subtask_configs를 정의하는 환경 설정입니다.
@configclass
class TBarPickPlaceMimicEnvCfg(FrankaTBarPickPlaceEnvCfg, MimicEnvCfg):
    rewards: EmptyRewardsCfg = EmptyRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: EmptyCurriculumCfg = EmptyCurriculumCfg()
    commands: object | None = None

    # mimic task config, subtask config, recorder/action 설정을 마무리합니다.
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "day3_tbar_pickplace_mimic"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = args_cli.generation_num_trials
        self.datagen_config.generation_select_src_per_subtask = args_cli.subtask_mode != "2subtask"
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = max(25, args_cli.generation_num_trials * 5)
        self.datagen_config.seed = 1

        # 하나의 SubTaskConfig를 간단히 생성합니다.
        def subtask(object_ref: str, signal: str | None, offset=(0, 8), noise: float = 0.002, interp: int = 8, fixed: int = 2):
            return SubTaskConfig(
                object_ref=object_ref,
                subtask_term_signal=signal,
                subtask_term_offset_range=offset,
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=noise,
                num_interpolation_steps=interp,
                num_fixed_steps=fixed,
                apply_noise_during_interpolation=False,
            )

        if args_cli.subtask_mode == "signal_2subtask":
            self.subtask_configs[EEF_NAME] = [
                subtask("object_0", "object_lifted", noise=0.002, interp=8, fixed=2),
                subtask("bin", None, (0, 0), noise=0.002, interp=8, fixed=2),
            ]
        elif args_cli.subtask_mode == "2subtask":
            self.subtask_configs[EEF_NAME] = [
                subtask("object_0", "object_lifted_from_height", noise=0.002, interp=8, fixed=2),
                subtask("bin", None, (0, 0), noise=0.002, interp=8, fixed=2),
            ]
        else:
            self.subtask_configs[EEF_NAME] = [
                subtask("object_0", "approach_done", (0, 4), noise=0.002, interp=8, fixed=2),
                subtask("object_0", "gripper_closed", (0, 4), noise=0.0, interp=8, fixed=4),
                subtask("object_0", "object_lifted_from_height", (0, 8), noise=0.002, interp=8, fixed=2),
                subtask("bin", "object_near_bin", (0, 8), noise=0.002, interp=8, fixed=2),
                subtask("bin", None, (0, 0), noise=0.002, interp=8, fixed=2),
            ]


# gym registry에 T-bar mimic 환경을 등록합니다.
def register_env() -> None:
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point=f"{__name__}:TBarPickPlaceMimicEnv", kwargs={"env_cfg_entry_point": TBarPickPlaceMimicEnvCfg}, disable_env_checker=True)


# DataGenerator rollout에 사용할 mimic env를 생성하고 recorder를 연결합니다.
def make_mimic_env(output_file: str):
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    env_cfg = TBarPickPlaceMimicEnvCfg()
    apply_spawn_randomization_preset(env_cfg, args_cli.spawn_randomization)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
    # [문제 4-1] DataGenerator가 success term을 따로 평가할 수 있도록 env cfg에서 분리하세요.
    # 힌트: success_term을 저장한 뒤 env_cfg.terminations.success/time_out을 비웁니다.
    # [문제 4-2] generated rollout을 HDF5로 저장할 CPU recorder를 설정하세요.
    # 힌트: CpuActionStateRecorderManagerCfg + DatasetExportMode.EXPORT_SUCCEEDED_ONLY.
    raise NotImplementedError("문제 4-1/4-2: success term 분리와 generated dataset recorder 설정을 작성하세요.")
    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    return env, success_term


# generation 종료 후 남은 asyncio task를 정리합니다.
def cancel_generation_tasks(setup: dict) -> None:
    loop = setup["event_loop"]
    tasks = setup.get("tasks", [])
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))


# DataGenerator action queue를 env.step에 연결해 generation이 끝날 때까지 rollout합니다.
def env_loop_until_done(env, setup: dict) -> None:
    # [문제 4-3] DataGenerator가 action_queue에 넣은 action을 env.step()으로 실행하세요.
    # reset_queue 처리, action_queue 수집, 성공 개수 확인이 핵심입니다.
    raise NotImplementedError("문제 4-3: DataGenerator action queue를 IsaacLab rollout loop에 연결하세요.")


# source HDF5를 setup_async_generation에 연결하고 목표 성공 개수까지 실행합니다.
# ============================================================================
# 4. 메인 함수
# ============================================================================
# isaaclab_mimic DataGenerator rollout 실습 파이프라인을 실행합니다.
def main() -> None:
    # [문제 4-4] source HDF5를 setup_async_generation()에 넘기고 rollout loop를 실행하세요.
    raise NotImplementedError("문제 4-4: source HDF5 -> setup_async_generation -> rollout 흐름을 작성하세요.")


if __name__ == "__main__":
    main()
    simulation_app.close()
