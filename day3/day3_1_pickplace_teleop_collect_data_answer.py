#!/usr/bin/env python3
# ============================================================
#  day3_1_2_pickplace_teleop_collect_data.py  ·  정답(Answer)
#  Pick&Place 원격조종 수집 — 키보드로 EE(pose)를 조작해 시연을 HDF5 로 저장.
# ============================================================
# -*- coding: utf-8 -*-
"""T-bar Pick & Place teleoperation data collector.

- One standalone AppLauncher only.
- Absolute differential IK: [x, y, z, qw, qx, qy, qz].
- Se3Keyboard delta translation is integrated into the absolute target.
- Rotation keyboard commands are ignored; the controller-frame quaternion is fixed.
- Binary gripper command is appended as a separate action term.
- Only successful episodes are exported.

Keep this runner OUTSIDE task/lift/config.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from typing import Any

from isaaclab.app import AppLauncher
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

TASK_ID = "Isaac-PickPlace-TBar-Franka-Custom-v0"

parser = argparse.ArgumentParser(
    description="T-bar pick-and-place teleop: fixed downward EE, xyz + gripper."
)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--dataset_file", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "tbar_pickpalce_teleop_practice.hdf5"))
parser.add_argument("--max_steps", type=int, default=2000, help="환경별 타임아웃 스텝")
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument(
    "--linear_speed",
    type=float,
    default=0.4,
    help="Held-key target speed in m/s. Internally converted to m/step.",
)
parser.add_argument(
    "--down_quat",
    type=float,
    nargs=4,
    default=(0.0, 1.0, 0.0, 0.0),
    metavar=("W", "X", "Y", "Z"),
    help="Fixed controller-frame quaternion in Isaac Lab wxyz order.",
)
parser.add_argument(
    "--workspace_min",
    type=float,
    nargs=3,
    default=(0.15, -0.45, 0.02),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--workspace_max",
    type=float,
    nargs=3,
    default=(0.75, 0.70, 0.70),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--fallback_target_pos",
    type=float,
    nargs=3,
    default=(0.2, 0.6, 0.15),
    metavar=("X", "Y", "Z"),
    help="Used only when the env has no success termination term.",
)
parser.add_argument("--fallback_success_radius", type=float, default=0.10)
parser.add_argument(
    "--align_steps",
    type=int,
    default=45,
    help="Steps used after reset to align the EE to down_quat before recording.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.step_hz <= 0:
    parser.error("--step_hz must be positive")
if args_cli.num_demos < 0:
    parser.error("--num_demos must be >= 0 (0 means infinite)")
if args_cli.num_success_steps <= 0:
    parser.error("--num_success_steps must be positive")
if args_cli.linear_speed <= 0.0:
    parser.error("--linear_speed must be positive")
if getattr(args_cli, "headless", False):
    parser.error("Keyboard teleop cannot run with --headless")

# 카메라 렌더링 기본 활성화 (데이터 수집 시 카메라 센서 렌더링 필요)
if not hasattr(args_cli, "enable_cameras") or not args_cli.enable_cameras:
    args_cli.enable_cameras = True

# Do not create another AppLauncher in any imported task/config module.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac/Omniverse imports must happen after AppLauncher.
import torch  # noqa: E402

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

from task.lift.config.ik_abs_env_cfg_3_1_answer import FrankaTBarPickPlaceEnvCfg  # noqa: E402
from task.lift.mdp_3_1.terminations import object_pickplace_goal  # noqa: E402

from isaaclab.envs import ManagerBasedEnv

class RateLimiter:
    def __init__(self, hz: int):
        self.period = 1.0 / float(hz)
        self.render_period = min(0.033, self.period)
        self.last_time = time.time()

    def reset(self) -> None:
        self.last_time = time.time()

    def sleep(self, env: Any) -> None:
        deadline = self.last_time + self.period
        while time.time() < deadline:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.period
        while self.last_time < time.time():
            self.last_time += self.period


def action_layout(env: Any) -> tuple[dict[str, slice], int]:
    names = list(env.action_manager.active_terms)
    dims = [int(dim) for dim in env.action_manager.action_term_dim]
    slices: dict[str, slice] = {}
    start = 0
    for name, dim in zip(names, dims):
        slices[name] = slice(start, start + dim)
        start += dim

    print(f"[CHECK] action terms = {dict(zip(names, dims))}")
    if "arm_action" not in slices or "gripper_action" not in slices:
        raise RuntimeError(f"Expected arm_action and gripper_action, got {names}")
    if slices["arm_action"].stop - slices["arm_action"].start != 7:
        raise RuntimeError(
            "Absolute pose IK requires arm_action dim=7 (xyz + wxyz). "
            f"Got {dict(zip(names, dims))}."
        )
    if slices["gripper_action"].stop - slices["gripper_action"].start != 1:
        raise RuntimeError(f"gripper_action must be dim=1, got {dict(zip(names, dims))}")
    return slices, start


def controller_frame_pose(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact frame controlled by arm_action, in the robot root frame."""
    arm_cfg = env.cfg.actions.arm_action
    robot = env.scene[arm_cfg.asset_name]
    body_ids, body_names = robot.find_bodies(arm_cfg.body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"body_name={arm_cfg.body_name!r} matched {body_names}")
    body_idx = body_ids[0]

    ee_pos_w = robot.data.body_pos_w[:, body_idx]
    ee_quat_w = robot.data.body_quat_w[:, body_idx]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )

    if arm_cfg.body_offset is not None:
        offset_pos = torch.tensor(
            arm_cfg.body_offset.pos, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
        offset_quat = torch.tensor(
            arm_cfg.body_offset.rot, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, offset_pos, offset_quat
        )
    return ee_pos_b, ee_quat_b


def make_action(
    env: Any,
    slices: dict[str, slice],
    total_dim: int,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    gripper: torch.Tensor,
) -> torch.Tensor:
    actions = torch.zeros((env.num_envs, total_dim), device=env.device)
    actions[:, slices["arm_action"]] = torch.cat((target_pos, target_quat)).unsqueeze(0)
    actions[:, slices["gripper_action"]] = gripper.reshape(1, 1)
    return actions


def align_down(
    env: Any,
    slices: dict[str, slice],
    total_dim: int,
    target_pos: torch.Tensor,
    down_quat: torch.Tensor,
    steps: int,
) -> None:
    """Optional step alignment if needed."""
    open_gripper = torch.tensor([1.0], device=env.device)
    action = make_action(env, slices, total_dim, target_pos, down_quat, open_gripper)
    for _ in range(max(0, steps)):
        env.step(action)
        env.sim.render()


def reset_episode(
    env: Any,
    teleop: Se3Keyboard,
    limiter: RateLimiter,
    slices: dict[str, slice],
    total_dim: int,
    down_quat: torch.Tensor,
    align_steps: int,
) -> torch.Tensor:
    # 1. 시뮬레이션 및 환경 즉시 순간이동 초기화
    env.sim.reset()
    env.reset()
    teleop.reset()
    limiter.reset()

    # 2. 컨트롤러의 초기 목표 위치 지정
    target_pos = torch.tensor([0.46590596437454224, 4.9243681132793427e-08, 0.38296937942504883], device=env.device)

    # 3. 데이터 녹화 시작 전 초기 자세(initial pose)로 로봇을 안정화
    print(f"로봇 초기 자세로 정렬 중... ({align_steps} 스텝)")
    align_down(env, slices, total_dim, target_pos, down_quat, align_steps)

    # 4. 정렬 직후 즉시 녹화 버퍼 리셋 (정렬 과정은 녹화되지 않음)
    env.recorder_manager.reset([0])

    print(f"[RESET] current controller pose pos={target_pos.tolist()}")
    print(f"[RESET] current controller quat(wxyz)={down_quat.tolist()}")
    return target_pos


def success_now(env: Any, success_term: Any | None) -> bool:
    if success_term is not None:
        result = success_term.func(env, **success_term.params)
        if isinstance(result, torch.Tensor):
            return bool(result.reshape(-1)[0].item())
        return bool(result[0])

    result = object_pickplace_goal(env)
    if isinstance(result, torch.Tensor):
        return bool(result.reshape(-1)[0].item())
    return bool(result)


def export_success(env: Any) -> int:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
    )
    env.recorder_manager.export_episodes([0])
    return int(env.recorder_manager.exported_successful_episode_count)


def main() -> None:
    dataset_path = os.path.abspath(os.path.expanduser(args_cli.dataset_file))
    output_dir = os.path.dirname(dataset_path) or os.getcwd()
    output_name = os.path.splitext(os.path.basename(dataset_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = FrankaTBarPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # 성공 판정 term을 분리 (종료는 안 시키되 판정용으로 사용)
    success_term = None
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None:
        success_term = getattr(terminations, "success", None)
        if success_term is not None:
            terminations.success = None
        if hasattr(terminations, "time_out"):
            terminations.time_out = None

    # RecorderManager 설정 - 성공한 에피소드만 HDF5로 저장
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = ManagerBasedEnv(cfg=env_cfg)
    try:
        env.sim.reset()
        env.reset()
        env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])

        slices, total_dim = action_layout(env)

        down_quat = torch.tensor(args_cli.down_quat, dtype=torch.float32, device=env.device)
        norm = torch.linalg.norm(down_quat)
        if norm.item() < 1.0e-8:
            raise ValueError("down_quat cannot be zero")
        down_quat /= norm

        workspace_min = torch.tensor(args_cli.workspace_min, dtype=torch.float32, device=env.device)
        workspace_max = torch.tensor(args_cli.workspace_max, dtype=torch.float32, device=env.device)
        if torch.any(workspace_min >= workspace_max):
            raise ValueError("workspace_min must be lower than workspace_max")

        # Se3Keyboard output is [delta xyz, delta rotation-vector, gripper].
        # We only integrate delta xyz. The rotation-vector is discarded.
        delta_per_step = args_cli.linear_speed / float(args_cli.step_hz)
        teleop = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=delta_per_step,
                rot_sensitivity=0.15,
            )
        )

        reset_requested = False

        def request_reset() -> None:
            nonlocal reset_requested
            reset_requested = True

        teleop.add_callback("R", request_reset)
        teleop.reset()
        limiter = RateLimiter(args_cli.step_hz)

        print(f"[CHECK] fixed target quat(wxyz)={down_quat.tolist()}")
        
        target_pos = reset_episode(
            env, teleop, limiter, slices, total_dim, down_quat, args_cli.align_steps
        )
        target_yaw = torch.tensor([0.0], device=env.device)

        success_steps = 0
        env_steps = 0
        recorded_count = int(env.recorder_manager.exported_successful_episode_count)
        print(f"[INFO] 기존 저장된 데모 수: {recorded_count} / 목표: {args_cli.num_demos}")

        print("\n=== T-bar Pick & Place Teleop ===")
        print("W/S: x, A/D: y, Q/E: z, K: gripper, R: discard/reset")
        print("Rotation keys are ignored; the EE controller frame is held at down_quat.")
        print(f"Action dim={total_dim}; expected 8 = arm7 + gripper1")
        print(f"Dataset: {dataset_path}\n")

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                if reset_requested or env_steps >= args_cli.max_steps:
                    target_pos = reset_episode(
                        env, teleop, limiter, slices, total_dim, down_quat, args_cli.align_steps
                    )
                    target_yaw = torch.tensor([0.0], device=env.device)
                    success_steps = 0
                    env_steps = 0
                    reset_requested = False
                    continue

                command = teleop.advance().to(env.device)
                if command.shape != (7,):
                    raise RuntimeError(f"Se3Keyboard returned {tuple(command.shape)}, expected (7,)")

                target_pos = target_pos + command[:3]
                target_pos = torch.maximum(
                    torch.minimum(target_pos, workspace_max), workspace_min
                )
                
                target_yaw = target_yaw + command[5:6]
                yaw_quat = math_utils.quat_from_euler_xyz(
                    torch.zeros(1, device=env.device), 
                    torch.zeros(1, device=env.device), 
                    target_yaw
                ).squeeze(0)
                final_quat = math_utils.quat_mul(yaw_quat, down_quat)
                
                gripper = command[6:7]  # +1 open, -1 close

                actions = make_action(
                    env, slices, total_dim, target_pos, final_quat, gripper
                )
                env.step(actions)
                env_steps += 1

                if success_now(env, success_term):
                    success_steps += 1
                    print(f"[바구니 안착] 유지 중... ({success_steps}/{args_cli.num_success_steps} 스텝)")
                else:
                    if success_steps > 0:
                        print("[바구니 이탈] 성공 카운트 초기화")
                    success_steps = 0
                    # 바구니 근처(수평 0.25m 이내)에 있지만 안착 성공 조건(z<0.10, xy<0.15) 미달 시 실시간 안내
                    if env_steps % 15 == 0 and "object_0" in env.scene.rigid_objects and "bin" in env.scene.rigid_objects:
                        obj_0 = env.scene["object_0"]
                        bin_obj = env.scene["bin"]
                        d_xy = torch.norm(bin_obj.data.root_pos_w[:, :2] - obj_0.data.root_pos_w[:, :2], dim=1).item()
                        d_z = abs(bin_obj.data.root_pos_w[0, 2] - obj_0.data.root_pos_w[0, 2]).item()
                        if d_xy < 0.25:
                            if d_z >= 0.10:
                                print(f"[안착 미달] 높이가 높습니다. Z를 더 낮춰 바구니 안에 떨어뜨려주세요! (Z차이: {d_z:.2f}m / 기준: <0.10m)")
                            elif d_xy >= 0.15:
                                print(f"[안착 미달] 바구니 중앙으로 더 이동해주세요. (수평거리: {d_xy:.2f}m / 기준: <0.15m)")

                if success_steps >= args_cli.num_success_steps:
                    recorded_count = export_success(env)
                    print(f">>> success: {recorded_count}/{args_cli.num_demos or 'infinite'}")
                    if args_cli.num_demos > 0 and recorded_count >= args_cli.num_demos:
                        break
                    target_pos = reset_episode(
                        env, teleop, limiter, slices, total_dim, down_quat, args_cli.align_steps
                    )
                    target_yaw = torch.tensor([0.0], device=env.device)
                    success_steps = 0
                    env_steps = 0
                    reset_requested = False
                    continue

                if env.sim.is_stopped():
                    break
                limiter.sleep(env)

        print(f"Finished with {recorded_count} successful demonstrations")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()