# ============================================================
#  day3_2_1_pusht_teleop_collect_data.py  ·  정답(Answer) - 수정본
#  PushT 원격조종 수집 — 키보드로 물체를 밀어 목표 자세에 맞추는 시연을 HDF5 로 저장.
# ============================================================
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
사람이 원격 조종(Teleoperation)을 통해 Isaac Lab 환경에서 데모스터레이션(Demonstrations)을 기록하기 위한 스크립트.

이 스크립트를 사용하면 특정 태스크에 대해 사람의 조종 데이터를 수집할 수 있습니다.
기록된 데모 데이터는 hdf5 파일의 에피소드 형태로 저장됩니다. 사용자는 커맨드라인 인자를 통해
태스크 이름, 원격 조종 장치, 데이터셋 저장 경로, 환경 실행 속도 등을 설정할 수 있습니다.

필수 인자:
    --task                    태스크 이름.

선택 인자:
    -h, --help                도움말 메시지를 출력하고 종료
    --teleop_device           환경과 상호작용할 원격 조종 장치. (기본값: keyboard)
    --dataset_file            기록된 데모를 내보낼 파일 경로. (기본값: "./datasets/dataset.hdf5")
    --step_hz                 환경 실행 속도(Hz). (기본값: 30)
    --num_demos               기록할 데모의 개수. 0으로 설정하면 무한 기록. (기본값: 0)
    --num_success_steps       데모를 성공으로 간주하기 위해 연속으로 성공해야 하는 스텝 수. (기본값: 10)
"""

"""먼저 Isaac Sim 시뮬레이터를 실행합니다."""

# 기본 라이브러리 임포트
import argparse
import contextlib

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

# argparse 인자 추가
parser = argparse.ArgumentParser(description="Isaac Lab 환경을 위한 데모 기록 스크립트.")
parser.add_argument("--task", type=str, default="Template-PushT-Franka-v0", help="태스크 이름.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "원격 조종 장치. 이곳에 설정하거나 환경 설정(config) 파일에서 설정할 수 있습니다. "
        "내장 지원: keyboard, spacemouse, gamepad."
    ),
)
parser.add_argument(
    "--dataset_file", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "tbar_pusht_teleop_practice.hdf5"), help="기록된 데모를 내보낼 파일 경로."
)
parser.add_argument("--step_hz", type=int, default=30, help="환경 실행 속도(Hz).")
parser.add_argument(
    "--num_demos", type=int, default=0, help="기록할 데모의 개수. 0으로 설정하면 무한 기록."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=15,
    help="데모를 성공으로 간주하기 위해 연속으로 성공해야 하는 스텝 수 (30Hz 기준 0.5초 = 15 스텝).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Pinocchio 활성화 여부.",
)

# AppLauncher 커맨드라인 인자 추가
AppLauncher.add_app_launcher_args(parser)
# 인자 파싱
args_cli = parser.parse_args()

# 카메라 렌더링 기본 활성화 (데이터 수집 시 카메라 센서 렌더링 필요)
if not hasattr(args_cli, "enable_cameras") or not args_cli.enable_cameras:
    args_cli.enable_cameras = True

# 필수 인자 검증
if args_cli.task is None:
    args_cli.task = "Template-PushT-Franka-v0"

# 카메라 렌더링 기본 활성화 (teleop 시 시각화 필요)
if not hasattr(args_cli, "enable_cameras") or not args_cli.enable_cameras:
    args_cli.enable_cameras = True

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # AppLauncher 이전에 pinocchio를 임포트하여 Isaac Sim 설치본이 아닌 
    # IsaacLab 버전을 강제로 사용하도록 합니다.
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# 시뮬레이터 런칭
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 서드파티 라이브러리 임포트
import logging
import os
import time

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, euler_xyz_from_quat, wrap_to_pi
import omni.ui as ui

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from collections.abc import Callable
from typing import Any

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedEnvCfg, ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401

# T-bar Pick & Place (PushT) 환경 등록 - ManagerBasedEnv 기반
from task.lift.custom_pusht_env_cfg_3_2_answer import PushTEnvCfg
from task.lift.mdp_3_2.terminations_answer import object_pusht_goal

# 로거 설정
logger = logging.getLogger(__name__)


class RateLimiter:
    """루프에서 실행 속도(주파수)를 강제하기 위한 편의 클래스."""

    def __init__(self, hz: int):
        """지정된 주파수로 RateLimiter를 초기화합니다.

        Args:
            hz: 강제할 주파수 (Hertz).
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """지정된 hz 속도에 맞춰 대기를 시도합니다.

        Args:
            env: 대기 시간 동안 렌더링할 환경 인스턴스.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # 시간이 앞으로 튀는 현상 방지 (루프가 너무 느린 경우)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """데모를 저장할 출력 디렉토리를 설정합니다."""
    # 커맨드라인 인자에서 디렉토리 경로와 파일 이름(확장자 제외) 가져오기
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성됨: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedEnvCfg | DirectRLEnvCfg, Any | None]:
    # 구성 파싱
    try:
        env_cfg = PushTEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"환경 구성을 파싱하는데 실패했습니다: {e}")
        exit(1)

    if args_cli.xr:
        # 카메라가 비활성화되어 있고 XR이 활성화된 경우 카메라 구성 제거
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    success_term = None
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None:
        success_term = getattr(terminations, "success", None)
        if success_term is not None:
            terminations.success = None
        if hasattr(terminations, "time_out"):
            terminations.time_out = None

    # 데이터 수집(Recorder) 매니저 설정
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedEnvCfg | DirectRLEnvCfg) -> ManagerBasedEnv:
    """구성을 바탕으로 시뮬레이션 환경을 생성합니다."""
    try:
        env = ManagerBasedEnv(cfg=env_cfg)
        return env
    except Exception as e:
        logger.error(f"환경을 생성하는데 실패했습니다: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    """구성을 바탕으로 원격 조종 장치를 설정합니다."""
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"환경 설정에 '{args_cli.teleop_device}' 장치가 없습니다. 기본 장치를 생성합니다."
            )
            # 대체 조종 장치 생성
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.4 / args_cli.step_hz, rot_sensitivity=0.15))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            else:
                logger.error(f"지원하지 않는 원격 조종 장치입니다: {args_cli.teleop_device}")
                logger.error("지원 장치: keyboard, spacemouse, handtracking")
                exit(1)

            # 대체 장치에 콜백 추가
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"원격 조종 장치를 생성하는데 실패했습니다: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("원격 조종 인터페이스 생성 실패")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """사용자 인터페이스(UI) 요소를 설정합니다."""
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_step_count: int, success_term: Any | None = None) -> tuple[int, bool]:
    """현재 스텝에 대한 성공 조건을 추적하여 처리합니다.
    환경 설정에 정의된 termination term을 최우선으로 사용합니다."""
    if success_term is not None:
        is_success = success_term.func(env, **success_term.params)
    else:
        is_success = object_pusht_goal(env)
    
    if isinstance(is_success, torch.Tensor):
        is_success_bool = bool(is_success.reshape(-1)[0].item())
    else:
        is_success_bool = bool(is_success[0])
    
    if is_success_bool:
        success_step_count += 1
        print(f"[목표 영역 진입] 유지 중... ({success_step_count}/{args_cli.num_success_steps} 스텝)")
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("성공 조건 달성! 기록이 완료되었습니다.")
            return success_step_count, True
    else:
        if success_step_count > 0:
            print("[목표 영역 이탈] 카운트 리셋")
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str, teleop_interface: object
) -> int:
    """환경 초기화(리셋) 처리를 담당합니다."""
    print("환경 리셋 중...")
    env.sim.reset()
    env.reset()
    if teleop_interface is not None:
        teleop_interface.reset()
        
    # 초기 정렬(Align) 수행: 리셋 직후 불안정한 상태를 잡고 그리퍼를 초기화
    target_pos = torch.tensor([[0.4, 0.0, 0.005]], device=env.device).repeat(env.num_envs, 1)
    down_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    gripper_command = torch.tensor([[-1.0]], device=env.device).repeat(env.num_envs, 1)
    action = torch.cat([target_pos, down_quat, gripper_command], dim=-1)
    
    for _ in range(10):
        env.step(action)
        env.sim.render()
        
    env.recorder_manager.reset()
    
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: ManagerBasedEnv,
    teleop_interface: object | None,
    rate_limiter: RateLimiter | None,
    success_term: Any | None = None,
) -> int:
    """데모 수집을 위한 메인 시뮬레이션 루프를 실행합니다."""
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr

    # 원격 조종 장치를 위한 콜백 클로저
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("기록 인스턴스 리셋 요청됨")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("기록 시작됨")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("기록 일시정지됨")

    # 조종 장치 콜백 등록
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    # "R" 콜백은 이미 teleoperation_callbacks에 포함되어 setup_teleop_device() 내부에서
    # 등록되므로, 여기서 다시 add_callback("R", ...)을 호출할 필요가 없습니다(중복 등록 방지).
    teleop_interface = setup_teleop_device(teleoperation_callbacks)

    # 시작 전 초기화
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    label_text = f"기록된 성공 데모 수: {current_recorded_demo_count}"
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    target_pos = torch.tensor([0.4, 0.0, 0.005], device=env.device)
    target_yaw = torch.tensor([0.0], device=env.device)
    down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)
    sensitivity = 0.15 / args_cli.step_hz  # 0.15 m/s

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            # 키보드(또는 조종기) 액션 획득 (dx, dy, dz, drx, dry, drz, gripper)
            command = teleop_interface.advance().to(device=env.device, dtype=torch.float32)
            
            # 회전: Z축 회전(yaw)만 반영하고 나머지는 무시
            # 높이(Z축) 고정: 조종기(키보드)의 Z축 입력은 무시
            target_pos[0] += command[0]
            target_pos[1] += command[1]
            target_pos[2] = 0.005  # EE 높이를 0.005로 강제 고정
            
            # EE(End-Effector) 이동 범위(Workspace) 제한 (Z >= 0.05, X/Y 안전 반경)
            target_pos[0] = torch.clamp(target_pos[0], min=0.15, max=0.7)
            target_pos[1] = torch.clamp(target_pos[1], min=-0.55, max=0.55)
            
            target_yaw = target_yaw + command[5]
            
            yaw_quat = quat_from_euler_xyz(
                torch.zeros(1, device=env.device), 
                torch.zeros(1, device=env.device), 
                target_yaw
            ).squeeze(0)
            
            # 절대 IK 액션 조립: (xyz + (yaw_quat * down_quat) + gripper)
            final_quat = quat_mul(yaw_quat, down_quat)
            # 그리퍼는 항상 닫힌 상태(-1.0)로 고정
            gripper_command = torch.tensor([-1.0], device=env.device)
            action = torch.cat([target_pos, final_quat, gripper_command], dim=-1)

            # 배치 차원으로 확장
            actions = action.repeat(env.num_envs, 1)

            # 환경에 액션 적용
            if running_recording_instance:
                # ManagerBasedEnv는 obv, extras를 리턴합니다 (RL 환경과 반환값 다름 주의)
                obv, _ = env.step(actions)  
            else:
                env.sim.render()

            # 성공 조건 확인
            if running_recording_instance:
                success_step_count, success_reset_needed = process_success_condition(env, success_step_count, success_term)
                if success_reset_needed:
                    should_reset_recording_instance = True

            # 데모 카운트가 변경되었을 경우 업데이트
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"기록된 성공 데모 수: {current_recorded_demo_count}"
                print(label_text)

            # 원하는 데모 개수에 도달했는지 확인
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"목표한 {current_recorded_demo_count}개 데모 기록을 모두 완료했습니다.\n앱을 종료합니다."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # 리셋 요청이 있으면 리셋 처리
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text, teleop_interface)
                target_pos = torch.tensor([0.4, 0.0, 0.005], device=env.device)
                target_yaw = torch.tensor([0.0], device=env.device)
                should_reset_recording_instance = False

            # 시뮬레이션이 정지되었는지 확인
            if env.sim.is_stopped():
                break

            # 속도 제한(Rate limiting) 적용
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    # 핸드트래킹(handtracking) 선택 시 OpenXR을 통해 속도 제한 적용
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # 텔레옵 시각화 매니저 할당
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # 출력 디렉토리 설정
    output_dir, output_file_name = setup_output_directories()

    # 환경 구성 생성 (env_cfg는 setup_teleop_device에서 접근할 수 있도록 전역 변수로 유지)
    global env_cfg
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # 환경 인스턴스 생성
    env = create_environment(env_cfg)

    # 시뮬레이션 루프 실행
    current_recorded_demo_count = run_simulation_loop(env, None, rate_limiter, success_term)

    # 정리
    env.close()
    print(f"기록 세션 종료됨. 총 {current_recorded_demo_count}개의 성공 데모 수집.")
    print(f"데모 저장 경로: {args_cli.dataset_file}")


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이터 앱 종료
    simulation_app.close()