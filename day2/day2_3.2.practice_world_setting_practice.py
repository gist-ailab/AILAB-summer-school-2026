"""
Isaac Lab 환경에서 Franka 로봇을 사용한 큐브 리프팅 태스크 실행 스크립트

이 스크립트는 다음과 같은 기능을 제공합니다:
1. Isaac Lab 시뮬레이션 환경 초기화
2. Franka 로봇과 큐브가 포함된 리프팅 환경 생성
3. 로봇의 역기구학(IK) 기반 동작 제어
4. 시뮬레이션 루프 실행 및 환경 관리
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트 및 명령행 인자 설정
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

# 명령행 인자 파서 설정 - 사용자가 다양한 옵션을 설정할 수 있도록 함
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# AppLauncher의 명령행 인자들을 파서에 추가 (headless, device, enable_cameras 등)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Isaac Lab 앱 초기화 - 시뮬레이션 환경을 시작
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 관련 라이브러리 임포트
# ============================================================================
import torch

from isaaclab.envs import ManagerBasedEnv

# 커스텀 Franka 큐브 리프팅 환경 config (gym 등록 없이 ManagerBasedEnv로 직접 사용)
from task.lift.config.day2_3_joint_pos_env_cfg_practice import FrankaCubeLiftEnvCfg

def main():
    """
    메인 함수 - 시뮬레이션의 전체 파이프라인 관리
    
    이 함수는 다음 단계로 구성됩니다:
    1. 환경 설정 파싱
    2. 환경 생성 및 초기화
    3. 액션 버퍼 생성
    4. 시뮬레이션 루프 실행
    """
    # ============================================================================
    # 3. 환경 설정 및 초기화
    # ============================================================================
    # 하드코딩된 환경 개수 (학습 시에는 더 많은 환경을 사용할 수 있음)
    num_envs = 1

    # 환경 config 생성 (gym 등록 없이 직접 구성)
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # 환경 생성 - ManagerBasedEnv 직접 생성
    env = ManagerBasedEnv(cfg=env_cfg)

    # 환경 리셋 - 초기 상태로 환경을 설정
    env.reset()

    # 환경 정보 출력
    print(f"Environment reset. Number of environments: {env.num_envs}")
    
    # ============================================================================
    # 4. 액션 버퍼 생성 및 초기화
    # ============================================================================
    # 액션 버퍼 생성 (팔 joint 7개+ 그리퍼)
    # 형태: (num_envs, 8) - 7개 joint+ 1개 그리퍼
    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros((num_envs, action_dim), device=env.device)
    
    # 초기 joint 각도 설정
    actions[:, 0] = 1.57  # 0번 joint
    actions[:, 1] = -1.57  # 1번 joint
    
    # # 그리퍼 액션 설정 (7번 인덱스)
    actions[:, 7] = -1.0  # 그리퍼 열기 (양수) 또는 닫기 (음수)

    # ============================================================================
    # 5. 시뮬레이션 메인 루프
    # ============================================================================
    print("Starting simulation loop...")

    # 시뮬레이션 앱이 실행 중인 동안 반복
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # 환경에 액션을 적용 (ManagerBasedEnv.step → (obs, extras) 2-튜플, 종료/보상 없음)
            obs, _ = env.step(actions)

    # ============================================================================
    # 6. 정리 작업
    # ============================================================================
    # 환경 종료
    env.close()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()