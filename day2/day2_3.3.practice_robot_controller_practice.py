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
"""Rest everything else."""
import torch

from isaaclab.envs import ManagerBasedEnv

# 커스텀 Franka 큐브 리프팅 환경 config (gym 등록 없이 ManagerBasedEnv로 직접 사용)
from task.lift.config.day2_3_ik_abs_env_cfg_practice import FrankaCubeLiftEnvCfg

def main():
    """
    메인 함수 - 시뮬레이션의 전체 파이프라인 관리
    
    1. 환경 설정 파싱
    2. 환경 생성 및 초기화
    3. 액션 버퍼 생성
    4. 시뮬레이션 루프 실행
    """
    # ============================================================================
    # 4. 환경 설정 및 초기화
    # ============================================================================
    # 환경 개수 (학습 시에는 더 많은 환경을 사용할 수 있음)
    num_envs = 1

    # parse configuration
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    
    # 환경 생성 - 실제 시뮬레이션 환경을 생성
    env = ManagerBasedEnv(cfg=env_cfg)
    
    # 환경 리셋 - 초기 상태로 환경을 설정
    env.reset()
    
    # 환경 정보 출력
    print(f"Environment reset. Number of environments: {env.num_envs}")
    
    # ============================================================================
    # 5. 액션 버퍼 생성 및 초기화
    # ============================================================================
    # IK 절대 위치 타겟을 위한 액션 버퍼 생성 (위치 + 쿼터니언 + 그리퍼)
    # 형태: (num_envs, 8) - 3개 위치 + 4개 쿼터니언 + 1개 그리퍼
    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros((num_envs, action_dim), device=env.device)
    

    ############ TODO: actions 값 바꿔서 IK 를 통한 position 제어 실습해보기 ############ 
    # 초기 유효한 쿼터니언 설정 (w=0.707, x=0.0, y=0.707, z=0.0)
    # 이는 90도 회전을 나타내는 쿼터니언
    actions[:, 3] = 0.7071068  # w 성분
    actions[:, 4] = 0.0        # x 성분  
    actions[:, 5] = 0.7071068  # y 성분
    actions[:, 6] = 0.0        # z 성분
    
    # 초기 위치 설정 (x=0.4, y=0.0, z=0.7)
    actions[:, 0] = 0.4  # x 위치
    actions[:, 1] = 0.0  # y 위치
    actions[:, 2] = 0.7  # z 위치
    
    # 그리퍼 액션 설정 (7번 인덱스)
    actions[:, 7] = 1.0  # 그리퍼 열기 (1.0) 또는 닫기 (0.0)
    ##############################################################################

    # ============================================================================
    # 6. 시뮬레이션 메인 루프
    # ============================================================================
    print("Starting simulation loop...")
    
    # 시뮬레이션 앱이 실행 중인 동안 반복
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # 환경에 액션을 적용 (ManagerBasedEnv.step -> (obs, extras) 2-튜플)
            obs, _ = env.step(actions)
            

    # ============================================================================
    # 7. 정리 작업
    # ============================================================================
    # 환경 종료
    env.close()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()