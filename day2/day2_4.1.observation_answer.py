"""
관측(Observation)

day2의 3.6에서는 '물체 위치를 이미 아는' 상태로 집어 들었다.
day2_4에서는 물체를 모른다고 가정하고, 먼저 로봇 손목 카메라로 '보는' 단계를 만든다.
(3.6에서 만든 pick&lift statemachine에 환경 관측 정보를 연결하는 첫 단계)

1. Franka + YCB 물체 + 바구니 환경 생성
2. 로봇을 고정된 관측 자세(비스듬히 내려다보는 자세)로 IK 이동
3. 관측 자세 도달 시 손목 카메라에서 RGB/Depth 취득 후 저장

실행하면 로봇이 관측 자세로 이동하고 data/observation_rgb.png 가 저장된다.
"""

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Day2.4 - robot observation")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 관련 라이브러리 임포트
# ============================================================================
import os
import sys
import numpy as np
import torch
import cv2

from isaaclab.envs import ManagerBasedEnv

# 손목 카메라 렌더링 활성화 (--enable_cameras flag 대체)
import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# day2 패키지 접근을 위한 경로 추가 (repo root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IK 기반 Franka + YCB pick-place 환경 config
from day2.task.lift.config.day2_4_ik_abs_env_cfg import FrankaYCBPickPlaceEnvCfg


# ============================================================================
# 3. 메인 함수
# ============================================================================
def main():
    """메인 함수 - 관측 파이프라인의 진입점"""
    # 환경 개수(1개로 고정)
    num_envs = 1

    # 환경 설정 생성 및 환경 생성
    env_cfg = FrankaYCBPickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    print(f"Environment reset. Number of environments: {env.num_envs}")

    # 전체 씬을 바라보는 뷰포트 카메라 시점
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    device = env.scene.device

    # 손목(handeye) 카메라 센서와 엔드이펙터 프레임 센서
    robot_camera = env.scene.sensors["camera"]
    ee_frame = env.scene["ee_frame"]
    position_threshold = 0.01  # 관측 자세 도달 판정 임계값 (4.6과 동일한 개념)

    # ------------------------------------------------------------------------
    # 관측 자세 (로봇 base 프레임): 물체를 비스듬히 내려다보도록 base y축 기준 tilt
    # (top-down으로 보면 컵 등이 평면 원으로 보여 검출이 어려우므로 기울여서 관측)
    #  - position: (x, y, z)
    #  - quaternion: (0,1,0,0)[수직]에 y축 tilt를 곱한 (0, cos(θ/2), 0, -sin(θ/2))
    # ------------------------------------------------------------------------
    VIEW_TILT_DEG = -20.0
    half = np.deg2rad(VIEW_TILT_DEG) / 2.0
    observe_pose = torch.tensor(
        [[0.20, -0.05, 0.60, 0.0, np.cos(half), 0.0, -np.sin(half)]],
        device=device, dtype=torch.float32,
    )  # (1, 7) = 위치(3) + 쿼터니언 wxyz(4)
    gripper_open = torch.full((num_envs, 1), 1.0, device=device)  # 그리퍼 열림(1.0)

    os.makedirs("data", exist_ok=True)
    saved = False

    # ========================================================================
    # 4. 시뮬레이션 메인 루프
    # ========================================================================
    while simulation_app.is_running():
        with torch.inference_mode():
            # -- 관측 자세를 유지하는 액션 구성: [ee_pose(7) + gripper(1)] = (num_envs, 8) --
            actions = torch.cat([observe_pose, gripper_open], dim=-1)

            # 환경 스텝 (ManagerBasedEnv.step → (obs, extras) 2-튜플)
            obs, _ = env.step(actions)

            # 현재 손끝(TCP) 위치 (env-local, table 높이 0.5 보정) — 4.6과 동일한 도달 판정
            tcp = ee_frame.data.target_pos_w[..., 0, :] - env.scene.env_origins
            tcp = tcp[0].clone()
            tcp[2] -= 0.5
            reached = torch.linalg.norm(tcp - observe_pose[0, :3]) < position_threshold

            # -- 관측 자세에 도달하면 카메라에서 RGB/Depth를 취득해 한 번 저장 --
            if reached and not saved:
                # 현재 물리 상태로 카메라를 다시 렌더링해 시점을 동기화
                for _ in range(2):
                    env.sim.render()
                robot_camera.update(dt=0.0, force_recompute=True)

                # 손목 카메라 출력에서 RGB / Depth 취득
                rgb = robot_camera.data.output["rgb"][0].detach().cpu().numpy()
                depth = robot_camera.data.output["distance_to_image_plane"][0].squeeze().detach().cpu().numpy()

                # RGB 저장 (SAM3 입력으로 쓸 이미지) — cv2는 BGR 순서
                img_rgb = np.array(rgb)[..., :3].astype(np.uint8)
                cv2.imwrite("data/observation_rgb.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                print(
                    f"[INFO] Saved observation image → data/observation_rgb.png "
                    f"(RGB {img_rgb.shape}, depth {depth.shape}, "
                    f"depth min={np.nanmin(depth):.3f} max={np.nanmax(depth):.3f})"
                )
                saved = True

    # ========================================================================
    # 5. 정리 작업
    # ========================================================================
    env.close()


# ============================================================================
# 6. 프로그램 실행
# ============================================================================
if __name__ == "__main__":
    main()
    simulation_app.close()
