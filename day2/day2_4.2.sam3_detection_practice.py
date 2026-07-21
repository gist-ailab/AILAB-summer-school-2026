"""
SAM3 Detection

Step 1에서 얻은 관측 이미지에서, 사용자가 입력한 텍스트 프롬프트로 '어떤 물체를 잡을지'를 SAM3로 검출.
(3.6은 물체 위치를 이미 알았지만, 여기서는 '어디에 무엇이 있는지'를 찾아야 함.)

1. Step 1처럼 로봇을 관측 자세로 보내고 RGB/Depth 취득
2. SAM3에 텍스트 프롬프트를 주어 타겟 물체를 검출 (mask/bbox/score)
3. 가장 신뢰도 높은 인스턴스를 타겟으로 선택하고 결과를 시각화 저장

완성 후 실행하면 프롬프트 입력 시 data/SAM3_result.png 에 검출 결과가 저장.
"""

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Day3 Step2 - SAM3 detection")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 / 모델 라이브러리 임포트
# ============================================================================
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

from isaaclab.envs import ManagerBasedEnv

# 손목 카메라 렌더링 활성화 (--enable_cameras flag 대체)
import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# cgnet/sam3 폴더 접근을 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SAM3 image model 및 processor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 시각화 유틸(랜덤 색상)
from utils.vision import get_random_color

# IK 기반 Franka + YCB pick-place 환경 config
from day2.task.lift.config.day2_4_ik_abs_env_cfg import FrankaYCBPickPlaceEnvCfg


# ============================================================================
# 3. 메인 함수
# ============================================================================
def main():
    """메인 함수 - 관측 + SAM3 검출"""
    num_envs = 1

    # 환경 생성
    env_cfg = FrankaYCBPickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    device = env.scene.device
    print(f"Environment reset. Number of environments: {env.num_envs}")

    # SAM3 모델 로드 (텍스트 프롬프트 기반 물체 검출)
    sam3_model = build_sam3_image_model(
        checkpoint_path='data/checkpoint/sam3/sam3.1_multiplex.pt',
        load_from_HF=False,
    )
    sam3_processor = Sam3Processor(sam3_model)

    print("[INFO]: SAM3 setup complete...")

    robot_camera = env.scene.sensors["camera"]
    ee_frame = env.scene["ee_frame"]
    position_threshold = 0.01

    # 관측 자세 (Step 1과 동일)
    VIEW_TILT_DEG = -20.0
    half = np.deg2rad(VIEW_TILT_DEG) / 2.0
    observe_pose = torch.tensor(
        [[0.20, -0.05, 0.60, 0.0, np.cos(half), 0.0, -np.sin(half)]],
        device=device, dtype=torch.float32,
    )
    gripper_open = torch.full((num_envs, 1), 1.0, device=device)

    os.makedirs("data", exist_ok=True)
    # ========================================================================
    # 4. 시뮬레이션 메인 루프
    # ========================================================================
    while simulation_app.is_running():
        with torch.inference_mode():
            # 관측 자세 유지 액션 (Step 1에서 구현한 부분)
            actions = torch.cat([observe_pose, gripper_open], dim=-1)
            obs, _ = env.step(actions)

            # 관측 자세 도달 판정 (Step 1과 동일)
            tcp = ee_frame.data.target_pos_w[..., 0, :] - env.scene.env_origins
            tcp = tcp[0].clone()
            tcp[2] -= 0.5
            reached = torch.linalg.norm(tcp - observe_pose[0, :3]) < position_threshold

            if reached:
                # -- 카메라 이미지 취득 (Step 1) --
                for _ in range(2):
                    env.sim.render()
                robot_camera.update(dt=0.0, force_recompute=True)
                img_np = robot_camera.data.output["rgb"][0].squeeze().detach().cpu().numpy()
                img_rgb = np.array(img_np)[..., :3].astype(np.uint8)

                print("Running SAM3 inference...")
                # PIL 이미지로 변환 (Sam3Processor.set_image는 CHW를 가정하므로 PIL로 넘긴다)
                img_pil = Image.fromarray(img_rgb)
                img_pil.save("data/SAM3_input_image.png")
                print("[INFO] Saved SAM3 input image → data/SAM3_input_image.png")

                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ############## TODO: SAM3 추론 ##############
                    # 1) 이미지를 SAM3에 등록: inference_state = sam3_processor.set_image(<PIL 이미지>)
                    # 2) 사용자 텍스트 프롬프트 입력받기 (아래 두 줄은 제공됨)
                    # 3) 텍스트 프롬프트로 검출: output = sam3_processor.set_text_prompt(state=..., prompt=...)
                    inference_state =
                    print("잡을 물체에 대한 텍스트를 영어로 입력하세요... ")
                    input_prompt = sys.stdin.readline().strip()
                    output =
                    ############################################

                # 검출 결과 취득
                pred_scores = output["scores"].float().cpu().numpy()   # (N,)
                pred_boxes = output["boxes"].float().cpu().numpy()     # (N, 4) xyxy
                pred_masks = output["masks"].float().cpu().numpy()     # (N, 1, H, W)
                print(f"Found {len(pred_scores)} object(s) for prompt '{input_prompt}'.")

                # 검출 결과가 없으면 다시 입력받도록 스킵
                if len(pred_scores) == 0:
                    print("[WARN] No object detected. Try another text.")
                    continue

                ############## TODO: 가장 신뢰도 높은 인스턴스 선택 ##############
                # pred_scores가 가장 큰 인덱스를 정수로. 힌트: int(np.argmax(...))
                best_idx =
                ############################################################
                print(f"[INFO] Selected target (score: {pred_scores[best_idx]:.2f}), bbox: {pred_boxes[best_idx]}")

                # -- 결과 시각화 (마스크 + bbox 오버레이) --
                vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
                mask_overlay = vis_img.copy()
                for i in range(len(pred_scores)):
                    color = (255, 0, 0) if i == best_idx else get_random_color()  # 타겟=파랑(BGR)
                    mask_overlay[pred_masks[i, 0] > 0.5] = color
                    x1, y1, x2, y2 = map(int, pred_boxes[i])
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3 if i == best_idx else 1)
                    cv2.putText(vis_img, f"{input_prompt}: {pred_scores[i]:.2f}", (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                final_result = cv2.addWeighted(mask_overlay, 0.3, vis_img, 0.7, 0)
                cv2.imwrite("data/SAM3_result.png", final_result)
                print("[INFO] Saved SAM3 result → data/SAM3_result.png")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
