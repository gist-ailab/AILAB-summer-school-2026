"""
파지 예측(Grasp Prediction)

Step 2에서 찾은 타겟 물체를, Contact-GraspNet으로 '어떻게 잡을지(6-DoF grasp)' 예측한다.
(3.6은 물체 자세 + 고정 offset으로 잡았지만, 여기서는 point cloud로부터 파지를 예측한다.)

0. Step 1~2: 관측 → SAM3 검출
1. Depth → point cloud 생성, 타겟 마스크를 점 단위로 변환
2. Contact-GraspNet 추론으로 파지 회전/위치 예측
3. Isaac Lab 형식(quaternion)으로 변환 + approach offset/pregrasp 계산
4. 예측 파지를 GraspPoseTarget 마커로 시각화 (아직 잡지는 않음)
"""

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Day3 Step3 - grasp prediction")
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
from scipy.spatial.transform import Rotation as R

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# 손목 카메라 렌더링 활성화 (--enable_cameras flag 대체)
import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# cgnet/sam3 폴더 접근을 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Contact-GraspNet
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 시각화/포인트클라우드 유틸
from utils.vision import depth2pc, get_random_color

# IK 기반 Franka + YCB pick-place 환경 config
from day2.task.lift.config.day2_4_ik_abs_env_cfg import FrankaYCBPickPlaceEnvCfg


# ============================================================================
# 3. 메인 함수
# ============================================================================
def main():
    """메인 함수 - 관측 + SAM3 검출 + GraspNet 파지 예측"""
    num_envs = 1
    env_num = 0

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

    # SAM3 모델 로드
    sam3_model = build_sam3_image_model(
        checkpoint_path='data/checkpoint/sam3/sam3.1_multiplex.pt',
        load_from_HF=False,
    )
    sam3_processor = Sam3Processor(sam3_model)


    # Contact-GraspNet 모델 로드
    DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grasp_model_config = cfg_from_yaml_file(os.path.join(DIR_PATH, 'cgnet/configs/config.yaml'))
    grasp_model = builder.model_builder(grasp_model_config.model)
    builder.load_model(grasp_model, os.path.join(DIR_PATH, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth'))
    grasp_model.to(device)
    grasp_model.eval()
    print("[INFO]: Setup complete...")

    robot_camera = env.scene.sensors["camera"]
    ee_frame = env.scene["ee_frame"]
    position_threshold = 0.01
    K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()  # 카메라 intrinsic

    # 예측 파지를 표시할 좌표축 마커
    grasp_marker_cfg = FRAME_MARKER_CFG.copy()
    grasp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    grasp_marker_cfg.prim_path = "/Visuals/GraspPoseTarget"
    grasp_marker = VisualizationMarkers(grasp_marker_cfg)

    # 관측 자세 (Step 1과 동일)
    VIEW_TILT_DEG = -20.0
    half = np.deg2rad(VIEW_TILT_DEG) / 2.0
    observe_pose = torch.tensor(
        [[0.20, -0.05, 0.60, 0.0, np.cos(half), 0.0, -np.sin(half)]],
        device=device, dtype=torch.float32,
    )
    gripper_open = torch.full((num_envs, 1), 1.0, device=device)

    os.makedirs("data", exist_ok=True)
    predicted = False
    grasp_pose_vis = None  # 예측된 grasp pose (x,y,z,qw,qx,qy,qz), 시각화용

    # ========================================================================
    # 4. 시뮬레이션 메인 루프
    # ========================================================================
    while simulation_app.is_running():
        with torch.inference_mode():
            # 관측 자세 유지 액션
            actions = torch.cat([observe_pose, gripper_open], dim=-1)
            obs, _ = env.step(actions)

            tcp = ee_frame.data.target_pos_w[..., 0, :] - env.scene.env_origins
            tcp = tcp[0].clone()
            tcp[2] -= 0.5
            reached = torch.linalg.norm(tcp - observe_pose[0, :3]) < position_threshold

            if reached and not predicted:
                # -- 이미지 취득 --
                for _ in range(2):
                    env.sim.render()
                robot_camera.update(dt=0.0, force_recompute=True)
                img_np = robot_camera.data.output["rgb"][env_num].squeeze().detach().cpu().numpy()
                depth_np = robot_camera.data.output["distance_to_image_plane"][env_num].squeeze().detach().cpu().numpy()
                img_rgb = np.array(img_np)[..., :3].astype(np.uint8)

                # -- SAM3 검출 (Step 2) --
                img_pil = Image.fromarray(img_rgb)
                img_pil.save("data/SAM3_input_image.png")
                print("[INFO] Saved SAM3 input image → data/SAM3_input_image.png")
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inference_state = sam3_processor.set_image(img_pil)
                    print("잡을 물체에 대한 텍스트를 영어로 입력하세요... ")
                    input_prompt = sys.stdin.readline().strip()
                    output = sam3_processor.set_text_prompt(state=inference_state, prompt=input_prompt)
                pred_scores = output["scores"].float().cpu().numpy()
                pred_boxes = output["boxes"].float().cpu().numpy()
                pred_masks = output["masks"].float().cpu().numpy()
                if len(pred_scores) == 0:
                    print("[WARN] No object detected. Try another text.")
                    continue
                best_idx = int(np.argmax(pred_scores))

                # -- SAM3 결과 시각화 저장 (Step 2와 동일: 마스크 + bbox 오버레이) --
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

                ############## TODO: Depth → point cloud + 타겟 마스크 변환 ##############
                # 1) depth_np와 카메라 intrinsic K로 point cloud 생성 (RGB 컬러도 함께)
                #    힌트: pc, pc_colors = depth2pc(depth_np, K, rgb=<RGB(H,W,3)>)
                # 2) 타겟(best_idx) 마스크를 point cloud 점 단위 마스크로 변환
                #    - obj_mask_img = pred_masks[best_idx, 0] > 0.5   # (H, W) bool
                #    - valid_depth  = depth_np > 0                    # depth2pc가 쓰는 유효 픽셀
                #    - pc_obj_mask  = obj_mask_img[valid_depth]       # (num_points,)
                rgb_for_pc = np.array(img_np)[..., :3]
                pc, pc_colors =
                obj_mask_img =
                valid_depth =
                pc_obj_mask =
                ######################################################################

                # 로봇 손(hand) 자세 (inference_cgnet의 손목 꺾임 회피에 사용)
                robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
                robot_entity_cfg.resolve(env.scene)
                hand_body_id = robot_entity_cfg.body_ids[0]
                hand_pose_w = env.scene["robot"].data.body_state_w[:, hand_body_id, :]

                if pc is None:
                    print("[WARN] Empty point cloud. Try again.")
                    continue

                ############## TODO: Contact-GraspNet 추론 ##############
                # inference_cgnet(pc, grasp_model, device, hand_pose_w, env,
                #                 object_mask=pc_obj_mask, pc_colors=pc_colors)
                # 반환: rot_ee(3x3 회전행렬), trans_ee(위치), width(그리퍼 폭)
                rot_ee, trans_ee, width =
                #####################################################
                print("[INFO] grasp predicted.")

                grasp_rot = rot_ee
                grasp_pos = trans_ee

                # approach축 180° 대칭 해소 (손목 꺾임/self-collision 회피) — 제공됨
                grasp_rot_flip = grasp_rot @ np.diag([-1.0, -1.0, 1.0])
                hand_rot_ref = R.from_quat(hand_pose_w[0, 3:7].cpu().numpy(), scalar_first=True).as_matrix()
                if np.trace(grasp_rot_flip.T @ hand_rot_ref) > np.trace(grasp_rot.T @ hand_rot_ref):
                    grasp_rot = grasp_rot_flip

                ############## TODO: 회전행렬 → 쿼터니언(wxyz) + offset/pregrasp ##############
                # 1) grasp_rot(3x3) → 쿼터니언. scipy는 (x,y,z,w) 순서 → Isaac은 (w,x,y,z)로 재배열
                #    힌트: q = R.from_matrix(grasp_rot).as_quat();  q = [q[3], q[0], q[1], q[2]]
                # 2) approach 방향(z축=grasp_rot[:, 2])으로 grasp 깊이 offset(0.03) 적용
                # 3) pregrasp는 grasp에서 approach 반대로 0.1m 뒤
                grasp_quat =
                grasp_quat =
                APPROACH_OFFSET = 0.03
                z_axis = grasp_rot[:, 2]
                grasp_pos =
                pregrasp_pos =
                #######################################################################

                grasp_pose_vis = torch.tensor(
                    np.concatenate([grasp_pos, grasp_quat]), device=device, dtype=torch.float32
                ).unsqueeze(0)  # (1, 7)
                print(f"[INFO] grasp_pose (base): {grasp_pose_vis[0].cpu().numpy()}")
                print(f"[INFO] pregrasp_pos (base): {pregrasp_pos}")
                predicted = True

            # 예측된 파지를 월드 좌표 마커로 표시 (base → world = root_pos + grasp_xyz)
            if grasp_pose_vis is not None:
                root_pos_w = env.scene["robot"].data.root_state_w[:, :3]
                grasp_marker.visualize(
                    translations=grasp_pose_vis[:, :3] + root_pos_w, orientations=grasp_pose_vis[:, 3:7]
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
