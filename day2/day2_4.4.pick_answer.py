"""
집기(Pick)

Step 1~3(관측→검출→파지예측)로 얻은 grasp를 실제로 '집어 드는' 단계.
day2의 4.6에서 만든 pick&lift 상태머신을 그대로 가져와, '알던 물체 자세' 대신
'예측된 grasp_pose'를 입력으로 씀.

state 흐름 (3.6과 같은 결):
  REST → PREDICT(관측/예측) → READY → PREGRASP → GRASP → CLOSE → LIFT(들어올린 채 유지)

statemachine의 PREGRASP / GRASP / CLOSE / LIFT 4개 state(3.6에서 해본 그 로직)
  각 상태는 (1) 목표 self.des_ee_pose[i] (2) self.des_gripper_state[i]
  (3) 도달+대기 시 다음 상태로 전환 을 포함.
"""

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Day3 Step4 - pick")
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

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from utils.vision import depth2pc, get_random_color

from day2.task.lift.config.day2_4_ik_abs_env_cfg import FrankaYCBPickPlaceEnvCfg


# ============================================================================
# 3. Pick 상태머신 (4.6의 PickAndLiftSm을 grasp 예측 입력에 맞게 사용)
# ============================================================================
class GripperState:
    OPEN = 1.0
    CLOSE = -1.0


class PickSmState:
    REST = 0
    PREDICT = 1
    READY = 2
    PREGRASP = 3
    GRASP = 4
    CLOSE = 5
    LIFT = 6


class PickSmWaitTime:
    REST = 1.0
    PREDICT = 0.0
    READY = 0.5
    PREGRASP = 1.0
    GRASP = 0.5
    CLOSE = 1.0
    LIFT = 0.5
    TIMEOUT = 3.0


class PickAndLiftSm:
    """물체를 집어 들어올리는 상태머신 (놓기 없음). 각 state가 목표 ee_pose와 그리퍼 상태를 정한다."""

    def __init__(self, dt, num_envs, device="cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.stall_threshold = 0.002

        self.sm_state = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)

        self.des_ee_pose = torch.zeros((num_envs, 7), device=device)
        self.des_gripper_state = torch.full((num_envs, 1), 0.0, device=device)

        # 관측 자세(비스듬히 내려다봄) — 4.6엔 없던, 카메라로 보기 위한 자세
        VIEW_TILT_DEG = -20.0
        half = np.deg2rad(VIEW_TILT_DEG) / 2.0
        self.capture_pose = torch.tensor(
            [[0.20, -0.05, 0.60, 0.0, np.cos(half), 0.0, -np.sin(half)]], device=device, dtype=torch.float32
        ).repeat(num_envs, 1)
        self.ready_pose = torch.tensor(
            [[0.30, -0.05, 0.60, 0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32
        ).repeat(num_envs, 1)

        # perception으로 채워지는 grasp 목표
        self.grasp_pose = torch.zeros((num_envs, 7), device=device)
        self.pregrasp_pose = torch.zeros((num_envs, 7), device=device)
        self.stack_ee_pose = []

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose, grasp_pose, pregrasp_pose):
        ee_pos = ee_pose[:, :3]
        ee_pos[:, 2] -= 0.5  # table 높이 보정 (4.6과 동일)

        for i in range(self.num_envs):
            state = self.sm_state[i]

            if state == PickSmState.REST:
                # 관측 자세로 이동, 그리퍼 열기 → 도달+대기 후 PREDICT
                self.des_ee_pose[i] = self.capture_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold \
                        or self.sm_wait_time[i] > PickSmWaitTime.TIMEOUT:
                    if self.sm_wait_time[i] >= PickSmWaitTime.REST:
                        self.sm_state[i] = PickSmState.PREDICT
                        self.sm_wait_time[i] = 0.0

            elif state == PickSmState.PREDICT:
                # 관측 자세 유지 (main이 이 state에서 SAM3+GraspNet 수행) → READY
                self.des_ee_pose[i] = self.capture_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickSmWaitTime.PREDICT:
                        self.sm_state[i] = PickSmState.READY
                        self.sm_wait_time[i] = 0.0

            elif state == PickSmState.READY:
                # 정면 준비 자세로 이동 → PREGRASP
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickSmWaitTime.READY:
                        self.sm_state[i] = PickSmState.PREGRASP
                        self.sm_wait_time[i] = 0.0

            elif state == PickSmState.PREGRASP:
                # 파지 직전 자세(pregrasp)로 이동, 그리퍼 열기 → GRASP
                self.des_ee_pose[i] = pregrasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                self.stack_ee_pose.append(ee_pos[i])
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickSmWaitTime.PREGRASP:
                        self.sm_state[i] = PickSmState.GRASP
                        self.sm_wait_time[i] = 0.0
                elif len(self.stack_ee_pose) > 50 and \
                        torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                    # 더 이상 움직이지 못하면(스톨) 넘어감
                    self.sm_state[i] = PickSmState.GRASP
                    self.sm_wait_time[i] = 0.0
                    self.stack_ee_pose = []

            elif state == PickSmState.GRASP:
                # 예측된 grasp 자세로 접근, 그리퍼는 아직 열림 → CLOSE
                self.des_ee_pose[i] = grasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                self.stack_ee_pose.append(ee_pos[i])
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickSmWaitTime.GRASP:
                        self.sm_state[i] = PickSmState.CLOSE
                        self.sm_wait_time[i] = 0.0
                        self.stack_ee_pose = []
                elif len(self.stack_ee_pose) > 100 and \
                        torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.stall_threshold:
                    self.sm_state[i] = PickSmState.CLOSE
                    self.sm_wait_time[i] = 0.0
                    self.stack_ee_pose = []

            elif state == PickSmState.CLOSE:
                # 현재 자세 유지 + 그리퍼 닫아 물체 집기 → LIFT (들어올릴 목표 설정)
                self.des_ee_pose[i] = ee_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if self.sm_wait_time[i] >= PickSmWaitTime.CLOSE:
                    self.sm_state[i] = PickSmState.LIFT
                    self.sm_wait_time[i] = 0.0
                    self.lift_pose = grasp_pose[i].clone()
                    self.lift_pose[2] = self.lift_pose[2] + 0.4  # 0.4m 위로 들어올림

            elif state == PickSmState.LIFT:
                # 들어올린 자세 유지 (Step 4는 여기서 끝 — 놓기는 Step 5에서)
                self.des_ee_pose[i] = self.lift_pose
                self.des_gripper_state[i] = GripperState.CLOSE

            self.sm_wait_time[i] += self.dt
            actions = torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)
        return actions


# ============================================================================
# 4. 메인 함수
# ============================================================================
def main():
    num_envs = 1
    env_num = 0

    env_cfg = FrankaYCBPickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    device = env.scene.device

    # 모델 로드
    sam3_model = build_sam3_image_model(
        checkpoint_path='data/checkpoint/sam3/sam3.1_multiplex.pt',
        load_from_HF=False,
    )
    sam3_processor = Sam3Processor(sam3_model)

    DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grasp_model_config = cfg_from_yaml_file(os.path.join(DIR_PATH, 'cgnet/configs/config.yaml'))
    grasp_model = builder.model_builder(grasp_model_config.model)
    builder.load_model(grasp_model, os.path.join(DIR_PATH, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth'))
    grasp_model.to(device)
    grasp_model.eval()
    print("[INFO]: Setup complete...")

    # 상태머신
    pick_sm = PickAndLiftSm(dt=env_cfg.sim.dt * env_cfg.decimation, num_envs=num_envs, device=device, position_threshold=0.01)

    robot_camera = env.scene.sensors["camera"]
    ee_frame = env.scene["ee_frame"]
    K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()

    grasp_marker_cfg = FRAME_MARKER_CFG.copy()
    grasp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    grasp_marker_cfg.prim_path = "/Visuals/GraspPoseTarget"
    grasp_marker = VisualizationMarkers(grasp_marker_cfg)

    predicted = False

    while simulation_app.is_running():
        with torch.inference_mode():
            # -- PREDICT state에서 한 번만 SAM3 + GraspNet 수행 (Step 1~3 내용) --
            if pick_sm.sm_state[env_num] == PickSmState.PREDICT and not predicted:
                for _ in range(2):
                    env.sim.render()
                robot_camera.update(dt=0.0, force_recompute=True)
                img_np = robot_camera.data.output["rgb"][env_num].squeeze().detach().cpu().numpy()
                depth_np = robot_camera.data.output["distance_to_image_plane"][env_num].squeeze().detach().cpu().numpy()
                img_rgb = np.array(img_np)[..., :3].astype(np.uint8)

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

                # -- SAM3 결과 시각화 저장 (마스크 + bbox 오버레이) --
                vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
                mask_overlay = vis_img.copy()
                for i in range(len(pred_scores)):
                    color = (255, 0, 0) if i == best_idx else get_random_color()  # 타겟=파랑(BGR)
                    mask_overlay[pred_masks[i, 0] > 0.5] = color
                    x1, y1, x2, y2 = map(int, pred_boxes[i])
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3 if i == best_idx else 1)
                    cv2.putText(vis_img, f"{input_prompt}: {pred_scores[i]:.2f}", (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imwrite("data/SAM3_result.png", cv2.addWeighted(mask_overlay, 0.3, vis_img, 0.7, 0))
                print("[INFO] Saved SAM3 result → data/SAM3_result.png")

                pc, pc_colors = depth2pc(depth_np, K, rgb=np.array(img_np)[..., :3])
                pc_obj_mask = (pred_masks[best_idx, 0] > 0.5)[depth_np > 0]
                robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
                robot_entity_cfg.resolve(env.scene)
                hand_pose_w = env.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], :]
                if pc is None:
                    continue
                rot_ee, trans_ee, width = inference_cgnet(
                    pc, grasp_model, device, hand_pose_w, env, object_mask=pc_obj_mask, pc_colors=pc_colors, vis=True
                )
                grasp_rot, grasp_pos = rot_ee, trans_ee
                grasp_rot_flip = grasp_rot @ np.diag([-1.0, -1.0, 1.0])
                hand_rot_ref = R.from_quat(hand_pose_w[0, 3:7].cpu().numpy(), scalar_first=True).as_matrix()
                if np.trace(grasp_rot_flip.T @ hand_rot_ref) > np.trace(grasp_rot.T @ hand_rot_ref):
                    grasp_rot = grasp_rot_flip
                q = R.from_matrix(grasp_rot).as_quat()
                grasp_quat = np.array([q[3], q[0], q[1], q[2]])
                z_axis = grasp_rot[:, 2]
                grasp_pos = grasp_pos + z_axis * 0.03
                pregrasp_pos = grasp_pos - z_axis * 0.1
                pick_sm.grasp_pose[env_num] = torch.tensor(np.concatenate([grasp_pos, grasp_quat]), device=device, dtype=torch.float32)
                pick_sm.pregrasp_pose[env_num] = torch.tensor(np.concatenate([pregrasp_pos, grasp_quat]), device=device, dtype=torch.float32)
                print("[INFO] grasp predicted, start picking.")
                predicted = True

            # -- 상태머신으로 액션 계산 --
            tcp_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
            actions = pick_sm.compute(ee_pose=ee_pose, grasp_pose=pick_sm.grasp_pose, pregrasp_pose=pick_sm.pregrasp_pose)

            # 예측 grasp 마커 시각화
            gp = pick_sm.grasp_pose
            grasp_marker.visualize(translations=gp[:, :3] + env.scene["robot"].data.root_state_w[:, :3], orientations=gp[:, 3:7])

            obs, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
