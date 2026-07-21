"""
놓기(Place) / 전체 통합
(SAM3 텍스트 검출 + Contact-GraspNet 파지 예측 + State Machine 제어)


Step 1~4에서 만든 관측·검출·파지예측·집기를 하나로 잇고,
물체를 바구니에 놓고 남은 물체까지 반복하는 최종 pick-and-place.

* statemachine 코드 부분이 많아 day2/utils/day2_4_statemachine_practice.py 파일로 분리
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트 및 명령행 인자 설정
# ============================================================================
import argparse
import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Isaac Lab 앱 런처
from isaaclab.app import AppLauncher

# Argparse로 CLI 인자 파싱 및 Omniverse 앱 실행
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
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
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# AILAB-summer-school-2026/cgnet 폴더에 접근하기 위한 시스템 파일 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Contact-GraspNet 모델 라이브러리 임포트
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet


# SAM3 image model 및 processor 로드
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 카메라 렌더링 옵션 --enable_cameras flag 를 대신하기 위함
import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)

# IK 기반 Franka 환경 config (RL이 아니므로 gym 등록 없이 직접 사용)
from day2.task.lift.config.day2_4_ik_abs_env_cfg import FrankaYCBPickPlaceEnvCfg

# main 에서 사용할 utils 함수들 임포트
from utils.vision import depth2pc, get_random_color
from day2.utils.day2_4_statemachine_practice import PickAndPlaceSm, PickAndPlaceSmState


# ============================================================================
# 3. 메인 함수
# ============================================================================
def main():
    """메인 함수 - 시뮬레이션의 진입점"""
    # 환경 갯수(1개로 고정)
    num_envs = 1

    # 환경 설정 생성 (RL이 아니므로 ManagerBasedEnvCfg 기반 config를 직접 구성)
    env_cfg = FrankaYCBPickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # 환경 생성 및 초기화 (ManagerBasedEnv 직접 생성, gym.make 미사용)
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    print(f"Environment reset. Number of environments: {env.num_envs}")

    # 환경 관측 카메라 시점 셋팅
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])

    # 환경 연산 디바이스(gpu)
    device = env.scene.device

    # SAM3 모델 로드 (텍스트 프롬프트 기반 물체 검출)
    sam3_model = build_sam3_image_model(
        checkpoint_path='data/checkpoint/sam3/sam3.1_multiplex.pt',
        load_from_HF=False,
    )
    sam3_processor = Sam3Processor(sam3_model)


    # Contact-GraspNet 모델 config를 불러오기 위한 경로 설정
    DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grasp_model_config_path = os.path.join(DIR_PATH, 'cgnet/configs/config.yaml')
    grasp_model_config = cfg_from_yaml_file(grasp_model_config_path)

    # Contact-GraspNet 모델 선언 및 checkpoint 입력을 통한 모델 weight 로드
    grasp_model = builder.model_builder(grasp_model_config.model)
    grasp_model_path = os.path.join(DIR_PATH, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth')
    builder.load_model(grasp_model, grasp_model_path)
    grasp_model.to(device)
    grasp_model.eval()

    print("[INFO]: Setup complete...")

    # 로봇 pick-and-place 제어를 위한 State machine 선언
    pick_and_place_sm = PickAndPlaceSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=num_envs,
        device=device,
        position_threshold=0.01
    )

    # 환경에서 robot handeye camera 변수 불러오기
    robot_camera = env.scene.sensors['camera']

    # 카메라 인트린식(intrinsics)
    K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()

    # 씬에 존재하는 YCB object 이름 목록 (object_0, object_1, ...)
    object_names = sorted([name for name in env.scene.rigid_objects.keys() if name.startswith("object_")])

    # 화면(SAM3 입력 이미지)에 표시할 YCB 물체 이름 리스트 로드
    with open("data/selected_ycb_objects.json") as f:
        ycb_names = json.load(f)

    # ---- 관측(observation) 자세를 joint 공간으로 고정 ----
    # 처음 한 번은 기울인 ready_pose(EE)로 IK 이동시키고, 그때 도달한 "깔끔한" arm joint 각도를
    # 저장한다. 이후 관측부터는 그 joint 각도를 직접 세팅해 카메라 시점을 결정적으로 고정한다
    # (IK 해의 모호성/손목 꺾임 회피). grasp 단계는 기존 IK를 그대로 사용.
    robot = env.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(["panda_joint[1-7]"])
    # reset 직후의 "기본 자세"를 캡처하면 안 된다. 첫 PREDICT(=REST에서 IK로 ready_pose에
    # 실제 도달한 시점)에 lazy 캡처하기 위해 None으로 시작한다.
    capture_joint_pos = None  # grasp 단계에서 캡처되는 grasp 자세의 joint 각도 (이후 재사용)
    ready_joint_pos = None   # 첫 관측 때 캡처되는 ready 자세의 joint 각도 (이후 재사용)


    # 명령하는 grasp_pose(목표)를 표시할 좌표축 마커 — 실제 손끝(ee_frame 마커)과 비교용
    grasp_marker_cfg = FRAME_MARKER_CFG.copy()
    grasp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    grasp_marker_cfg.prim_path = "/Visuals/GraspPoseTarget"
    grasp_marker = VisualizationMarkers(grasp_marker_cfg)

    env_num = 0  # 단일 env만 사용하므로 env_num=0 고정
    objects_in_bin = set()

    # ============================================================================
    # 4. 시뮬레이션 메인 루프
    # ============================================================================
    while simulation_app.is_running():
        # 모델 추론 상태 - 학습 연산 비활성화
        with torch.inference_mode():
            # 현재 state가 Predict일때, SAM3 -> GraspPrediction 순으로 추론 진행
            if pick_and_place_sm.sm_state[env_num] == PickAndPlaceSmState.REST:
                if capture_joint_pos is not None:
                    # READY state에서 관측 자세를 joint 각도로 고정
                    robot.write_joint_state_to_sim(
                        capture_joint_pos,
                        torch.zeros_like(capture_joint_pos),
                        joint_ids=arm_joint_ids,
                    )
                    robot.update(dt=0.0)

            elif pick_and_place_sm.sm_state[env_num] == PickAndPlaceSmState.PREDICT:
                # ---- 관측 자세를 joint 각도로 고정 ----
                if capture_joint_pos is None:
                    # 첫 관측: REST에서 IK로 ready_pose(EE)에 "실제 도달한" 현재 joint 각도를 저장.
                    # (reset 직후의 기본 자세가 아니라, 도달한 자세를 캡처해야 시점이 ready_pose가 된다.)
                    capture_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
                    print("[INFO] Captured ready joint configuration from first IK pose.")

                # [시간 동기화] 손이 READY에 멈춘 '현재' 물리 상태로 카메라를 새로 렌더링하여,
                # depth 이미지와 재구성에 쓰는 hand_pose_w의 시점을 일치시킨다.
                # (카메라는 update_period로 갱신되므로, 캡처 이미지가 과거(이동 중) 손 자세로
                #  찍혀 있으면 PC 재구성이 현재 hand_pose와 어긋나 offset이 생긴다.)
                for _ in range(2):
                    env.sim.render()
                robot_camera.update(dt=0.0, force_recompute=True)

                # 시각화를 위한 RGB 이미지 및 Depth 이미지 얻기
                image_ = robot_camera.data.output["rgb"][env_num]
                img_np = image_.squeeze().detach().cpu().numpy()
                depth = robot_camera.data.output["distance_to_image_plane"][env_num]
                depth_np = depth.squeeze().detach().cpu().numpy()

                ############################ SAM3 Model Inference ############################
                print("Running SAM3 inference...")

                img_rgb = np.array(img_np)[..., :3].astype(np.uint8)

                # save the RGB image for visualization
                vis_input = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
                x0, y0, dy = 12, 28, 24
                overlay = vis_input.copy()
                cv2.rectangle(overlay, (5, 5), (5 + 360, y0 + dy * (len(ycb_names) + 1)), (0, 0, 0), -1)
                vis_input = cv2.addWeighted(overlay, 0.45, vis_input, 0.55, 0)
                cv2.putText(vis_input, "YCB objects (green=in bin):", (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                for j, nm in enumerate(ycb_names):
                    # ycb_names[j] ↔ scene의 object_j 대응. objects_in_bin에 있으면 넣은 것.
                    in_bin = f"object_{j}" in objects_in_bin
                    color = (0, 255, 0) if in_bin else (255, 255, 255)   # 넣은 것=초록, 남은 것=흰색 (BGR)
                    tag = " (bin)" if in_bin else ""
                    cv2.putText(vis_input, f"- {nm}{tag}", (x0, y0 + dy * (j + 1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                cv2.imwrite("data/SAM3_input_image.png", vis_input)
                print("[INFO] Saved SAM3 input image to 'data/SAM3_input_image.png'")

                # PIL 이미지로 변환 (Sam3Processor.set_image는 ndarray 입력을 CHW로 가정하므로,
                #  HWC numpy를 그대로 넘기면 width가 채널 수(3)로 잘못 잡혀 마스크/박스 shape이 깨짐)
                img_pil = Image.fromarray(img_rgb)

                # SAM3 모델 추론 (사용자 텍스트 프롬프트로 타겟 물체를 직접 검출)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inference_state = sam3_processor.set_image(img_pil)
                    print("잡을 물체에 대한 텍스트를 영어로 입력하세요... ")
                    input_prompt = sys.stdin.readline().strip()
                    output = sam3_processor.set_text_prompt(state=inference_state, prompt=input_prompt)

                # 검출 결과 취득
                pred_scores = output["scores"].float().cpu().numpy()    # (N,)
                pred_boxes = output["boxes"].float().cpu().numpy()      # (N, 4) xyxy 픽셀 좌표
                pred_masks = output["masks"].float().cpu().numpy()      # (N, 1, H, W)

                print(f"Found {len(pred_scores)} object(s) for prompt '{input_prompt}'.")

                # 검출 결과가 없으면 grasp 추론을 건너뛰고 다시 입력을 받음
                if len(pred_scores) == 0:
                    print("[WARN] No object detected for the given prompt. Try another text.")
                    continue

                # 가장 신뢰도(score)가 높은 인스턴스를 타겟 물체로 선택
                best_idx = int(np.argmax(pred_scores))
                target_obj_bbox = pred_boxes[best_idx]    # [x_min, y_min, x_max, y_max]
                print(f"[INFO] Selected target object (score: {pred_scores[best_idx]:.2f}), bbox: {target_obj_bbox}")

                # ---- 결과 시각화 (마스크 + bbox 오버레이) ----
                vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
                mask_overlay = vis_img.copy()
                for i in range(len(pred_scores)):
                    # 선택된 타겟은 초록색, 그 외 후보는 랜덤 색상으로 표시
                    color = (255, 0, 0) if i == best_idx else get_random_color()

                    # 마스크 영역 색칠
                    binary_mask = pred_masks[i, 0] > 0.5
                    mask_overlay[binary_mask] = color

                    # 바운딩 박스 및 라벨 (타겟은 더 두껍게)
                    x1, y1, x2, y2 = map(int, pred_boxes[i])
                    thickness = 3 if i == best_idx else 1
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
                    label_text = f"{input_prompt}: {pred_scores[i]:.2f}"
                    cv2.putText(vis_img, label_text, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 마스크와 박스를 합성하여 결과 이미지 저장
                alpha = 0.3  # 마스크 투명도
                final_result = cv2.addWeighted(mask_overlay, alpha, vis_img, 1 - alpha, 0)
                os.makedirs("data", exist_ok=True)
                save_path = "data/SAM3_result.png"
                cv2.imwrite(save_path, final_result)
                print(f"[INFO] Saved SAM3 result to '{save_path}'")
                ####################################################################################

                ############################ Grasp Model Inference ##################################
                # 취득한 Depth 이미지를 통한 Point Cloud 생성 (RGB 컬러도 함께 추출)
                rgb_for_pc = np.array(img_np)[..., :3]   # (H, W, 3) RGB
                if num_envs > 1:
                    pc, pc_colors = depth2pc(depth_np, K[env_num], rgb=rgb_for_pc)
                else:
                    pc, pc_colors = depth2pc(depth_np, K, rgb=rgb_for_pc)

                # 타겟 물체의 SAM3 마스크를 point cloud 점 단위 마스크로 변환
                # (depth2pc는 depth>0 픽셀을 행 우선 순서로 pc에 담으므로 동일 마스킹으로 1:1 대응)
                obj_mask_img = pred_masks[best_idx, 0] > 0.5     # (H, W) bool, 타겟 물체 영역
                valid_depth = depth_np > 0                        # depth2pc가 사용하는 유효 픽셀
                pc_obj_mask = obj_mask_img[valid_depth]           # (num_points,) 각 pc 점의 타겟 소속 여부

                # Robot의 end-effector 위치 얻기
                robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
                robot_entity_cfg.resolve(env.scene)
                hand_body_id = robot_entity_cfg.body_ids[0]
                hand_pose_w = env.scene["robot"].data.body_state_w[:, hand_body_id, :]  # (num_envs, 13)

                if pc is not None:
                    # 전체 장면 point cloud는 그대로 두고(충돌검사/맥락 유지), 타겟 마스크를 함께 전달하여
                    # 바닥면/옆 물체에 생기는 파지를 inference_cgnet 내부에서 제거
                    # (inference_cgnet이 타겟 점(파랑)과 필터링 전/후 파지를 시각화함)
                    rot_ee, trans_ee, width = inference_cgnet(
                        pc, grasp_model, device, hand_pose_w, env, object_mask=pc_obj_mask, pc_colors=pc_colors
                    )
                    print("[INFO] Received ee coordinates from inference_cgnet")
                    print(f"[INFO] Gripper width: {width}")

                    # 예측한 파지점을 Isaaclab 형식으로 변환 (rotation matrix -> quat)
                    grasp_rot = rot_ee
                    grasp_pos = trans_ee

                    # approach축 180° 대칭 해소: 2지 그리퍼는 approach(z)축 180° 회전에 대해 동일한
                    # 물리적 파지다. 두 표현 중 '현재 손 자세에 더 가까운'(손목이 덜 꺾이는) 쪽을 선택해
                    # Franka 손목 한계 초과로 인한 self-collision을 회피한다.
                    # (z축=approach는 불변이므로 잡는 위치/접근 방향은 그대로 유지된다.)
                    grasp_rot_flip = grasp_rot @ np.diag([-1.0, -1.0, 1.0])
                    hand_rot_ref = R.from_quat(hand_pose_w[0, 3:7].cpu().numpy(), scalar_first=True).as_matrix()
                    # trace가 클수록 두 회전 사이 각도가 작음(=더 가까움)
                    align_orig = np.trace(grasp_rot.T @ hand_rot_ref)
                    align_flip = np.trace(grasp_rot_flip.T @ hand_rot_ref)
                    if align_flip > align_orig:
                        print("[INFO] approach축 180° 플립 적용 (손목 꺾임 최소화 → self-collision 회피)")
                        grasp_rot = grasp_rot_flip

                    grasp_quat = R.from_matrix(grasp_rot).as_quat()  # (x, y, z, w)
                    grasp_quat = np.array([grasp_quat[3], grasp_quat[0], grasp_quat[1], grasp_quat[2]]) # (w, x, y, z)

                    # rotation matrix를 사용하여 예측한 파지점의 offset 맞추기
                    # trans_ee는 이미 접점 근처(offset 0에서도 거의 닿음)이므로, 이 값은 손 길이(0.1034)가
                    # 아니라 "손끝을 파지점보다 더 밀어넣는 그립 깊이"다. graspnetAPI depth(0.02)에 맞춰 소량만.
                    # (GraspPoseTarget 마커가 물체 표면에 얹히도록 0.01씩 미세조정)
                    APPROACH_OFFSET = 0.03
                    z_axis = grasp_rot[:, 2]
                    grasp_pos = grasp_pos + z_axis * APPROACH_OFFSET
                    pregrasp_pos = grasp_pos - z_axis * 0.1

                    # 예측한 파지점 pose를 torch tensor로 변환
                    pregrasp_pose = np.concatenate([pregrasp_pos, grasp_quat])
                    grasp_pose = np.concatenate([grasp_pos, grasp_quat])
                    pregrasp_pose = torch.tensor(pregrasp_pose, device=device).unsqueeze(0)
                    grasp_pose = torch.tensor(grasp_pose, device=device).unsqueeze(0)

                    ######## TODO: 예측 grasp/pregrasp 를 statemachine 에 업데이트 ########


                    ###################################################################

                    # SAM3로 선택한 타겟이 씬의 어떤 object인지 식별
                    # (grasp 위치(env-local xy)에 가장 가까운 object를 타겟으로 간주)
                    env_origin = env.scene.env_origins[env_num]
                    grasp_xy = grasp_pose[0, :2]
                    min_dist = None
                    for name in object_names:
                        obj_xy = env.scene[name].data.root_pos_w[env_num, :2] - env_origin[:2]
                        d = torch.norm(obj_xy - grasp_xy)
                        if min_dist is None or d < min_dist:
                            min_dist = d
                            target_obj_name = name
                    print(f"[INFO] Target scene object: {target_obj_name} (dist to grasp: {min_dist:.3f})")
            ####################################################################################
            elif pick_and_place_sm.sm_state[env_num] == PickAndPlaceSmState.READY:
                # ---- 관측 자세를 joint 각도로 고정 ----
                # READY 진입이 아니라 "ready_pose에 실제 도달"했을 때만 캡처
                ee_now = env.scene["ee_frame"].data.target_pos_w[0, 0, :] - env.scene.env_origins[0]
                ee_now[2] -= 0.5

                if ready_joint_pos is not None:
                    robot.write_joint_state_to_sim(
                        ready_joint_pos,
                        torch.zeros_like(ready_joint_pos),
                        joint_ids=arm_joint_ids,
                    )
                    robot.update(dt=0.0)
                elif torch.linalg.norm(ee_now - pick_and_place_sm.ready_pose[env_num, :3]) < pick_and_place_sm.position_threshold:
                    ready_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()

            # 로봇의 End-Effector 위치와 자세를 기반으로 actions 계산
            ee_frame_sensor = env.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            ########### TODO: ee_pose 구성 + statemachine compute ###########
            # 1) ee_pose = 위치(tcp_rest_position) + 쿼터니언(tcp_rest_orientation) 이어붙이기
            #    힌트: torch.cat([...], dim=-1)
            # 2) actions = pick_and_place_sm.compute(ee_pose=..., grasp_pose=..., pregrasp_pose=...)
            #    grasp_pose/pregrasp_pose 는 pick_and_place_sm.grasp_pose / .pregrasp_pose
            ee_pose =
            actions =
            ################################################################

            # [VIS] 명령하는 grasp_pose(목표)를 월드 좌표 마커로 표시 → 실제 TCP 마커(ee_frame)와 비교
            # grasp_pose는 base(env-local) 기준이고 로봇 base 회전=identity 이므로 world = root_pos + grasp_xyz
            root_pos_w = env.scene["robot"].data.root_state_w[:, :3]
            gp = pick_and_place_sm.grasp_pose                     # (num_envs, 7) (x,y,z,qw,qx,qy,qz)
            grasp_marker.visualize(translations=gp[:, :3] + root_pos_w, orientations=gp[:, 3:7])

            # 환경에 대한 액션을 실행 (ManagerBasedEnv.step → (obs, extras) 2-튜플 반환)
            obs, _ = env.step(actions)

            # 시뮬레이션 종료 여부 체크: 테이블의 "모든" 물체가 바구니에 들어갔는지 확인
            bin_pos = env.scene["bin"].data.root_pos_w   # (num_envs, 3)
            # 이미 바구니에 들어간 물체 집합(상태 기억). 매 step이 아니라 "새로" 들어간 순간에만 로그를 찍기 위함.

            for name in object_names:
                obj_pos = env.scene[name].data.root_pos_w   # (num_envs, 3)
                dist_xy = torch.norm(bin_pos[:, :2] - obj_pos[:, :2], dim=1)   # 수평(x, y) 거리
                dist_z = torch.abs(bin_pos[:, 2] - obj_pos[:, 2])             # 수직(z) 거리
                # 수평 0.25m, 수직 0.15m 이내이면 바구니에 들어간 것으로 판단 (필요시 임계값 튜닝)
                in_bin = bool(((dist_xy < 0.40) & (dist_z < 0.15)).all().item())
                if in_bin and name not in objects_in_bin:
                    # 이번 step에 "새로" 들어온 물체일 때만 출력
                    objects_in_bin.add(name)
                    print(f"[INFO] Objects in bin: {len(objects_in_bin)}/{len(object_names)} (+{name})")


            ############## TODO: 완료 시 종료하지 말고 리셋 후 반복 ##############
            # 모든 물체가 들어갔는지 판정
            dones = 
            
            # dones 이면: env.reset() → pick_and_place_sm.reset_idx() → objects_in_bin.clear()
            # (day2의 4.6에서 done 시 env.reset() 하던 것과 같은 패턴)
            if dones:
                print("[INFO] All objects placed in the bin. Resetting...")

            ###############################################################

    # ============================================================================
    # 5. 정리 작업
    # ============================================================================
    # 환경 종료
    env.close()


# ============================================================================
# 6. 프로그램 실행
# ============================================================================
if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
