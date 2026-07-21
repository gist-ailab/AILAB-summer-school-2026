# ============================================================
#  [문제 4] 로봇 액션 정의 · 관절 위치 제어 + 이진 그리퍼 (joint-space)
#  1교시 · Pick&Place
#  저장 위치: task/lift/config/joint_pos_env_cfg_3_1.py
#  실습용 — 아래 코드에서 ??? 부분만 채운 뒤, 위 경로의 파일로 저장/교체하세요.
#  (이 파일 하나만으로는 실행되지 않습니다: 나머지 프로젝트 코드가 필요)
# ============================================================
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from  task.lift.custom_pickplace_env_cfg_3_1_answer import TBarPickPlaceEnvCfg

# 미리 정의된 마커/로봇/카메라 config 불러오기
from isaaclab.markers.config import FRAME_MARKER_CFG  # 프레임(좌표계) 시각화 마커 설정
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # 프랑카 로봇 기본 config
from isaaclab.sim import PinholeCameraCfg               # 핀홀 카메라 모델 설정


@configclass
class FrankaTBarPickPlaceEnvCfg(TBarPickPlaceEnvCfg):
    def __post_init__(self):
        # 부모 환경 config 초기화
        super().__post_init__()

        # 1. 로봇 모델을 Franka로 지정, prim_path도 변경
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. 액션 설정
        # 2-1. 로봇 팔(arm): 관절 위치 제어(Joint Position Control)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            # TODO(문제4) 이 관절 위치 액션이 제어할 '팔' 관절들을 정규표현식으로 지정하라.
            #   Franka 팔 관절 이름은 panda_joint1 ~ panda_joint7. 하나의 정규식으로 전부 잡으려면?
            joint_names=???,
            scale=0.5,                 # (제공) 정책 출력에 곱하는 스케일
            use_default_offset=True    # (제공) 기본자세 기준 상대 이동
        )
        # 2-2. 그리퍼(gripper): 바이너리(open/close) 제어
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],     # 모든 그리퍼 관절 대상
            # TODO(문제4) 이진(binary) 그리퍼는 '열림/닫힘' 두 상태뿐이다. 각 상태의 손가락 관절 목표값을 지정하라.
            #   Franka 손가락 관절 범위: 0.0(완전히 닫힘) ~ 0.04(완전히 열림). 형식: {"panda_finger_.*": 값}
            open_command_expr=???,     # 열 때
            close_command_expr=???,    # 닫을 때
        )

        # 3. 카메라 센서 설정 (handeye 카메라: 프랑카 손 끝에 부착)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/handeye_camera",      # 카메라가 위치할 prim 경로
            update_period=0.0,      # 매 시뮬레이션 스텝마다 갱신 (50Hz 저장용)
            height=480, width=640,  
            data_types=["rgb", "distance_to_image_plane"],      # RGB+Depth
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            # 카메라 위치/방향 오프셋 (ROS convention, Z축 90도 회전)
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.035, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="ros"),
        )

        # 3-2. 전역(탑다운) 카메라 - 테이블 전체, T_bar, 바구니를 모두 비춤
        self.scene.top_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_camera",
            update_period=0.0,      # 매 시뮬레이션 스텝마다 갱신 (50Hz 저장용)
            height=480, width=640,  # 640x480 해상도
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.3, 2.5),   # 작업 영역(T_bar 0.5,0.0 ~ 바구니 0.2,0.6) 중앙 위
                rot=(-0.7071068, 0, -0.7071068, 0),  # 아래를 향함
                convention="world",
            ),
        )

        # 4. 엔드이펙터(EE) 프레임(좌표계) 설정 및 프레임 시각화 마커 세팅
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)         # 시각화 마커 크기 조정
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",           # 로봇 base frame
            debug_vis=False,                                        # 디버그용 시각화 on/off
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",    # EE 프레임 대상
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),       # TCP 오프셋
                ),
            ],
        )


@configclass
class FrankaTBarPickPlaceEnvCfg_PLAY(FrankaTBarPickPlaceEnvCfg):
    """
    데모/테스트/시연(play)용 작은 환경 설정.
    - 환경 수 적고, 랜덤성/노이즈 없이 항상 같은 관측값 제공.
    """
    def __post_init__(self):
        # 부모 환경 config 초기화
        super().__post_init__()
        # 1. 환경 수와 spacing 축소
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 2. 관측 노이즈/랜덤화 비활성화
        self.observations.policy.enable_corruption = False