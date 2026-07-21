# ============================================================
#  [문제 7] 도메인 랜덤화 · 커스텀 리셋 (좌우 랜덤 + sim 직접 기록)
#  2교시 · Teleop  |  저장 위치: task/lift/custom_pusht_env_cfg_3_2.py
#  ── 할 일: 아래 TODO(문제7) 주석의 ??? 3곳을 채우세요.
#     T-bar 를 좌/우 무작위로 배치하고, 계산한 pose/velocity 를 시뮬레이터에 직접 기록한다.
#  (이 파일 하나만으로는 실행되지 않습니다: 나머지 프로젝트 코드 필요)
# ============================================================
from dataclasses import MISSING
import math

# Isaac Lab 관련 라이브러리 임포트
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import torch
from isaaclab.envs import ManagerBasedEnv  
from isaaclab.envs import ManagerBasedEnvCfg  
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners import materials
from isaaclab.envs import mdp as base_mdp
from pathlib import Path

DAY3_ASSET_DIR = Path(__file__).resolve().parents[2] / "data" / "assets"

# mdp 관련 함수와 config
from . import mdp_3_2 as mdp

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """로봇과 물체가 포함된 기본 Scene 구성 Config"""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),   # 테이블 상단면(z=0.5)에 로봇 베이스 배치
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.4975,
                "panda_joint3": 0.0,
                "panda_joint4": -2.3555,
                "panda_joint5": 0.0,
                "panda_joint6": 2.8516,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.0,   # 닫힘
                "panda_finger_joint2": 0.0,   # 닫힘
            },
        ),
    )
    ee_frame: FrameTransformerCfg = MISSING
    camera: CameraCfg = MISSING
    top_camera: CameraCfg = MISSING

    # 테이블 오브젝트
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
                size=(1.6, 2.0, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.2, roughness=0.5),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.5, restitution=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.4, 0.25)),
    )

    # 바닥 Plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=GroundPlaneCfg(),
    )

    # 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    #################################### Set T-bar Object in Scene ####################################
    def __post_init__(self):
        """환경 생성 후 자동으로 실행되는 추가 세팅 코드"""
        TBAR_USD_PATH = str(DAY3_ASSET_DIR / "t_bar" / "T_bar.usd")
        GOAL_USD_PATH = str(DAY3_ASSET_DIR / "t_bar" / "T_bar_goal.usd")
        attr_name = "object_0"

        obj_cfg = RigidObjectCfg(                                                                                            
            prim_path=f"{{ENV_REGEX_NS}}/{attr_name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.55),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=TBAR_USD_PATH,
                scale=(0.001, 0.001, 0.001),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )
        setattr(self, attr_name, obj_cfg)

        # 타겟 영역(초록색 T-bar): 완전히 납작하게(Z scale) 만들어서 테이블 위 스티커처럼 보이게 하여 물리적 충돌 방지
        target_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/target_object",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.501),
                rot=(-0.7071068, 0.0, 0.0, -0.7071068),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=GOAL_USD_PATH,
                scale=(0.001, 0.001, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )
        setattr(self, "target_object", target_cfg)


""" MDP 세팅 (액션, 관측, 이벤트, 종료조건) """

@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_vel = ObsTerm(func=base_mdp.joint_vel)
        wrist_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
        )
        top_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "data_type": "rgb"},
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()


def reset_tbar_left_right(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
):
    """T-bar를 중앙(y=0.0)을 피해서 좌우에 무작위로 배치합니다."""
    asset = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state[env_ids].clone()
    
    num_envs = len(env_ids)
    # TODO(문제7) T-bar 를 중앙(y=0)이 아니라 좌/우 중 하나로 보낼 부호를 무작위로 만든다.
    #   torch.randint(0, 2, (num_envs,), device=env.device) 는 0 또는 1 을 준다.
    #   이 값을 -1 또는 +1 로 바꾸려면 어떤 산술을 적용해야 하는가?
    left_right = ???
    # 좌우 오프셋 0.2m ~ 0.3m
    y_offset = left_right * (0.2 + torch.rand(num_envs, device=env.device) * 0.1)
    
    # X축은 약간의 노이즈만
    x_noise = (torch.rand(num_envs, device=env.device) - 0.5) * 0.1
    
    default_root_state[:, 0] = 0.4 + x_noise
    default_root_state[:, 1] = 0.0 + y_offset
    
    # Yaw 회전 노이즈
    yaw = (torch.rand(num_envs, device=env.device) - 0.5) * math.pi
    # quaternion (w, x, y, z) for z-axis rotation
    default_root_state[:, 3] = torch.cos(yaw / 2)
    default_root_state[:, 4] = 0.0
    default_root_state[:, 5] = 0.0
    default_root_state[:, 6] = torch.sin(yaw / 2)
    
    # TODO(문제7) 계산한 상태를 시뮬레이터에 직접 기록한다. (이벤트 방식이 아닌 커스텀 리셋)
    #   default_root_state 레이아웃: [:, 0:3]=위치, [:, 3:7]=쿼터니언, [:, 7:13]=선속도/각속도
    #   pose 에는 앞 7개, velocity 에는 그 뒤를 슬라이스해서 넘겨라.
    asset.write_root_pose_to_sim(???, env_ids)
    asset.write_root_velocity_to_sim(???, env_ids)


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=reset_tbar_left_right,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_0"),
        },
    )


""" 최종 환경 config """
@configclass
class PushTEnvCfg(ManagerBasedEnvCfg):
    """PushT 환경 전체 설정 (Data Collection 및 Teleop용)"""
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    
    events: EventCfg = EventCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 6.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # 1. 로봇 세팅 (상단 ObjectTableSceneCfg에서 설정된 초기값을 사용합니다)
        
        # 2. 카메라 세팅
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/handeye_camera",
            update_period=0.0,
            height=480, width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.035, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="ros"),
        )
        self.scene.top_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_camera",
            update_period=0.0,
            height=480, width=640,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 2.5),
                rot=(-0.7071068, 0, -0.7071068, 0),
                convention="world",
            ),
        )
        
        # 3. 엔드이펙터 프레임
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )
        
        # 4. 액션 세팅 (IK 및 그리퍼)
        # IK 절대 pose 제어: 목표 '위치+자세(pose)'를 절대좌표로 받아 관절각을 계산
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )