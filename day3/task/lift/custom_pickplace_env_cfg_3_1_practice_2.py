# ============================================================
#  [문제 2] 관측 설계 · 손목+탑 카메라 이미지 관측 (concat 불가)
#  1교시 · Pick&Place  |  저장 위치: task/lift/custom_pickplace_env_cfg_3_1.py
#  ── 할 일: 아래 TODO(문제2) 주석의 ??? 3곳을 채우세요.
#     손목/탑 카메라 RGB 이미지 관측을 정의하고, 이미지+벡터는 이어붙일 수 없음을 반영한다.
#  (이 파일 하나만으로는 실행되지 않습니다: 나머지 프로젝트 코드 필요)
# ============================================================
from dataclasses import MISSING
import math

# Isaac Lab 관련 라이브러리 임포트
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from . import mdp_3_1 as mdp

@configclass
class CustomUsdFileCfg(UsdFileCfg):
    """커스텀 USD 파일 config - 물리 소재 경로를 지정하기 위함(기존의 UsdFileCfg 에는 물리 소재가 없음)"""
    physics_material_path: str = "material"
    physics_material: materials.PhysicsMaterialCfg | None = None


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """로봇과 물체가 포함된 기본 Scene 구성 Config"""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    camera: CameraCfg = MISSING
    top_camera: CameraCfg = MISSING

    # 테이블 오브젝트 (기본 환경 오브젝트)
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

    # 바구니(Bin) 오브젝트
    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.6, 0.555), rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DAY3_ASSET_DIR / "basket" / "basket.usd"),
            scale=(0.8, 0.25, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.7, 0.5), metallic=0.2, roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        ),
    )
    
    #################################### Set T-bar Object in Scene ####################################
    def __post_init__(self):
        """환경 생성 후 자동으로 실행되는 추가 세팅 코드"""

        TBAR_USD_PATH = str(DAY3_ASSET_DIR / "t_bar" / "T_bar.usd")
        attr_name = "object_0"

        obj_cfg = RigidObjectCfg(                                                                                            
            prim_path=f"{{ENV_REGEX_NS}}/{attr_name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.55),
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


""" MDP 세팅 (액션, 관측, 이벤트) """

@configclass
class ActionsCfg:
    """MDP에 사용되는 액션(action) 정의"""
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """ACT 학습용 관측: 관절 상태 + 카메라 이미지 2개"""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_vel = ObsTerm(func=base_mdp.joint_vel)
        # TODO(문제2) 손목 카메라 / 탑 카메라의 RGB 이미지를 관측으로 추가하라.
        #   힌트: func 는 base_mdp.image 를 쓰고, params 에
        #         "sensor_cfg": SceneEntityCfg("<센서이름>") 와 "data_type": "<타입>" 을 넣는다.
        #         센서이름은 Scene 에 정의된 camera / top_camera 를 가리킨다.
        wrist_cam = ObsTerm(
            func=base_mdp.image,
            params=???,
        )
        top_cam = ObsTerm(
            func=base_mdp.image,
            params=???,
        )

        def __post_init__(self):
            self.enable_corruption = False
            # TODO(문제2) 관절 벡터(1D)와 카메라 이미지(3D)는 shape 이 달라 하나의 텐서로 이어붙일 수 없다.
            #   각 관측 항목을 dict 형태로 따로 유지하려면 이 값을 True/False 중 무엇으로 두어야 하는가?
            self.concatenate_terms = ???

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_0"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "yaw": (-math.pi/4, math.pi/4),
            },
            "velocity_range": {},
        },
    )


""" 최종 환경 config """
@configclass
class TBarPickPlaceEnvCfg(ManagerBasedEnvCfg):
    """환경 전체 설정 (Data Collection 전용)"""

    # Scene 구성
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # MDP 세팅
    events: EventCfg = EventCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        """환경 생성 후 추가 세팅"""
        self.decimation = 2
        self.episode_length_s = 6.0
        
        # 시뮬레이션 기본 설정
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        
        # PhysX 물리엔진 세부 튜닝
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
