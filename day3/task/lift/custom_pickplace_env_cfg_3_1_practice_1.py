# ============================================================
#  [문제 1] Scene 구성 · T-bar 배치 / 스케일(mm→m) / 질량
#  1교시 · Pick&Place  |  저장 위치: task/lift/custom_pickplace_env_cfg_3_1.py
#  ── 할 일: 아래 TODO(문제1) 주석의 ??? 3곳을 채우세요.
#     T-bar의 초기 위치·스케일(mm→m)·질량을 채워 물체를 테이블 위에 올바르게 배치한다.
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
                # TODO(문제1) T-bar 초기 위치 (x, y, z).
                #   테이블 윗면 높이가 z=0.5 이므로, 그 위에 살짝 얹히도록 z 를 정하라. (x 는 로봇 앞쪽)
                pos=???,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=TBAR_USD_PATH,
                # TODO(문제1) 스케일. USD 원본이 mm 단위로 제작되어 있다.
                #   m 단위 시뮬레이션에서 실제 크기로 보이게 하려면 몇 배로 줄여야 하는가?
                scale=???,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                # TODO(문제1) T-bar 질량(kg). 너무 무거우면 못 집고, 너무 가벼우면 접촉 시 튀어 날아간다.
                mass_props=sim_utils.MassPropertiesCfg(mass=???),
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
