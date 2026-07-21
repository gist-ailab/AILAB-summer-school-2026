"""
Isaac Lab 환경에서 리프팅 태스크를 위한 기본 환경 설정

1. 리프팅 태스크의 기본 씬 구조 정의
2. MDP(Markov Decision Process) 구성 요소 설정
3. 보상 함수, 종료 조건, 관측 설정
4. 강화학습을 위한 환경 구성
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트
# ============================================================================
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab.envs import mdp

# ============================================================================
# 2. 씬 정의 (Scene Definition)
# ============================================================================

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """
    로봇과 오브젝트가 포함된 리프팅 씬의 설정
    
    구체적인 씬은 자식 클래스에서 정의.
    자식 클래스에서는 타겟 오브젝트, 로봇, 엔드 이펙터 프레임을 설정.
    """

    # 로봇: 자식 클래스에서 구체적으로 설정됨
    robot: ArticulationCfg = MISSING
    # 엔드 이펙터 센서: 자식 클래스에서 구체적으로 설정됨
    ee_frame: FrameTransformerCfg = MISSING
    # 타겟 오브젝트: 자식 클래스에서 구체적으로 설정됨
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # ============================================================================
    # 3. 커스텀 테이블 설정
    # ============================================================================
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",  # 테이블의 USD 경로
        spawn=sim_utils.CuboidCfg(
                size=(2.0, 1.5, 0.5),  # 테이블 크기 (가로, 세로, 높이)
                # 시각적 재질 설정
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5),  # 회색
                    metallic=0.2,  # 금속성
                    roughness=0.5  # 거칠기
                ),
                # 물리적 재질 설정
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8,   # 정지 마찰 계수
                    dynamic_friction=0.5,  # 동적 마찰 계수
                    restitution=0.1        # 탄성 계수
                ),
                # 충돌 속성 설정
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        # 초기 상태 설정
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0, 0.25)),  # 테이블 위치
    )

    ############# TODO: Plane 생성 #############
    #### GroundPlaneCfg 사용 ####
    

    ############# TODO: Light 생성 #############
    #### sim_utils.DomeLightCfg 사용 ####
    
    
# ============================================================================
# 6. MDP 설정 (Markov Decision Process)
# ============================================================================

@configclass
class ActionsCfg:
    """MDP의 액션(Action) 설정"""
    ############# TODO: Robot Joint Action Config 생성 #############
    # 팔 액션: 자식 클래스에서 구체적으로 설정됨

    # 그리퍼 액션: 자식 클래스에서 구체적으로 설정됨

    ###############################################################

@configclass
class ObservationsCfg:
    """MDP의 관측(Observation) 설정"""
    # 스크립트에서 obs를 사용하지 않으므로 관측을 비움.


@configclass
class EventCfg:
    """이벤트 설정"""

    # 전체 씬을 기본 상태로 리셋
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 오브젝트 위치만 랜덤하게 리셋
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 오브젝트 초기 위치 범위
            "pose_range": {
                "x": (-0.1, 0.1),    # x 위치 범위
                "y": (-0.25, 0.25),  # y 위치 범위
                "z": (0.0, 0.0)      # z 위치 범위 (테이블 위)
            },
            "velocity_range": {},  # 속도 범위 (비어있음)
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),  # 오브젝트 설정
        },
    )

# ============================================================================
# 7. 환경 설정 (Environment Configuration)
# ============================================================================

@configclass
class LiftEnvCfg(ManagerBasedEnvCfg):
    """리프팅 환경의 설정"""

    # 씬 설정
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # ManagerBasedEnvCfg 기본 설정
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """초기화 후 실행되는 메서드"""
        # ============================================================================
        # 8. 일반 설정
        # ============================================================================
        self.decimation = 2              # 액션 반복 횟수
        self.episode_length_s = 5.0      # 에피소드 길이 (5초)
        
        # ============================================================================
        # 9. 시뮬레이션 설정
        # ============================================================================
        self.sim.dt = 0.01  # 시뮬레이션 시간 스텝 (100Hz)
        self.sim.render_interval = self.decimation  # 렌더링 간격

        # ============================================================================
        # 10. PhysX 물리 엔진 설정
        # ============================================================================
        # 튀어오름 임계 속도 설정
        self.sim.physx.bounce_threshold_velocity = 0.01
        
        # GPU 메모리 설정
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4  # 4MB
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024  # 16KB
        
        # 마찰 상관 거리 설정
        self.sim.physx.friction_correlation_distance = 0.00625
