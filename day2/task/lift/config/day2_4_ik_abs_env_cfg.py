from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


from day2.task.lift.config import day2_4_joint_pos_env_cfg

# 사전 정의된 Franka Panda High PD 세팅 import
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


# 강체 들어올리는 환경
@configclass
class FrankaYCBPickPlaceEnvCfg(day2_4_joint_pos_env_cfg.FrankaYCBPickPlaceEnvCfg):
    """
    Franka Panda 로봇이 큐브(또는 물체)를 들어올리는 환경 Config 클래스
    (joint_pos_env_cfg_3_3.FrankaYCBPickPlaceEnvCfg 를 상속받아 사용)
    """
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 먼저 실행
        super().__post_init__()

        # 로봇 설정: PD 제어를 사용하는 Franka로 설정 (IK 성능 향상 목적)
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)   # 초기 로봇 위치
        
        # 액션(Action) 설정: Franka에 맞는 IK 기반 액션으로 세팅
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",                 # 대상 로봇 이름
            joint_names=["panda_joint.*"],      # 제어할 관절 이름(정규표현식)
            body_name="panda_hand",             # IK 기준 바디(엔드이펙터)
            controller=DifferentialIKControllerCfg(
                command_type="pose",            # pose 명령 타입
                use_relative_mode=False,        # 절대좌표로 명령
                ik_method="dls"                 # damped least squares IK 방법 사용
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.1034]           # 엔드이펙터와 tcp 오프셋 (z축 방향)
            ),         
        )
