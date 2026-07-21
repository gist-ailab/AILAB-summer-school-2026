# day2_2.1_custom_tabletop_practice.py
#
# 실행 방법:
#   isaaclab -p day2_2.1_custom_tabletop_practice.py
#
# 목표:
#   빈칸을 채워 Ground, Table, Light, Camera가 포함된 tabletop scene을 생성함

import argparse
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. Isaac Sim 실행 옵션 설정
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Create a simple tabletop scene with Isaac Lab.")

# Problem 1:
# Isaac Lab에서 자주 쓰는 실행 옵션을 parser에 추가
____________________________(parser)
args_cli = parser.parse_args()

# Isaac Sim 앱을 실행 옵션에 맞게 시작
app_launcher = _______________(args_cli)

# 실행 중인 Isaac Sim app 객체를 가져옴
simulation_app = app_launcher.____


# Isaac Sim / Isaac Lab 관련 import는 AppLauncher 이후에 수행
import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


# Problem 2:
# 아래 class를 Isaac Lab 설정 class로 사용하겠다는 표시
@___________
class TabletopSceneCfg(___________________):
    """GUI에서 구성한 tabletop scene을 Isaac Lab config로 다시 생성함."""

    # -------------------------------------------------------------------------
    # 2. Ground Plane 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   Create > Physics > Ground Plane
    #
    # Code:
    #   Ground처럼 별도 제어가 필요 없는 기본 asset을 scene에 등록
    ground = ____________(
        __________="/World/Ground",
        ______=sim_utils.GroundPlaneCfg(
            size=(50.0, 50.0),
            color=(0.0, 0.0, 0.0),
        ),
    )

# Problem 3:
    # -------------------------------------------------------------------------
    # 3. Table 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   Create > Shape > Cube
    #   Add > Physics > Rigid Body
    #   Add > Physics > Collider
    #
    # Code:
    #   Table을 물리 object로 관리하도록 등록
    table = ______________(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.__________(
            size=(3.2, 4.0, 1.0),
            rigid_props=sim_utils.________________________(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            collision_props=sim_utils.______________________(
                collision_enabled=True,
            ),
            visual_material=sim_utils.__________________(
                diffuse_color=(0.5, 0.5, 0.5),
                roughness=0.5,
                metallic=0.0,
            ),
        ),
        __________=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

# Problem 4:
    # -------------------------------------------------------------------------
    # 4. Dome Light 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   World 우클릭 > Create > Light > Dome Light
    #
    # Code:
    #   Light를 scene의 기본 구성 요소로 등록
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.____________(
            __________=4000.0,
            _____=(1.0, 1.0, 1.0),
        ),
    )

# Problem 5:
    # -------------------------------------------------------------------------
    # 5. Camera 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   World 우클릭 > Create > Camera
    #
    # Code:
    #   Camera를 Isaac Lab sensor로 관리하도록 등록
    camera = _________(
        prim_path="{ENV_REGEX_NS}/Camera",
        ______=480,
        _____=640,
        __________=["rgb"],
        spawn=sim_utils.________________(),
        offset=CameraCfg._________(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

# Problem 6:
def main():
    # -------------------------------------------------------------------------
    # 6. Simulation 설정
    # -------------------------------------------------------------------------
    # 물리 timestep 같은 simulation 기본 설정을 정의
    sim_cfg = sim_utils._____________(dt=1.0 / 60.0)

    # GUI의 Play/Step에 해당하는 simulation 실행 관리자 생성
    sim = sim_utils._________________(sim_cfg)

    # -------------------------------------------------------------------------
    # 7. Scene 생성
    # -------------------------------------------------------------------------
    scene_cfg = TabletopSceneCfg(
        _________=1,
        ___________=4.0,
    )

    # TabletopSceneCfg를 읽어 실제 Stage에 Ground, Table, Light, Camera를 생성
    scene = ________________(scene_cfg)

    # -------------------------------------------------------------------------
    # 8. Simulation 초기화
    # -------------------------------------------------------------------------
    # config에 선언한 요소들이 실제 Stage에 생성되도록 초기화
    sim._____()

    print("[INFO] Tabletop scene이 Isaac Lab config로 생성되었습니다.")
    print("[INFO] GUI에서 /World/envs/env_0/Table, Camera 등을 확인하세요.")

    # -------------------------------------------------------------------------
    # 9. Simulation loop
    # -------------------------------------------------------------------------
    while simulation_app.is_running():
        # Isaac Lab scene 안의 asset 데이터를 simulator에 반영
        scene._________________()

        # 물리/렌더링 step 진행
        sim.______()

        # simulation 결과를 Isaac Lab buffer로 업데이트
        scene.______(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()