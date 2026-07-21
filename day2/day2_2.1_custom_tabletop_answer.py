# day2_2.1_custom_tabletop_answer.py
# Isaac Sim 5.1.0 + Isaac Lab 2.3.x 계열 기준 예시
#
# 실행 방법:
#   isaaclab -p day2_2.1_custom_tabletop_answer.py

import argparse
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. Isaac Sim 실행 옵션 설정
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Create a simple tabletop scene with Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim / Isaac Lab 관련 import는 AppLauncher 이후에 수행
import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """GUI에서 구성한 tabletop scene을 Isaac Lab config로 다시 생성함."""

    # -------------------------------------------------------------------------
    # 2. Ground Plane 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   Create > Physics > Ground Plane
    #   GroundPlane 우클릭 > Create > Material > OmniPBR
    #   Shader > Albedo > Color Tint = black
    #
    # Code:
    #   sim_utils.GroundPlaneCfg:
    #       실제 Ground Plane Prim을 어떻게 만들지 정의함
    #   AssetBaseCfg:
    #       Isaac Lab scene 안에 이 Prim을 asset으로 등록함
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(50.0, 50.0),
            color=(0.0, 0.0, 0.0),
        ),
    )

    # -------------------------------------------------------------------------
    # 3. Table 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   Create > Shape > Cube
    #   Transform:
    #       Translate = (0, 0, 0.5)
    #       Scale     = (3.2, 4.0, 1.0)
    #
    #   Add > Physics > Rigid Body
    #   Add > Physics > Collider
    #
    # Code:
    #   sim_utils.CuboidCfg:
    #       Cube/Cuboid Prim을 어떻게 만들지 정의함
    #   RigidObjectCfg:
    #       Isaac Lab이 이 물체를 rigid object로 관리하도록 등록함
    #
    # 주의:
    #   GUI의 기본 Cube size=1에 scale=(3.2, 4.0, 1.0)을 준 것과 같게
    #   코드에서는 size=(3.2, 4.0, 1.0)으로 바로 생성함.
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(3.2, 4.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
                roughness=0.5,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -------------------------------------------------------------------------
    # 4. Dome Light 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   World 우클릭 > Create > Light > Dome Light
    #   Property > Light > Intensity = 4000
    #
    # Code:
    #   sim_utils.DomeLightCfg:
    #       Dome Light Prim을 어떻게 만들지 정의함
    #   AssetBaseCfg:
    #       Isaac Lab scene에 light asset으로 등록함
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=4000.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    # -------------------------------------------------------------------------
    # 5. Camera 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   World 우클릭 > Create > Camera
    #
    # Code:
    #   CameraCfg:
    #       Isaac Lab이 camera sensor로 관리할 수 있도록 등록함
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def main():
    # -------------------------------------------------------------------------
    # 6. Simulation 설정
    # -------------------------------------------------------------------------
    # GUI에서 Play 버튼을 누를 때 사용할 simulation context를 코드에서 생성함.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)

    # -------------------------------------------------------------------------
    # 7. Scene 생성
    # -------------------------------------------------------------------------
    # num_envs=1:
    #   GUI에서 보던 것처럼 하나의 scene만 생성함
    #
    # env_spacing:
    #   나중에 num_envs를 늘렸을 때 각 environment 사이 간격을 의미함
    scene_cfg = TabletopSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # -------------------------------------------------------------------------
    # 8. Simulation 초기화
    # -------------------------------------------------------------------------
    # 여기서 config에 선언한 Ground, Table, Light, Camera가 실제 Stage에 생성됨.
    sim.reset()

    print("[INFO] Tabletop scene이 Isaac Lab config로 생성되었습니다.")
    print("[INFO] GUI에서 Stage Tree를 열어 /World/envs/env_0/Table, Camera 등을 확인하세요.")

    # -------------------------------------------------------------------------
    # 9. Simulation loop
    # -------------------------------------------------------------------------
    while simulation_app.is_running():
        # Isaac Lab scene 안의 asset 데이터를 시뮬레이터에 반영
        scene.write_data_to_sim()

        # 물리/렌더링 step 진행
        sim.step()

        # 시뮬레이션 결과를 Isaac Lab buffer로 업데이트
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()