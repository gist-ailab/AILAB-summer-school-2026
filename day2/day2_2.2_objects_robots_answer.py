# day2_2.2_objects_robots_answer.py
#
# 실행 방법:
#   isaaclab -p day2_2.2_objects_robots_answer.py
#
# 실행 후 확인:
#   GUI에서 /World/envs/env_0/Tabletop, Object1, Object2, Object3, Robot을 확인

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. Isaac Sim 실행 옵션 설정
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Load tabletop scene and add objects/robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


DAY2_DIR = Path(__file__).resolve().parent


@configclass
class ObjectsRobotsSceneCfg(InteractiveSceneCfg):
    """이전 시간에 만든 tabletop scene을 불러오고 object와 robot을 추가함."""

    # -------------------------------------------------------------------------
    # 2. 이전 시간에 만든 Tabletop Scene 불러오기
    # -------------------------------------------------------------------------
    # GUI:
    #   2_2.1 시간에 Ground, Table, Light, Camera를 만들고
    #   scenes/2_2.1_tabletop.usda로 저장
    #
    # Code:
    #   UsdFileCfg로 저장된 tabletop scene을 그대로 불러옴
    tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(DAY2_DIR / "scenes/2_2.1_tabletop.usda").as_posix(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -------------------------------------------------------------------------
    # 3. Object 추가
    # -------------------------------------------------------------------------
    object1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(DAY2_DIR / "assets/objects/006_mustard_bottle/final.usd").as_posix(),
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.1, -0.17, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(DAY2_DIR / "assets/objects/013_apple/final.usd").as_posix(),
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.74, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(DAY2_DIR / "assets/objects/065-b_cups/final.usd").as_posix(),
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.16, 0.24, 1.1),
            rot=(0.86, -0.29, 0.32, -0.27),
        ),
    )

    # -------------------------------------------------------------------------
    # 4. Custom Franka Robot 추가
    # -------------------------------------------------------------------------
    # GUI:
    #   Franka Panda에 Hand-E gripper를 결합한 custom robot asset 사용
    #
    # Code:
    #   완성된 robot USD를 ArticulationCfg로 불러와 배치함
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(DAY2_DIR / "assets/robots/CustomFranka/franka_hande.usd").as_posix(),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.06, 0.019, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.5,
                "panda_joint3": 0.0,
                "panda_joint4": -1.5,
                "panda_joint5": 0.0,
                "panda_joint6": 1.5,
                "panda_joint7": 0.0,
                "Slider_1": 0.0,
                "Slider_2": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint.*"],
                stiffness=400.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["Slider_.*"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )


def main():
    # -------------------------------------------------------------------------
    # 5. Simulation 설정
    # -------------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)

    # -------------------------------------------------------------------------
    # 6. Scene 생성
    # -------------------------------------------------------------------------
    scene_cfg = ObjectsRobotsSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # -------------------------------------------------------------------------
    # 7. Simulation 초기화
    # -------------------------------------------------------------------------
    sim.reset()

    print("[INFO] Objects and custom robot scene이 생성되었습니다.")
    print("[INFO] GUI에서 /World/envs/env_0/Tabletop, Object1, Object2, Object3, Robot을 확인하세요.")

    # -------------------------------------------------------------------------
    # 8. Simulation loop
    # -------------------------------------------------------------------------
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()