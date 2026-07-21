# day2_2.3_get_observations_practice.py
#
# 실행 방법:
#   isaaclab -p day2_2.3_get_observations_practice.py
#
# 목표:
#   기존 scene의 Camera를 sensor로 연결하고,
#   RGB, depth, semantic mask, bbox를 observations 폴더에 저장함

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. Isaac Sim 실행 옵션 설정
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Get camera observations from an existing scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera sensor를 쓰려면 실행 시 --enable_cameras 필요
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Gf, UsdGeom

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.utils.semantics import add_labels
from isaaclab.utils import configclass


DAY2_DIR = Path(__file__).resolve().parent

BASE_SCENE_PRIM_PATH = "/World/envs/env_0/BaseScene"
CAMERA_PRIM_PATH = f"{BASE_SCENE_PRIM_PATH}/Camera"

# Problem 1
# -------------------------------------------------------------------------
# 2. Semantic label 대상 정의
# -------------------------------------------------------------------------
# object prim 경로와 class label을 dictionary로 정의
OBJECT_LABELS = {
    f"{BASE_SCENE_PRIM_PATH}/Object1": "________",
    f"{BASE_SCENE_PRIM_PATH}/Object2": "________",
    f"{BASE_SCENE_PRIM_PATH}/Object3": "________",
}

TARGET_OBJECT_LABELS = set(OBJECT_LABELS.values())


@configclass
class ObservationSceneCfg(InteractiveSceneCfg):
    """2_2.2 scene을 불러옴."""

    # -------------------------------------------------------------------------
    # 3. 이전 시간에 만든 Scene 불러오기
    # -------------------------------------------------------------------------
    base_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BaseScene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(DAY2_DIR / "scenes/2_2.2_objects_robots.usda").as_posix(),
        ),
    )

def apply_semantic_labels():
    # -------------------------------------------------------------------------
    # 5. Semantic Label 부여
    # -------------------------------------------------------------------------
    # GUI:
    #   Tools > Replicator > Semantics Schema Editor
    #
    # Code:
    #   Stage에서 object prim을 찾고 semantic label을 추가함

    # 현재 열려 있는 USD Stage 가져오기
    stage = omni.usd.____________().___________()

    for prim_path, label in OBJECT_LABELS.items():
        # Stage 안에서 prim_path에 해당하는 object prim 찾기
        prim = stage._____________(prim_path)

        # prim이 실제로 존재하는지 확인
        if not prim._________():
            raise RuntimeError(f"Prim not found: {prim_path}")

        # object prim에 semantic label 추가
        __________(
            prim,
            labels=[label],
            instance_name="class",
            overwrite=True,
        )

        print(f"[INFO] Added semantic label: {prim_path} -> {label}")


def configure_existing_camera():
    # -------------------------------------------------------------------------
    # 4. 기존 Camera 설정 변경
    # -------------------------------------------------------------------------
    # 기존 scene 안에 있던 Camera prim의 transform과 camera property를 수정
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

    if not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {CAMERA_PRIM_PATH}")

    xform = UsdGeom.Xformable(prim)

    translate_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
    orient_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    scale_op = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    # Camera가 table 위 object들을 바라보도록 pose 설정
    translate_op.Set(Gf.Vec3d(3.0, 0.0, 3.0))
    orient_op.Set(
        Gf.Quatd(
            0.6123724356957947,
            Gf.Vec3d(
                0.3535533905932736,
                0.3535533905932738,
                0.6123724356957944,
            ),
        )
    )
    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
    xform.SetXformOpOrder([translate_op, orient_op, scale_op])

    camera = UsdGeom.Camera(prim)
    camera.GetFocalLengthAttr().Set(12.0)
    camera.GetHorizontalApertureAttr().Set(12.0)
    camera.GetVerticalApertureAttr().Set(40.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000000.0))

# Problem 2
def create_existing_camera_sensor():
    # -------------------------------------------------------------------------
    # 6. 기존 Camera를 Isaac Lab Sensor로 연결
    # -------------------------------------------------------------------------
    # GUI:
    #   Data Recorder에서 RGB, depth, semantic segmentation을 선택
    #
    # Code:
    #   CameraCfg.data_types에 취득할 observation type을 지정

    # 기존 Camera prim을 Isaac Lab camera sensor로 사용하기 위한 설정
    camera_cfg = __________(
        prim_path=CAMERA_PRIM_PATH,
        height=720,
        width=1280,
        __________=[
            "rgb",
            "__________________",
            "______________________",
        ],
        ______________________________=False,
        spawn=None,
    )
    # CameraCfg를 실제 Isaac Lab Camera 객체로 생성
    camera = ________(camera_cfg)

    # depth 계산에 필요한 pinhole camera parameter 설정
    camera.cfg.spawn = sim_utils.________________(
        focal_length=12.0,
        horizontal_aperture=12.0,
        vertical_aperture=40.0,
        clipping_range=(0.01, 10000000.0),
    )

    return camera


def extract_target_label(label_info):
    """semantic label 정보에서 object1, object2, object3만 추출함."""
    if not isinstance(label_info, dict):
        return None

    class_label = label_info.get("class", "")

    if isinstance(class_label, list):
        class_names = class_label
    else:
        class_names = [name.strip() for name in str(class_label).split(",")]

    for class_name in class_names:
        if class_name in TARGET_OBJECT_LABELS:
            return class_name

    return None


def save_one_observation_frame(output_dir: Path, camera: Camera):
    # -------------------------------------------------------------------------
    # 7. RGB, Depth, Mask, BBox 저장
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO 9:
    # camera sensor가 만든 RGB, depth, semantic mask tensor dictionary 가져오기
    output = camera.data.________

    # TODO 10:
    # semantic id와 label 이름의 mapping 정보 가져오기
    info = camera.data.____[0]

    # -------------------------------------------------------------------------
    # 7-1. RGB 저장
    # -------------------------------------------------------------------------
    # TODO 11:
    # batch dimension 중 첫 번째 image를 선택하고 RGB 3채널만 사용
    rgb = output["rgb"][0, ..., ___].detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(rgb).save(output_dir / "rgb.png")

    # -------------------------------------------------------------------------
    # 7-2. Depth 저장
    # -------------------------------------------------------------------------
    # TODO 12:
    # 첫 번째 camera의 distance_to_camera depth tensor를 가져와 2D array로 변환
    depth = output["__________________"][0].detach().cpu().numpy().________()
    np.save(output_dir / "depth.npy", depth)

    depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth_vis > 0.0
    if valid.any():
        d_min = depth_vis[valid].min()
        d_max = depth_vis[valid].max()
        depth_vis = (depth_vis - d_min) / max(d_max - d_min, 1e-6)

    depth_vis = (depth_vis * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(output_dir / "depth_vis.png")

    # -------------------------------------------------------------------------
    # 7-3. Semantic segmentation mask 저장
    # -------------------------------------------------------------------------
    # TODO 13:
    # 첫 번째 camera의 semantic id mask를 2D array로 변환
    semantic = (
        output["______________________"][0]
        .detach()
        .cpu()
        .numpy()
        .________()
        .astype(np.int32)
    )
    np.save(output_dir / "semantic_segmentation.npy", semantic)

    semantic_vis = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for semantic_id in np.unique(semantic):
        if semantic_id == 0:
            continue

        color = np.array(
            [
                (semantic_id * 53) % 255,
                (semantic_id * 97) % 255,
                (semantic_id * 193) % 255,
            ],
            dtype=np.uint8,
        )
        semantic_vis[semantic == semantic_id] = color

    Image.fromarray(semantic_vis).save(output_dir / "semantic_segmentation_vis.png")

    # -------------------------------------------------------------------------
    # 7-4. Semantic mask에서 visible 2D BBox 계산
    # -------------------------------------------------------------------------
    id_to_labels = info.get("semantic_segmentation", {}).get("idToLabels", {})
    bboxes = []

    for semantic_id in np.unique(semantic):
        if semantic_id == 0:
            continue

        raw_label = id_to_labels.get(str(int(semantic_id)), id_to_labels.get(int(semantic_id), {}))
        object_label = extract_target_label(raw_label)

        if object_label is None:
            continue

        # TODO 14:
        # semantic mask에서 현재 semantic_id에 해당하는 pixel 좌표 찾기
        ys, xs = np.______(semantic == semantic_id)

        if len(xs) == 0 or len(ys) == 0:
            continue

        bboxes.append(
            {
                "semantic_id": int(semantic_id),
                "label": object_label,
                # TODO 15:
                # object pixel 영역의 최소/최대 좌표로 bbox 계산
                "x_min": int(xs._____()),
                "y_min": int(ys._____()),
                "x_max": int(xs._____()),
                "y_max": int(ys._____()),
            }
        )

    with open(output_dir / "bbox_2d.json", "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2, ensure_ascii=False)

    print("[INFO] Saved observation files:")
    print(f"  - {output_dir / 'rgb.png'}")
    print(f"  - {output_dir / 'depth.npy'}")
    print(f"  - {output_dir / 'depth_vis.png'}")
    print(f"  - {output_dir / 'semantic_segmentation.npy'}")
    print(f"  - {output_dir / 'semantic_segmentation_vis.png'}")
    print(f"  - {output_dir / 'bbox_2d.json'}")


def main():
    # -------------------------------------------------------------------------
    # 8. Simulation 설정
    # -------------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)

    # -------------------------------------------------------------------------
    # 9. Scene 생성
    # -------------------------------------------------------------------------
    scene_cfg = ObservationSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # -------------------------------------------------------------------------
    # 10. Camera / Semantic / Sensor 설정
    # -------------------------------------------------------------------------
    configure_existing_camera()
    apply_semantic_labels()
    camera = create_existing_camera_sensor()

    # -------------------------------------------------------------------------
    # 11. Simulation 초기화
    # -------------------------------------------------------------------------
    sim.reset()

    # -------------------------------------------------------------------------
    # 12. Observation 취득을 위한 Step 진행
    # -------------------------------------------------------------------------
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        # TODO 16:
        # Camera sensor output buffer 업데이트
        camera.______(sim.get_physics_dt())

    camera_output = camera.data.output
    print("[INFO] RGB shape:", camera_output["rgb"].shape)
    print("[INFO] Depth shape:", camera_output["distance_to_camera"].shape)
    print("[INFO] Segmentation shape:", camera_output["semantic_segmentation"].shape)

    # -------------------------------------------------------------------------
    # 13. Observation 저장
    # -------------------------------------------------------------------------
    save_one_observation_frame(
        output_dir=DAY2_DIR / "observations",
        camera=camera,
    )

    # -------------------------------------------------------------------------
    # 14. GUI 확인용 Simulation Loop
    # -------------------------------------------------------------------------
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()