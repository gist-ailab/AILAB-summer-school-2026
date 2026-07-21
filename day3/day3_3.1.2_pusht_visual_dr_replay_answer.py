#!/usr/bin/env python3
"""Day3-3.1.2 ANSWER: PushT visual domain randomization after re-rendering.

This script restores original PushT states and adds
visual domain randomization before camera re-rendering. It changes T-bar/goal/table/ground/light
appearance before re-rendering camera images.

Run from project root:
    $ISAACLAB_PATH/isaaclab.sh -p day3/day3_3.1.2_pusht_visual_dr_replay_answer.py \
        --input_file day3/datasets/tbar_pusht_teleop_practice.hdf5 \
        --output_file day3/datasets/pusht/pusht_visual_dr_test.hdf5 \
        --num_demos 3 \
        --copies_per_demo 1 \
        --enable_cameras

수업 연결:
- 3.1.1의 state restore + camera re-render 흐름을 그대로 재사용합니다.
- Day2 4교시 camera observation과 같은 카메라 활성화 방식을 사용합니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
import random
import re
import sys
import time
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DAY3_ROOT.parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apply visual domain randomization while re-rendering PushT demos.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/tbar_pusht_teleop_practice.hdf5"))
parser.add_argument("--output_file", default=str(DAY3_ROOT / "datasets/pusht/pusht_visual_dr_test.hdf5"))
parser.add_argument("--num_demos", type=int, default=3, help="0 means replay all demos.")
parser.add_argument("--copies_per_demo", type=int, default=1, help="Number of visual variants to replay per source demo.")
parser.add_argument("--camera_width", type=int, default=640, help="Output camera width.")
parser.add_argument("--camera_height", type=int, default=480, help="Output camera height.")
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--progress_interval", type=int, default=50, help="Print replay progress every N steps. 0 disables it.")
# Day2 3/4교시와 같은 AppLauncher 인자 추가 패턴입니다.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 / 데이터 처리 라이브러리 임포트
# ============================================================================
import carb  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from task.lift.custom_pusht_env_cfg_3_2_answer import PushTEnvCfg  # noqa: E402
from day3_3_utils import copy_h5_attrs, copy_h5_item, delete_if_exists, ensure_h5_data_attrs, selected_demo_names  # noqa: E402

# 카메라 렌더링 활성화 (--enable_cameras flag 대체)
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# ============================================================================
# 3. 핵심 함수 및 문제 코드
# ============================================================================

# USD stage의 하위 prim을 재귀적으로 순회합니다.
def _iter_prims(root_prim):
    yield root_prim
    for child in root_prim.GetAllChildren():
        yield from _iter_prims(child)


# USD prim의 material 색상 속성을 지정합니다.
def _set_color(prim, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    vec = Gf.Vec3f(*color)
    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([vec])
    shader = UsdShade.Shader(prim)
    if shader:
        for input_name in ("diffuseColor", "diffuse_color_constant"):
            inp = shader.GetInput(input_name)
            if inp:
                inp.Set(vec)
            else:
                shader.CreateInput(input_name, Sdf.ValueTypeNames.Color3f).Set(vec)


# 정규식과 맞는 USD prim들을 찾아 visual DR 대상으로 모읍니다.
def collect_prims_by_regex(stage, path_regex: str) -> list:
    pattern = re.compile(path_regex)
    prims = []
    for prim in stage.Traverse():
        if pattern.fullmatch(str(prim.GetPath())):
            prims.extend(list(_iter_prims(prim)))
    return prims


# 찾아둔 여러 prim에 같은 색을 적용합니다.
def set_color_on_prims(prims: list, color: tuple[float, float, float]) -> int:
    for prim in prims:
        _set_color(prim, color)
    return len(prims)


# Dome light의 색과 세기를 설정해 조명 domain randomization을 적용합니다.
def set_dome_light(stage, path: str, color: tuple[float, float, float], intensity: float) -> None:
    from pxr import Gf, UsdLux

    prim = stage.GetPrimAtPath(path)
    if prim:
        light = UsdLux.DomeLight(prim)
        if light:
            light.CreateColorAttr().Set(Gf.Vec3f(*color))
            light.CreateIntensityAttr().Set(float(intensity))


# 한 episode/copy에 적용할 색상과 조명 값을 무작위로 샘플링합니다.
def sample_visual_style(rng: random.Random) -> dict[str, object]:
    return {
        "object_color": (rng.uniform(0.55, 1.0), rng.uniform(0.02, 0.30), rng.uniform(0.02, 0.30)),
        "target_color": (rng.uniform(0.02, 0.30), rng.uniform(0.55, 1.0), rng.uniform(0.02, 0.30)),
        "table_color": (rng.uniform(0.35, 0.9), rng.uniform(0.35, 0.9), rng.uniform(0.35, 0.9)),
        "ground_color": (rng.uniform(0.45, 0.9), rng.uniform(0.45, 0.9), rng.uniform(0.45, 0.9)),
        "light_color": (rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0)),
        "light_intensity": rng.uniform(1800.0, 4200.0),
    }


# PushT scene에서 색을 바꿀 object/table/ground/light prim들을 미리 찾습니다.
def build_visual_dr_targets():
    from omni.usd import get_context

    stage = get_context().get_stage()
    targets = {
        "object": collect_prims_by_regex(stage, r"/World/envs/env_[0-9]+/object_0.*"),
        "target": collect_prims_by_regex(stage, r"/World/envs/env_[0-9]+/target_object.*"),
        "table": collect_prims_by_regex(stage, r"/World/envs/env_[0-9]+/Table.*"),
        "ground": collect_prims_by_regex(stage, r"/World/GroundPlane.*"),
        "stage": stage,
    }
    print(
        "[DR] cached prim counts "
        f"object={len(targets['object'])}, target={len(targets['target'])}, "
        f"table={len(targets['table'])}, ground={len(targets['ground'])}"
    )
    return targets


# 샘플링한 visual style을 실제 USD stage와 조명에 적용합니다.
def apply_visual_dr(env, rng: random.Random, targets: dict[str, object]) -> dict[str, object]:
    style = sample_visual_style(rng)
    set_color_on_prims(targets["object"], style["object_color"])
    set_color_on_prims(targets["target"], style["target_color"])
    set_color_on_prims(targets["table"], style["table_color"])
    set_color_on_prims(targets["ground"], style["ground_color"])
    set_dome_light(targets["stage"], "/World/light", style["light_color"], style["light_intensity"])
    env.sim.render()
    return style


# HDF5 group/dataset을 IsaacLab state 복원에 쓸 torch tensor tree로 변환합니다.
def h5_tree_to_torch(item, device: str, step: int | None = None):
    if isinstance(item, h5py.Group):
        return {key: h5_tree_to_torch(item[key], device, step) for key in item.keys()}
    if step is None:
        data = item[()]
    else:
        data = item[step : step + 1]
    return torch.as_tensor(data, device=device)


# 저장된 simulator state를 복원한 뒤 카메라 관측을 다시 계산합니다.
def restore_state_and_render(env: ManagerBasedEnv, state, env_ids: torch.Tensor):
# 3.1.1에서 다룬 핵심 패턴: 저장된 simulator state를 그대로 복원합니다.
    env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
    env.sim.forward()
    env.sim.render()
    return env.observation_manager.compute_group("policy", update_history=False)


# PushT visual DR replay에 사용할 환경과 recorder 설정을 만듭니다.
def make_env(output_file: str) -> ManagerBasedEnv:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = PushTEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.camera.width = args_cli.camera_width
    env_cfg.scene.camera.height = args_cli.camera_height
    env_cfg.scene.top_camera.width = args_cli.camera_width
    env_cfg.scene.top_camera.height = args_cli.camera_height
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
    env_cfg.recorders = None

# Day2 3교시부터 사용한 ManagerBasedEnv 생성 방식입니다.
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.4, 0.0, 0.5])
    return env


# ============================================================================
# 4. 메인 함수
# ============================================================================
# PushT visual DR replay 전체 파이프라인을 실행합니다.
def main() -> None:
    rng = random.Random()
    env = make_env(args_cli.output_file)
    visual_targets = build_visual_dr_targets()
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)

    input_path = Path(args_cli.input_file)
    output_path = Path(args_cli.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    exported = 0
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_h5_attrs(src, dst)
        src_data = src["data"]
        dst_data = dst.create_group("data")
        ensure_h5_data_attrs(dst_data, src_data, env_name="Template-PushT-Franka-v0")
        names = selected_demo_names(src_data, args_cli.num_demos)
        expected = len(names) * args_cli.copies_per_demo
        print(
            f"[START] source_demos={len(names)}, copies_per_demo={args_cli.copies_per_demo}, "
            f"mode=state_rerender, expected={expected}",
            flush=True,
        )

        for name in names:
            src_demo = src_data[name]
            if "states" not in src_demo:
                print(f"[SKIP] {name}: no states group", flush=True)
                continue
            num_steps = min(int(src_demo["actions"].shape[0]), args_cli.max_steps)

            for copy_idx in range(args_cli.copies_per_demo):
                if not simulation_app.is_running():
                    break

                demo_out_name = f"demo_{exported}"
                dst_demo = dst_data.create_group(demo_out_name)
                copy_h5_attrs(src_demo, dst_demo)
                dst_demo.attrs["source_demo"] = name
                dst_demo.attrs["source_copy_index"] = copy_idx
                dst_demo.attrs["augmentation"] = "pusht_visual_domain_randomization_state_rerender"

                for key in src_demo.keys():
                    copy_h5_item(src_demo, dst_demo, key)

                style = apply_visual_dr(env, rng, visual_targets)
                for style_key, style_value in style.items():
                    dst_demo.attrs[f"visual_dr/{style_key}"] = style_value

                top_frames = []
                wrist_frames = []
                t0 = time.perf_counter()
                for step_idx in range(num_steps):
                    state = h5_tree_to_torch(src_demo["states"], env.device, step=step_idx)
                    obs = restore_state_and_render(env, state, env_ids)
                    top_frames.append(obs["top_cam"][0].detach().cpu().numpy())
                    wrist_frames.append(obs["wrist_cam"][0].detach().cpu().numpy())

                    if args_cli.progress_interval > 0 and step_idx > 0 and step_idx % args_cli.progress_interval == 0:
                        print(f"[STEP] {name}/copy_{copy_idx}: {step_idx}/{num_steps}", flush=True)

                obs_group = dst_demo["obs"]
                delete_if_exists(obs_group, "top_cam")
                delete_if_exists(obs_group, "wrist_cam")
                obs_group.create_dataset("top_cam", data=top_frames, compression="gzip")
                obs_group.create_dataset("wrist_cam", data=wrist_frames, compression="gzip")

                exported += 1
                dt = time.perf_counter() - t0
                print(
                    f"[SUCCESS] {name}/copy_{copy_idx} -> {demo_out_name}, "
                    f"frames={num_steps}, render+write={dt:.2f}s, style={style}",
                    flush=True,
                )
                dst_data.attrs["total"] = sum(int(demo["actions"].shape[0]) for demo in dst_data.values() if "actions" in demo)
                dst.flush()

    env.close()
    print(f"[DONE] exported {exported} visual-DR demos to {args_cli.output_file}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
