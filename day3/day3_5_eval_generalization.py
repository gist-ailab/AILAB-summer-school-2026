#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# day3_5 · 일반화 성능 평가 (Generalization Eval)
#
# 텔레옵 vs 증강 데이터셋으로 학습한 모델의 일반화 성능을 비교합니다.
#
# 증강 전략별 평가:
#   PushT      → Visual DR (색상/조명 랜덤화) 일반화 테스트
#   PickPlace  → Trajectory 증강 (spawn 범위 확대) 일반화 테스트
#
# 사용법:
#   # PushT: Visual DR 일반화
#   $ISAACLAB_PATH/isaaclab.sh -p day3_5_eval_generalization.py \
#       --task_type pusht --visual_dr \
#       --checkpoint <체크포인트.pth> --num_rollouts 20
#
#   # PickPlace: 넓은 spawn 범위 일반화
#   $ISAACLAB_PATH/isaaclab.sh -p day3_5_eval_generalization.py \
#       --task_type pickplace --spawn_range wide \
#       --checkpoint <체크포인트.pth> --num_rollouts 20
# =====================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generalization eval for Diffusion Policy.")
parser.add_argument("--task_type", type=str, default="pusht", choices=["pickplace", "pusht"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth).")
parser.add_argument("--num_rollouts", type=int, default=20, help="Number of eval rollouts.")
parser.add_argument("--max_steps", type=int, default=300, help="Max steps per rollout.")
parser.add_argument("--spawn_range", type=str, default="original",
                    choices=["original", "wide", "extreme"],
                    help="Spawn randomization range for initial object placement.")
parser.add_argument("--visual_dr", action="store_true",
                    help="Enable visual domain randomization (color/lighting).")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=960)
parser.add_argument("--video_fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import sys
import os
import json
import random
import re
import numpy as np
import torch
from PIL import Image

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sim import PinholeCameraCfg

# Add day3/ and day3/task/ to path
_day3_dir = os.path.abspath(os.path.dirname(__file__))
if _day3_dir not in sys.path:
    sys.path.insert(0, _day3_dir)
_task_dir = os.path.join(_day3_dir, "task")
if _task_dir not in sys.path:
    sys.path.insert(0, _task_dir)

# =====================================================================
# Spawn Randomization Presets
# =====================================================================
# "original": 기존 데이터 수집과 동일한 범위
# "wide":     mimic augmentation에서 사용한 확장 범위
# "extreme":  wide보다 더 넓은 범위 (학습 범위 외 테스트)

PUSHT_SPAWN_PRESETS = {
    "original": {
        # 기존 reset_tbar_left_right: y=±(0.2~0.3), x=0.4±0.05, yaw=±π/2
        "x_center": 0.4, "x_range": 0.05,
        "y_min": 0.2, "y_max": 0.3,
        "yaw_range": math.pi / 2,
    },
    "wide": {
        # mimic augmentation에서 사용한 범위
        "x_center": 0.4, "x_range": 0.15,
        "y_min": 0.1, "y_max": 0.35,
        "yaw_range": math.pi,
    },
    "extreme": {
        # Out of Distribution
        "x_center": 0.4, "x_range": 0.25,
        "y_min": 0.05, "y_max": 0.4,
        "yaw_range": math.pi,
    },
}

PICKPLACE_SPAWN_PRESETS = {
    "original": {
        "object_x": (-0.1, 0.1), "object_y": (-0.1, 0.1),
        "object_yaw": (-math.pi / 4, math.pi / 4),
        "bin_pose_range": None,
    },
    "wide": {
        "object_x": (-0.18, 0.18), "object_y": (-0.18, 0.18),   
        "object_yaw": (-math.pi / 4, 3 * math.pi / 4),
        "bin_pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "yaw": (-math.pi / 6, math.pi / 6)},
    },
    "extreme": {
        "object_x": (-0.25, 0.25), "object_y": (-0.25, 0.25),
        "object_yaw": (-math.pi / 2, math.pi),
        "bin_pose_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "yaw": (-math.pi / 4, math.pi / 4)},
    },
}

    
def get_success_fn(task_type: str):
    """task_type에 맞는 성공 판정 함수를 불러옵니다."""
    if task_type == "pickplace":
        from task.lift.mdp_3_1.terminations import object_pickplace_goal
        return object_pickplace_goal
    elif task_type == "pusht":
        from task.lift.mdp_3_2.terminations_answer import object_pusht_goal

        # 성공 기준 완화
        import functools
        return functools.partial(object_pusht_goal, pos_threshold=0.05, yaw_threshold=0.15)
    return None


def get_env_cfg(task_type: str):
    """환경 설정 클래스를 불러옵니다."""
    import importlib
    if task_type == "pickplace":
        mod = importlib.import_module("task.lift.config.ik_abs_env_cfg_3_1_answer")
        return mod.FrankaTBarPickPlaceEnvCfg
    elif task_type == "pusht":
        mod = importlib.import_module("task.lift.custom_pusht_env_cfg_3_2_answer")
        return mod.PushTEnvCfg
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def load_policy(checkpoint_path: str, device: torch.device):
    """Robomimic 체크포인트에서 policy를 로드합니다."""
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path, device=device, verbose=True,
    )
    return policy


def obs_to_robomimic(obs_policy: dict, device: torch.device, frame_stack: int = 2) -> dict:
    """Isaac Lab obs → diffusion policy batched format."""
    obs_batched = {}
    for key, val in obs_policy.items():
        if val.ndim == 1:
            v = val.float().to(device)
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, -1)
        elif val.ndim == 3:
            v = val.float().to(device)
            if v.shape[-1] == 4:
                v = v[..., :3]
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, *v.shape)
        else:
            v = val.float().to(device)
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, *val.shape)
        obs_batched[key] = v
    return obs_batched


# =====================================================================
# Spawn Randomization (에피소드 시작 시 오브젝트 위치 재배치)
# =====================================================================

def randomize_pusht_spawn(env, preset_name: str):
    """PushT 환경의 T-bar 초기 위치를 지정된 범위로 랜덤화합니다."""
    preset = PUSHT_SPAWN_PRESETS[preset_name]
    obj = env.scene["object_0"]
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
    state = obj.data.default_root_state[env_ids].clone()

    # Position
    x_noise = (torch.rand(1, device=env.device) - 0.5) * 2 * preset["x_range"]
    state[:, 0] = preset["x_center"] + x_noise

    left_right = torch.randint(0, 2, (1,), device=env.device) * 2 - 1
    y_offset = left_right * (preset["y_min"] + torch.rand(1, device=env.device) * (preset["y_max"] - preset["y_min"]))
    state[:, 1] = 0.0 + y_offset

    # Yaw
    yaw = (torch.rand(1, device=env.device) - 0.5) * 2 * preset["yaw_range"]
    state[:, 3] = torch.cos(yaw / 2)
    state[:, 4] = 0.0
    state[:, 5] = 0.0
    state[:, 6] = torch.sin(yaw / 2)

    obj.write_root_pose_to_sim(state[:, :7], env_ids)
    obj.write_root_velocity_to_sim(state[:, 7:], env_ids)


def randomize_pickplace_spawn(env, preset_name: str):
    """PickPlace 환경의 T-bar + bin 초기 위치를 지정된 범위로 랜덤화합니다."""
    preset = PICKPLACE_SPAWN_PRESETS[preset_name]
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)

    # Object (T-bar)
    obj = env.scene["object_0"]
    obj_state = obj.data.default_root_state[env_ids].clone()
    ox = torch.FloatTensor(1).uniform_(*preset["object_x"]).to(env.device)
    oy = torch.FloatTensor(1).uniform_(*preset["object_y"]).to(env.device)
    obj_state[:, 0] += ox
    obj_state[:, 1] += oy

    yaw = torch.FloatTensor(1).uniform_(*preset["object_yaw"]).to(env.device)
    obj_state[:, 3] = torch.cos(yaw / 2)
    obj_state[:, 6] = torch.sin(yaw / 2)
    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

    # Bin (x/y + yaw만 적용)
    # NOTE: bin의 default quat이 identity가 아니므로 (X축 90° 회전),
    #       yaw를 적용할 때 기존 quaternion과 합성해야 합니다.
    bin_range = preset.get("bin_pose_range")
    if bin_range is not None:
        bin_obj = env.scene["bin"]
        bin_state = bin_obj.data.default_root_state[env_ids].clone()
        bx = torch.FloatTensor(1).uniform_(*bin_range["x"]).to(env.device)
        by = torch.FloatTensor(1).uniform_(*bin_range["y"]).to(env.device)
        bin_state[:, 0] += bx
        bin_state[:, 1] += by
        if "yaw" in bin_range:
            bin_yaw = torch.FloatTensor(1).uniform_(*bin_range["yaw"]).to(env.device)
            # yaw quaternion: (cos(yaw/2), 0, 0, sin(yaw/2))
            yaw_quat = torch.zeros(1, 4, device=env.device)
            yaw_quat[:, 0] = torch.cos(bin_yaw / 2)  # qw
            yaw_quat[:, 3] = torch.sin(bin_yaw / 2)  # qz
            # 기존 default quat과 합성 (yaw * default)
            from isaaclab.utils.math import quat_mul
            default_quat = bin_state[:, 3:7].clone()  # (qw, qx, qy, qz)
            bin_state[:, 3:7] = quat_mul(yaw_quat, default_quat)
        bin_obj.write_root_pose_to_sim(bin_state[:, :7], env_ids)
        bin_obj.write_root_velocity_to_sim(bin_state[:, 7:], env_ids)


# =====================================================================
# Visual Domain Randomization (USD prim 색상/조명 변경)
# =====================================================================

def _iter_prims(root_prim):
    yield root_prim
    for child in root_prim.GetAllChildren():
        yield from _iter_prims(child)


def _set_color(prim, color):
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


def collect_prims_by_regex(stage, path_regex: str):
    pattern = re.compile(path_regex)
    prims = []
    for prim in stage.Traverse():
        if pattern.fullmatch(str(prim.GetPath())):
            prims.extend(list(_iter_prims(prim)))
    return prims


def set_color_on_prims(prims, color):
    for prim in prims:
        _set_color(prim, color)


def set_dome_light(stage, path, color, intensity):
    from pxr import Gf, UsdLux
    prim = stage.GetPrimAtPath(path)
    if prim:
        light = UsdLux.DomeLight(prim)
        if light:
            light.CreateColorAttr().Set(Gf.Vec3f(*color))
            light.CreateIntensityAttr().Set(float(intensity))


def build_visual_dr_targets():
    """USD stage에서 Visual DR 대상 prim들을 미리 수집합니다."""
    from omni.usd import get_context
    stage = get_context().get_stage()
    targets = {
        "object": collect_prims_by_regex(stage, r"/World/envs/env_[0-9]+/object_0.*"),
        "target": collect_prims_by_regex(stage, r"/World/envs/env_[0-9]+/target_object.*"),
        "table": collect_prims_by_regex(stage, r"/World/envs/env_[0-9]+/Table.*"),
        "ground": collect_prims_by_regex(stage, r"/World/GroundPlane.*"),
        "stage": stage,
    }
    print(f"[Visual DR] cached prim counts: "
          f"object={len(targets['object'])}, target={len(targets['target'])}, "
          f"table={len(targets['table'])}, ground={len(targets['ground'])}")
    return targets


def apply_visual_dr(env, rng, targets):
    """에피소드마다 색상/조명을 랜덤화합니다."""
    style = {
        "object_color": (rng.uniform(0.55, 1.0), rng.uniform(0.02, 0.30), rng.uniform(0.02, 0.30)),
        "target_color": (rng.uniform(0.02, 0.30), rng.uniform(0.55, 1.0), rng.uniform(0.02, 0.30)),
        "table_color": (rng.uniform(0.35, 0.9), rng.uniform(0.35, 0.9), rng.uniform(0.35, 0.9)),
        "ground_color": (rng.uniform(0.45, 0.9), rng.uniform(0.45, 0.9), rng.uniform(0.45, 0.9)),
        "light_color": (rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0)),
        "light_intensity": rng.uniform(1800.0, 4200.0),
    }
    set_color_on_prims(targets["object"], style["object_color"])
    set_color_on_prims(targets["target"], style["target_color"])
    set_color_on_prims(targets["table"], style["table_color"])
    set_color_on_prims(targets["ground"], style["ground_color"])
    set_dome_light(targets["stage"], "/World/light", style["light_color"], style["light_intensity"])
    env.sim.render()
    return style


# =====================================================================
# Main
# =====================================================================

def main():
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    rng = random.Random()

    # ---- Load policy ----
    print(f"\n[GEN-EVAL] Loading checkpoint: {args_cli.checkpoint}")
    policy = load_policy(args_cli.checkpoint, device)

    # ---- Create environment ----
    EnvCfgClass = get_env_cfg(args_cli.task_type)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.concatenate_terms = False

    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    # Match obs camera resolution
    TRAIN_H, TRAIN_W = 480, 640
    for cam_name in ("camera", "top_camera"):
        if hasattr(env_cfg.scene, cam_name):
            cam_cfg = getattr(env_cfg.scene, cam_name)
            if hasattr(cam_cfg, "height"):
                cam_cfg.height = TRAIN_H
                cam_cfg.width = TRAIN_W

    # Add video camera
    video_camera_name = "video_camera"
    VIDEO_H, VIDEO_W = args_cli.video_height, args_cli.video_width
    if hasattr(env_cfg.scene, "top_camera"):
        ref_cam = env_cfg.scene.top_camera
        video_offset = ref_cam.offset
        video_prim = ref_cam.prim_path.replace("top_camera", "video_camera")
    else:
        video_offset = CameraCfg.OffsetCfg(
            pos=(0.4, 0.0, 2.5), rot=(-0.7071068, 0, -0.7071068, 0), convention="world")
        video_prim = "{ENV_REGEX_NS}/video_camera"

    env_cfg.scene.video_camera = CameraCfg(
        prim_path=video_prim,
        update_period=0.0,
        height=VIDEO_H, width=VIDEO_W,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
        ),
        offset=video_offset,
    )

    env = ManagerBasedEnv(cfg=env_cfg)

    # Success function
    success_fn = get_success_fn(args_cli.task_type)

    # Visual DR targets
    visual_targets = None
    if args_cli.visual_dr:
        visual_targets = build_visual_dr_targets()

    # Video recording
    save_video = True
    try:
        import imageio
    except ImportError:
        print("[WARN] imageio not installed, disabling video recording.")
        save_video = False

    # Output directory
    ckpt_abs = os.path.abspath(args_cli.checkpoint)
    ckpt_parent = os.path.dirname(ckpt_abs)
    if os.path.basename(ckpt_parent) == "models":
        run_dir = os.path.dirname(ckpt_parent)
    else:
        run_dir = ckpt_parent

    dr_tag = "visual_dr" if args_cli.visual_dr else "no_dr"
    eval_dir = os.path.join(run_dir, f"eval_gen_{args_cli.task_type}_{args_cli.spawn_range}_{dr_tag}")
    video_dir = os.path.join(eval_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # ---- Print eval config ----
    print(f"\n{'='*60}")
    print(f"  [GEN-EVAL] 일반화 성능 평가")
    print(f"  Task:           {args_cli.task_type}")
    print(f"  Spawn range:    {args_cli.spawn_range}")
    print(f"  Visual DR:      {'ON' if args_cli.visual_dr else 'OFF'}")
    print(f"  Num rollouts:   {args_cli.num_rollouts}")
    print(f"  Max steps:      {args_cli.max_steps}")
    print(f"  Output:         {eval_dir}")
    print(f"{'='*60}\n")

    # ---- Run rollouts ----
    success_count = 0
    total_rewards = []
    rollout_results = []

    for ep in range(args_cli.num_rollouts):
        obs_full, _ = env.reset()

        # 초기 위치 랜덤화 (env.reset 이후 오브젝트 위치 덮어쓰기)
        if args_cli.task_type == "pusht":
            randomize_pusht_spawn(env, args_cli.spawn_range)
        elif args_cli.task_type == "pickplace":
            randomize_pickplace_spawn(env, args_cli.spawn_range)

        # sim forward를 다시 호출하여 randomize한 위치 반영
        env.sim.forward()
        env.sim.render()

        # Visual DR 적용
        if args_cli.visual_dr and visual_targets is not None:
            style = apply_visual_dr(env, rng, visual_targets)
        else:
            style = None

        # obs 재계산 (랜덤화/DR 반영)
        obs_full = {"policy": env.observation_manager.compute_group("policy", update_history=False)}

        policy.start_episode()
        ep_reward = 0.0
        ep_success = False
        frames = []

        for step in range(args_cli.max_steps):
            with torch.no_grad():
                obs_policy = obs_full["policy"]
                obs_single = {k: v[0] for k, v in obs_policy.items()}
                obs_batched = obs_to_robomimic(obs_single, device)

                if ep == 0 and step == 0:
                    print(f"[DEBUG] Observation shapes:")
                    for k, v in obs_single.items():
                        print(f"  {k}: {tuple(v.shape)}")

                action = policy(obs_batched, batched_ob=True)
                action = action[0]

                if isinstance(action, np.ndarray):
                    action_tensor = torch.from_numpy(action).float().to(device)
                else:
                    action_tensor = action.float().to(device)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                actions = action_tensor.repeat(env.num_envs, 1)

                obs_full = env.step(actions)[0]

                # Video frame
                if save_video:
                    cam = env.scene[video_camera_name]
                    raw_frame = cam.data.output["rgb"][0].cpu().numpy()
                    if raw_frame.shape[-1] == 4:
                        raw_frame = raw_frame[..., :3]
                    if raw_frame.dtype != np.uint8:
                        fmin, fmax = raw_frame.min(), raw_frame.max()
                        if fmax - fmin > 1e-6:
                            raw_frame = (raw_frame - fmin) / (fmax - fmin)
                        else:
                            raw_frame = np.zeros_like(raw_frame)
                        raw_frame = (raw_frame * 255).clip(0, 255).astype(np.uint8)
                    frames.append(raw_frame)

                # Success check
                is_success = False
                if success_fn is not None:
                    try:
                        result = success_fn(env)
                        if isinstance(result, torch.Tensor):
                            is_success = bool(result.reshape(-1)[0].item())
                        else:
                            is_success = bool(result)
                    except Exception:
                        pass
                if is_success:
                    print(f"    ✓ SUCCESS @ step {step+1}")
                    ep_success = True
                    break

        if ep_success:
            success_count += 1

        status = "✓ SUCCESS" if ep_success else "✗ FAIL"
        print(f"  Rollout {ep+1:3d}/{args_cli.num_rollouts} | "
              f"Steps: {step+1:4d} | {status}")

        rollout_results.append({
            "rollout": ep + 1,
            "steps": step + 1,
            "success": ep_success,
            "visual_dr_style": style if style else None,
        })

        # Save video
        if save_video and len(frames) > 0:
            vid_path = os.path.join(video_dir, f"rollout_{ep:03d}.mp4")
            imageio.mimsave(vid_path, frames, fps=args_cli.video_fps)

    # ---- Summary ----
    success_rate = success_count / args_cli.num_rollouts * 100

    summary = {
        "checkpoint": ckpt_abs,
        "task": args_cli.task_type,
        "spawn_range": args_cli.spawn_range,
        "visual_dr": args_cli.visual_dr,
        "num_rollouts": args_cli.num_rollouts,
        "max_steps": args_cli.max_steps,
        "success_count": success_count,
        "success_rate": success_rate,
        "rollouts": rollout_results,
    }

    print(f"\n{'='*60}")
    print(f"  일반화 성능 평가 결과")
    print(f"  Checkpoint: {os.path.basename(args_cli.checkpoint)}")
    print(f"  Task:         {args_cli.task_type}")
    print(f"  Spawn range:  {args_cli.spawn_range}")
    print(f"  Visual DR:    {'ON' if args_cli.visual_dr else 'OFF'}")
    print(f"  Success Rate: {success_count}/{args_cli.num_rollouts} ({success_rate:.1f}%)")
    print(f"{'='*60}")

    # Save JSON
    results_path = os.path.join(eval_dir, "eval_generalization_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    if save_video:
        print(f"  Videos:  {video_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
