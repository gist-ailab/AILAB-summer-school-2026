#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# [ANSWER] day3_5 · Diffusion Policy 평가 (Eval)
#
# 학습된 Diffusion Policy 체크포인트를 Isaac Lab 환경에서 평가합니다.
# - 체크포인트 로드 → 환경 생성 → rollout 실행 → 성공률/보상 집계
# - Isaac Lab obs를 robomimic 형식으로 변환하는 파이프라인 포함
# - 비디오 녹화 및 JSON 결과 저장 지원
# =====================================================================
"""
Evaluate a trained Diffusion Policy checkpoint in Isaac Lab environments.

Directly instantiates the env cfg without gymnasium registration.

Usage (from day3/):
    <ISAACLAB_PATH>/isaaclab.sh -p day3_5_eval_answer.py \
        --task_type pusht \
        --checkpoint <path_to_checkpoint.pth> \
        --num_rollouts 20 
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy in Isaac Lab environment.")
parser.add_argument("--task_type", type=str, default="pusht", choices=["pickplace", "pusht"],
                    help="Task type: 'pickplace' or 'pusht'.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (use 1 for eval).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to robomimic checkpoint (.pth).")
parser.add_argument("--num_rollouts", type=int, default=20, help="Number of evaluation rollouts.")
parser.add_argument("--max_steps", type=int, default=300, help="Max steps per rollout.")
parser.add_argument("--video_width", type=int, default=1280, help="Video recording width (default: 1280).")
parser.add_argument("--video_height", type=int, default=960, help="Video recording height (default: 960).")
parser.add_argument("--video_camera", type=str, default=None,
                    help="Camera name for video recording (auto-detected if not set).")
parser.add_argument("--video_fps", type=int, default=30, help="Video FPS (default: 30).")
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
import numpy as np
import torch
from PIL import Image

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sim import PinholeCameraCfg

import functools

# Add day3/ and day3/task/ to path so env configs with 'from task.lift.xxx' work
_day3_dir = os.path.abspath(os.path.dirname(__file__))
if _day3_dir not in sys.path:
    sys.path.insert(0, _day3_dir)
_task_dir = os.path.join(_day3_dir, "task")
if _task_dir not in sys.path:
    sys.path.insert(0, _task_dir)


def get_success_fn(task_type: str):
    """task_type에 맞는 성공 판정 함수를 불러옵니다."""
    if task_type == "pickplace":
        from task.lift.mdp_3_1.terminations import object_pickplace_goal
        return object_pickplace_goal
    elif task_type == "pusht":
        from task.lift.mdp_3_2.terminations_answer import object_pusht_goal
        # 성공 기준 조정
        return functools.partial(object_pusht_goal, pos_threshold=0.05, yaw_threshold=0.15)
    return None
    

def load_policy(checkpoint_path: str, device: torch.device):
    """
    Load a trained robomimic Diffusion Policy from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Torch device.

    Returns:
        policy: robomimic PolicyAlgo instance in eval mode.
    """
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=True,
    )
    policy.start_episode()
    return policy




def obs_to_robomimic(obs_policy: dict, device: torch.device, frame_stack: int = 2) -> dict:
    """
    Convert Isaac Lab observations (from env.step) to robomimic batched format.

    Camera resolution is already set to 240x320 (matching training data),
    so no image resize is needed here. Raw float values are preserved.
    robomimic DiffusionPolicy expects (B, T, ...) inputs.

    Args:
        obs_policy: dict of observation tensors from env (first env only).
        device: torch device.
        frame_stack: number of frames to stack (must match config.train.frame_stack).

    Returns:
        obs_batched: dict ready for robomimic policy call with batched_ob=True.
    """
    obs_batched = {}
    for key, val in obs_policy.items():
        if val.ndim == 1:
            # Low-dim: (dim,) -> (1, T, dim)
            v = val.float().to(device)
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, -1)
        elif val.ndim == 3:
            # Image: (H, W, C) -> keep raw values (same format as training data)
            v = val.float().to(device)
            # Drop alpha if present
            if v.shape[-1] == 4:
                v = v[..., :3]
            # (H,W,3) -> (1, T, H, W, 3)
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, *v.shape)
        else:
            # Fallback: just add batch+time dims
            v = val.float().to(device)
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, *val.shape)
        obs_batched[key] = v
    return obs_batched


def get_env_cfg(task_type: str):
    """Import and return the env cfg class based on task type.

    Uses standard Python imports via importlib.import_module.
    Requires __init__.py in task/, task/lift/, task/lift/config/.
    """
    import importlib

    if task_type == "pickplace":
        mod = importlib.import_module("task.lift.config.ik_abs_env_cfg_3_1_answer")
        return mod.FrankaTBarPickPlaceEnvCfg
    elif task_type == "pusht":
        mod = importlib.import_module("task.lift.custom_pusht_env_cfg_3_2_answer")
        return mod.PushTEnvCfg
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main():
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda:0")

    # ---- Load policy ----
    print(f"[EVAL] Loading checkpoint: {args_cli.checkpoint}")
    policy = load_policy(args_cli.checkpoint, device)
    print(f"[EVAL] Policy loaded successfully.")

    # ---- Create environment (without gym registration) ----
    EnvCfgClass = get_env_cfg(args_cli.task_type)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    # Set observations to dictionary mode for robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Extract success term BEFORE creating env, then remove it + timeout
    # so the environment doesn't auto-terminate -- we control the loop.
    success_term = None
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

    # Disable recorder if present
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    # ---- Match obs camera resolution to training data ----
    TRAIN_H, TRAIN_W = 480, 640
    for cam_name in ("camera", "top_camera"):
        if hasattr(env_cfg.scene, cam_name):
            cam_cfg = getattr(env_cfg.scene, cam_name)
            if hasattr(cam_cfg, "height"):
                cam_cfg.height = TRAIN_H
                cam_cfg.width = TRAIN_W
                print(f"[EVAL] Obs camera '{cam_name}' → {TRAIN_H}x{TRAIN_W}")

    # ---- Add dedicated high-res video camera ----
    # Separate from obs cameras so video quality is independent of training resolution.
    video_camera_name = "video_camera"
    VIDEO_H, VIDEO_W = args_cli.video_height, args_cli.video_width
    # Copy position/orientation from top_camera (or fallback)
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
    print(f"[EVAL] Video camera: {video_camera_name} → {VIDEO_H}x{VIDEO_W} (native)")

    env = ManagerBasedEnv(cfg=env_cfg)

    # 성공 판정 함수 로드 (mdp 모듈에서)
    success_fn = get_success_fn(args_cli.task_type)

    print(f"[EVAL] Environment created: {args_cli.task_type}")
    if success_term:
        print(f"[EVAL] Success criterion: {success_term.func.__name__}"
              f" (params: {success_term.params})")
    else:
        print(f"[EVAL] No success criterion found (reward-only eval)")
    if video_camera_name:
        print(f"[EVAL] Video camera: {video_camera_name} ({args_cli.video_width}x{args_cli.video_height})")

    # ---- Setup eval output directory ----
    ckpt_abs = os.path.abspath(args_cli.checkpoint)
    ckpt_parent = os.path.dirname(ckpt_abs)
    if os.path.basename(ckpt_parent) == "models":
        run_dir = os.path.dirname(ckpt_parent)
    else:
        run_dir = ckpt_parent

    # epoch 번호 추출 (model_epoch_50.pth → epoch_50)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_abs))[0]  # "model_epoch_50"
    eval_dir = os.path.join(run_dir, f"eval_{args_cli.task_type}_{ckpt_stem}")
    video_dir = os.path.join(eval_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Check imageio for video recording
    save_video = video_camera_name is not None
    if save_video:
        try:
            import imageio
        except ImportError:
            print("[WARN] imageio not installed, disabling video recording.")
            save_video = False

    print(f"[EVAL] Eval output: {eval_dir}")
    if video_camera_name:
        print(f"[EVAL] Video camera: {video_camera_name}")

    # ---- Run rollouts ----
    success_count = 0
    total_rewards = []
    rollout_results = []

    for ep in range(args_cli.num_rollouts):
        obs_full, _ = env.reset()
        policy.start_episode()

        ep_reward = 0.0
        ep_success = False
        frames = []

        for step in range(args_cli.max_steps):
            with torch.no_grad():
                # Get observations from env (dict mode)
                obs_policy = obs_full["policy"]
                # Use first env only (index 0)
                obs_single = {k: v[0] for k, v in obs_policy.items()}

                # Debug: print obs info on first step of first rollout
                if ep == 0 and step == 0:
                    print(f"\n[DEBUG] === Observation Debug Info ===")
                    for k, v in obs_single.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                              f"min={v.min().item():.4f}, max={v.max().item():.4f}")

                # Convert to robomimic batched format
                obs_batched = obs_to_robomimic(obs_single, device)

                if ep == 0 and step == 0:
                    print(f"[DEBUG] === After obs_to_robomimic ===")
                    for k, v in obs_batched.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                              f"min={v.min().item():.4f}, max={v.max().item():.4f}")

                # Run policy
                action = policy(obs_batched, batched_ob=True)
                action = action[0]  # unbatch -> (ac_dim,)

                if ep == 0 and step == 0:
                    print(f"[DEBUG] === Action Output ===")
                    print(f"  action: shape={action.shape if hasattr(action, 'shape') else len(action)}, "
                          f"values={action}")

                # Convert to tensor and expand to all envs
                if isinstance(action, np.ndarray):
                    action_tensor = torch.from_numpy(action).float().to(device)
                else:
                    action_tensor = action.float().to(device)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                actions = action_tensor.repeat(env.num_envs, 1)

                obs_full = env.step(actions)[0]

                # Save frame for video (from dedicated high-res video camera)
                if save_video and video_camera_name:
                    cam = env.scene[video_camera_name]
                    raw_frame = cam.data.output["rgb"][0].cpu().numpy()
                    # Drop alpha if present
                    if raw_frame.shape[-1] == 4:
                        raw_frame = raw_frame[..., :3]
                    # Convert raw float to uint8 [0, 255] for video
                    if raw_frame.dtype != np.uint8:
                        fmin, fmax = raw_frame.min(), raw_frame.max()
                        if fmax - fmin > 1e-6:
                            raw_frame = (raw_frame - fmin) / (fmax - fmin)
                        else:
                            raw_frame = np.zeros_like(raw_frame)
                        raw_frame = (raw_frame * 255).clip(0, 255).astype(np.uint8)
                    frames.append(raw_frame)

                # Check success (mdp 모듈의 함수 사용)
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
                elif success_term is not None:
                    try:
                        is_success = bool(success_term.func(env, **success_term.params)[0])
                    except Exception:
                        pass
                if is_success:
                    print(f"    ✓ SUCCESS @ step {step+1}")
                    ep_success = True
                    break

        if ep_success:
            success_count += 1
        total_rewards.append(ep_reward)

        status = "\u2713 SUCCESS" if ep_success else "\u2717 FAIL"
        print(f"  Rollout {ep+1:3d}/{args_cli.num_rollouts} | "
              f"Steps: {step+1:4d} | Reward: {ep_reward:8.3f} | {status}")

        rollout_results.append({
            "rollout": ep + 1,
            "steps": step + 1,
            "reward": float(ep_reward),
            "success": ep_success,
        })

        # Save video
        if save_video and len(frames) > 0:
            vid_path = os.path.join(video_dir, f"rollout_{ep:03d}.mp4")
            imageio.mimsave(vid_path, frames, fps=args_cli.video_fps)

    # ---- Summary ----
    success_rate = success_count / args_cli.num_rollouts * 100
    avg_reward = np.mean(total_rewards)

    summary = {
        "checkpoint": ckpt_abs,
        "task": args_cli.task_type,
        "num_rollouts": args_cli.num_rollouts,
        "max_steps": args_cli.max_steps,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": float(avg_reward),
        "rollouts": rollout_results,
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Task:       {args_cli.task_type}")
    print(f"  Rollouts:     {args_cli.num_rollouts}")
    print(f"  Success Rate: {success_count}/{args_cli.num_rollouts} ({success_rate:.1f}%)")
    print(f"  Avg Reward:   {avg_reward:.3f}")

    # Save results JSON to eval dir
    results_path = os.path.join(eval_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results log:  {results_path}")

    if save_video:
        print(f"  Videos saved: {video_dir}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
