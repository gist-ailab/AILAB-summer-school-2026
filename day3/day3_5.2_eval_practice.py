#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# [PRACTICE 2/2] day3_5 · obs 변환 + 전체 Rollout 평가
#
# 5.1에서 환경을 생성한 것에 이어,
# Isaac Lab obs를 diffusion policy 형식으로 변환하고
# 전체 rollout을 실행하여 성공률을 측정합니다.
#
# 채워야 할 곳: 
#   TODO 1: Low-dim 관측값 shape 변환 (dim,) → (1, T, dim)
#   TODO 2: 이미지 관측값 shape 변환 (H,W,C) → (1, T, H, W, C)
#
# 실행하면: 여러 에피소드의 rollout 결과(성공률)가 출력됩니다.
# =====================================================================
"""
[5.2] obs 변환 + 전체 Rollout 평가

Isaac Lab 환경의 관측값을 diffusion policy가 기대하는 형식으로 변환하고,
전체 rollout을 실행하여 성공률을 측정합니다.

Usage:
    $ISAACLAB_PATH/isaaclab.sh -p day3_5.2_eval_practice.py \
        --task_type pusht \
        --checkpoint <체크포인트.pth> \
        --num_rollouts 10 \
        --max_steps 300
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="[5.2] obs conversion + full rollout evaluation.")
parser.add_argument("--task_type", type=str, default="pusht", choices=["pickplace", "pusht"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth).")
parser.add_argument("--num_rollouts", type=int, default=20, help="Number of evaluation rollouts.")
parser.add_argument("--max_steps", type=int, default=300, help="Max steps per rollout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import sys
import os
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from isaaclab.envs import ManagerBasedEnv

# Add day3/ and day3/task/ to sys.path
_day3_dir = os.path.abspath(os.path.dirname(__file__))
if _day3_dir not in sys.path:
    sys.path.insert(0, _day3_dir)
_task_dir = os.path.join(_day3_dir, "task")
if _task_dir not in sys.path:
    sys.path.insert(0, _task_dir)


def get_env_cfg(task_type: str):
    """Import and return the env cfg class based on task type."""
    import importlib

    if task_type == "pickplace":
        mod = importlib.import_module("task.lift.config.ik_abs_env_cfg_3_1_answer")
        return mod.FrankaTBarPickPlaceEnvCfg
    elif task_type == "pusht":
        mod = importlib.import_module("task.lift.custom_pusht_env_cfg_3_2_answer")
        return mod.PushTEnvCfg
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def get_success_fn(task_type: str):
    """task_type에 맞는 성공 판정 함수를 불러옵니다."""
    if task_type == "pickplace":
        from task.lift.mdp_3_1.terminations import object_pickplace_goal
        return object_pickplace_goal
    elif task_type == "pusht":
        from task.lift.mdp_3_2.terminations_answer import object_pusht_goal
        import functools
        # 성공 기준 완화
        return functools.partial(object_pusht_goal, pos_threshold=0.05, yaw_threshold=0.15)
    return None


def obs_to_robomimic(obs_policy: dict, device: torch.device, frame_stack: int = 2) -> dict:
    """
    Isaac Lab 관측값(obs)을 diffusion policy 입력 형식으로 변환합니다.

    diffusion policy는 (Batch, Time, ...) 형태의 입력을 기대합니다.

    Args:
        obs_policy: env.step()에서 반환된 관측값 dict (단일 env)
        device: torch device
        frame_stack: 쌓을 프레임 수 (config.train.frame_stack과 일치해야 함)

    Returns:
        obs_batched: diffusion policy에 전달할 수 있는 batched dict
    """
    obs_batched = {}
    for key, val in obs_policy.items():
        if val.ndim == 1:
            # ================================================================
            # TODO 1: Low-dim 관측값 변환
            # ================================================================
            # val의 shape: (dim,)  (예: joint_pos는 (9,))
            #
            # diffusion policy가 기대하는 shape: (1, T, dim)
            #   - 1 = batch size
            #   - T = frame_stack (시간 축)
            #   - dim = 원래 차원
            #
            # 단계:
            #   1. val을 float으로 변환하고 device로 이동
            v = val.float().to(device)

            #   2. unsqueeze(0)으로 batch 차원 추가: (dim,) → (1, dim)
            #   3. unsqueeze(1)으로 time 차원 추가: (1, dim) → (1, 1, dim)
            #   4. expand(-1, frame_stack, -1)로 time 복제: (1, 1, dim) → (1, T, dim)
            v = ???
            # ================================================================
            raise NotImplementedError("TODO 1: Low-dim 관측값을 (1, T, dim) 형태로 변환하세요!")

        elif val.ndim == 3:
            # ================================================================
            # TODO 2: 이미지 관측값 변환
            # ================================================================
            # val의 shape: (H, W, C)  (예: top_cam은 (240, 320, 3))
            #
            # diffusion policy가 기대하는 shape: (1, T, H, W, C)
            #
            # 단계:
            #   1. val을 float으로 변환하고 device로 이동:
            v = val.float().to(device)

            #   2. alpha 채널 제거 (4채널→3채널):
            if v.shape[-1] == 4:
                v = v[..., :3]

            #   3. unsqueeze(0).unsqueeze(0)으로 (1, 1, H, W, C)
            #   4. expand(-1, frame_stack, ...)로 (1, T, H, W, C)
            v = ???
            # ================================================================
            raise NotImplementedError("TODO 2: 이미지 관측값을 (1, T, H, W, C) 형태로 변환하세요!")

        else:
            v = val.float().to(device)
            v = v.unsqueeze(0).unsqueeze(0).expand(-1, frame_stack, *val.shape)

        obs_batched[key] = v

    return obs_batched


def main():
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda:0")

    # ---- 체크포인트 로드 (5.1에서 검증 완료) ----
    print(f"\n[5.2] Loading checkpoint...")
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint, device=device, verbose=True,
    )

    # ---- 환경 생성 (5.1에서 검증 완료) ----
    EnvCfgClass = get_env_cfg(args_cli.task_type)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.concatenate_terms = False
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    TRAIN_H, TRAIN_W = 480, 640
    for cam_name in ("camera", "top_camera"):
        if hasattr(env_cfg.scene, cam_name):
            cam_cfg = getattr(env_cfg.scene, cam_name)
            if hasattr(cam_cfg, "height"):
                cam_cfg.height = TRAIN_H
                cam_cfg.width = TRAIN_W

    env = ManagerBasedEnv(cfg=env_cfg)
    print(f"[5.2] Environment: {args_cli.task_type}")

    # 성공 판정 함수 로드 (mdp 모듈에서)
    success_fn = get_success_fn(args_cli.task_type)

    # ---- Rollout 실행 ----
    success_count = 0

    for ep in range(args_cli.num_rollouts):
        obs_full, _ = env.reset()
        policy.start_episode()

        ep_success = False

        for step in range(args_cli.max_steps):
            with torch.no_grad():
                # obs 변환 (TODO 1, 2가 구현되어야 동작)
                obs_policy = obs_full["policy"]
                obs_single = {k: v[0] for k, v in obs_policy.items()}
                obs_batched = obs_to_robomimic(obs_single, device)

                # 첫 에피소드 첫 스텝에서 shape 확인
                if ep == 0 and step == 0:
                    print(f"\n[5.2] Isaac Lab obs (raw):")
                    for k, v in obs_single.items():
                        print(f"    {k:15s} shape={tuple(v.shape)}")
                    print(f"\n[5.2] Converted obs:")
                    for k, v in obs_batched.items():
                        print(f"    {k:15s} shape={tuple(v.shape)}")

                # Policy 추론
                action = policy(obs_batched, batched_ob=True)
                action = action[0]  # unbatch → (ac_dim,)

                # Action 변환: numpy/tensor → (num_envs, ac_dim) GPU 텐서
                if isinstance(action, np.ndarray):
                    action_tensor = torch.from_numpy(action).float().to(device)
                else:
                    action_tensor = action.float().to(device)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                actions = action_tensor.repeat(env.num_envs, 1)

                # ManagerBasedEnv.step() returns (obs, extras)
                obs_full = env.step(actions)[0]

                # 성공 판정 (mdp 모듈의 함수 사용)
                if success_fn is not None:
                    result = success_fn(env)
                    if isinstance(result, torch.Tensor):
                        is_success = bool(result.reshape(-1)[0].item())
                    else:
                        is_success = bool(result)
                    if is_success:
                        ep_success = True
                        break

        if ep_success:
            success_count += 1

        status = "✓ SUCCESS" if ep_success else "✗ FAIL"
        print(f"  Rollout {ep+1:3d}/{args_cli.num_rollouts} | "
              f"Steps: {step+1:4d} | {status}")

    # ---- Summary ----
    success_rate = success_count / args_cli.num_rollouts * 100

    print(f"\n{'='*60}")
    print(f"  [5.2] Evaluation Results")
    print(f"  Task:         {args_cli.task_type}")
    print(f"  Rollouts:     {args_cli.num_rollouts}")
    print(f"  Success Rate: {success_count}/{args_cli.num_rollouts} ({success_rate:.1f}%)")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
