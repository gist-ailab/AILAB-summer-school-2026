#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# [ANSWER 1/2] day3_5 · 체크포인트 로드 + 환경 생성
#
# TODO 1 정답: importlib으로 환경 설정 불러오기
# TODO 2 정답: obs dict 모드 설정
# =====================================================================
"""
[5.1] 체크포인트 로드 & 환경 생성 확인 (정답)

ManagerBasedEnv를 직접 생성하는 방식.

Usage:
    $ISAACLAB_PATH/isaaclab.sh -p day3_5.1_eval_answer.py \
        --task_type pusht \
        --checkpoint <체크포인트.pth>
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="[5.1] Load checkpoint & create environment.")
parser.add_argument("--task_type", type=str, default="pusht", choices=["pickplace", "pusht"],
                    help="Task type: 'pickplace' or 'pusht'.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import sys
import os
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
    # [TODO 1 정답]
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

    # ---- Step 1: 체크포인트 로드 ----
    print(f"\n[5.1] Loading checkpoint: {args_cli.checkpoint}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint,
        device=device,
        verbose=True,
    )
    policy.start_episode()
    print(f"[5.1] ✓ Policy loaded successfully!")

    # ---- Step 2: 환경 생성 ----
    EnvCfgClass = get_env_cfg(args_cli.task_type)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    # [TODO 2 정답]
    # obs를 dict로 반환하도록 설정 (diffusion policy가 key별로 처리하기 위해)
    env_cfg.observations.policy.concatenate_terms = False

    # Disable recorder if present
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    # 카메라 해상도를 학습 데이터와 맞추기 (240×320)
    TRAIN_H, TRAIN_W = 480, 640
    for cam_name in ("camera", "top_camera"):
        if hasattr(env_cfg.scene, cam_name):
            cam_cfg = getattr(env_cfg.scene, cam_name)
            if hasattr(cam_cfg, "height"):
                cam_cfg.height = TRAIN_H
                cam_cfg.width = TRAIN_W

    env = ManagerBasedEnv(cfg=env_cfg)

    # ---- 결과 확인 ----
    print(f"\n{'='*60}")
    print(f"  [5.1] 환경 생성 완료!")
    print(f"  Task type:  {args_cli.task_type}")
    print(f"  Num envs:   {env.num_envs}")

    # obs 확인
    obs, _ = env.reset()
    obs_policy = obs["policy"]
    print(f"\n  Observation keys:")
    for k, v in obs_policy.items():
        print(f"    {k:15s} shape={tuple(v.shape)}  dtype={v.dtype}  "
              f"range=[{v.float().min():.4f}, {v.float().max():.4f}]")

    print(f"\n  ✓ 환경 생성을 정상적으로 완료했습니다! 다음 단계(5.2)를 진행하세요.")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
