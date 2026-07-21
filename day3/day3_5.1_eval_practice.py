#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# [PRACTICE 1/2] day3_5 · 체크포인트 로드 + 환경 생성
#
# 이 스크립트는 학습된 Diffusion Policy 체크포인트를 로드하고
# Isaac Lab 환경을 생성하는 첫 번째 단계입니다.
#
# 채워야 할 곳: 2군데 (TODO 1, 2)
#   TODO 1: importlib으로 환경 설정(env_cfg) 불러오기
#   TODO 2: obs를 dictionary 모드로 설정하기
#
# 실행하면: 체크포인트 정보와 환경 정보가 출력된 후 종료됩니다.
# =====================================================================
"""
[5.1] 체크포인트 로드 & 환경 생성 확인

학습된 모델을 로드하고, Isaac Lab 환경이 올바르게 생성되는지 확인합니다.
성공하면 체크포인트/환경 정보가 출력됩니다.

Usage:
    $ISAACLAB_PATH/isaaclab.sh -p day3_5.1_eval_practice.py \
        --task_type pusht\
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
    """
    task_type에 해당하는 환경 설정 클래스를 불러옵니다.

    Isaac Lab에서 환경을 만들려면 해당 환경의 Config 클래스가 필요합니다.
    예를 들어 pickplace는 FrankaTBarPickPlaceEnvCfg,
    pusht는 PushTEnvCfg 클래스를 사용합니다.
    """
    # ================================================================
    # TODO 1: importlib으로 환경 설정 모듈을 불러오세요
    # ================================================================
    # Python의 importlib.import_module()을 사용하면
    # 문자열로 모듈을 동적으로 import할 수 있습니다.
    #
    # 예시:
    #   import importlib
    #   mod = importlib.import_module("패키지.모듈이름")
    #   MyClass = mod.클래스이름
    #
    # 환경 설정 파일 위치:
    #   pickplace: "task.lift.config.ik_abs_env_cfg_3_1_answer"
    #              → 클래스: ???Cfg
    #   pusht:     "task.lift.custom_pusht_env_cfg_3_2_answer"
    #              → 클래스: ???
    #
    # 작성 코드:
    import importlib
    if task_type == "pickplace":
        mod = importlib.import_module("task.lift.config.ik_abs_env_cfg_3_1_answer")
        return mod.???
    elif task_type == "pusht":
        mod = importlib.import_module("task.lift.custom_pusht_env_cfg_3_2_answer")
        return mod.???
    # ================================================================
    raise NotImplementedError("TODO 1: importlib으로 환경 설정을 불러오세요!")


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

    # ================================================================
    # TODO 2: obs를 dictionary 모드로 설정
    # ================================================================
    # diffusion policy는 obs를 key별로 처리합니다 (joint_pos, top_cam 등).
    # Isaac Lab은 기본적으로 모든 obs를 하나의 텐서로 합치지만(concatenate),
    # 이를 False로 설정하면 dict 형태로 반환됩니다.
    #
    # 힌트:
    env_cfg.observations.policy.concatenate_terms = ???
    # ================================================================
    raise NotImplementedError("TODO 2: obs를 dict 모드로 설정하세요!")

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
