# terminations.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
#  [문제 8] T-bar가 목표 위치·회전에 도달했는지 판정
#  2교시 · Teleop  |  
#  ── 할 일: T-bar가 목표 위치·회전에 도달했는지 판정하는 성공 조건 로직을 완성하세요.
# ============================================================

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_pusht_goal(
    env: ManagerBasedEnv,
    pos_threshold: float = 0.01,
    yaw_threshold: float = 0.1,
    goal_pos: tuple[float, float] = (0.4, 0.0),
    goal_yaw: float = 1.57079632679,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
) -> torch.Tensor:
    """T-bar가 목표 위치와 회전에 도달했는지 판정합니다.

    Args:
        env: 시뮬레이션 환경
        pos_threshold: 위치(x, y) 허용 오차
        yaw_threshold: 회전(yaw) 허용 오차
        goal_pos: 목표 위치 (x, y)
        goal_yaw: 목표 회전 (yaw)
        object_cfg: 판단할 객체(T-bar)

    Returns:
        성공 여부
    """
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w - env.scene.env_origins

    # 1. 위치(Position) 오차 계산 (x, y)
    obj_pos = obj_pos_w[:, :2]
    goal_pos_tensor = torch.tensor([goal_pos], device=env.device)
    pos_error = torch.norm(obj_pos - goal_pos_tensor, dim=-1)

    # 2. 회전(Yaw) 오차 계산
    obj_quat = obj.data.root_quat_w
    _, _, yaw = euler_xyz_from_quat(obj_quat)
    goal_yaw_tensor = torch.tensor([goal_yaw], device=env.device)
    yaw_error = torch.abs(wrap_to_pi(yaw - goal_yaw_tensor))

    # 임계값(Threshold) 비교
    # 에러값이 작을 때 디버깅 (day3_5 모델 평가 시 변화량 파악용)
    if pos_error.min() < 0.1:
        print(f"[DEBUG PushT] pos_error: {pos_error.min().item():.3f}, yaw_error: {yaw_error.min().item():.3f}",end="\r")

    # 3. 위치와 회전 오차가 모두 허용치 이내인지 확인
    is_success = (pos_error < pos_threshold) & (yaw_error < yaw_threshold)
    return is_success
