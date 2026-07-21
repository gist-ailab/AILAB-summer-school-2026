# terminations.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pickplace_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.15,
    threshold_z: float = 0.10,
    object_0_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """T_bar(object_0)가 바구니(bin) 안에 들어왔는지 판정.

    Args:
        env: 시뮬레이션 환경
        threshold: 바구니와 T_bar의 수평(x,y) 거리 임계값
        threshold_z: 바구니와 T_bar의 수직(z) 거리 임계값
        object_0_cfg: T_bar
        bin_cfg: 바구니

    Returns:
        각 환경별 성공 여부 (True: 바구니에 안착, False: 미완료)
    """
    object_0: RigidObject = env.scene[object_0_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]

    # 수평 거리 (x, y)
    distance_xy = torch.norm(
        bin.data.root_pos_w[:, :2] - object_0.data.root_pos_w[:, :2], dim=1
    )
    # 수직 거리 (z)
    distance_z = torch.abs(
        bin.data.root_pos_w[:, 2] - object_0.data.root_pos_w[:, 2]
    )
    # 둘 다 임계값 이내면 성공 (배치 전체를 한 번에 계산 → 병렬에서도 동작)
    done = (distance_xy < threshold) & (distance_z < threshold_z)
    return done