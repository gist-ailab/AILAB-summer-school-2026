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
parser.add_argument("--eval_seed", type=int, default=None,
                    help="rollout i 의 초기 배치를 (eval_seed+i) 로 고정. 조건 간 공정 비교용. "
                         "visual DR 스타일도 이 시드로 재현된다.")
parser.add_argument("--no_video", action="store_true",
                    help="비디오 녹화 비활성화. 대량 평가 시 시간 절약.")
parser.add_argument("--ddim_steps", type=int, default=None,
                    help="지정하면 추론 샘플러를 DDIM 으로 전환하고 이 스텝 수로 추론한다"
                         "(권장 10~16, 재학습 불필요). 미지정 시 학습된 DDPM(기본 100) 유지.")
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

import robomimic.utils.file_utils as FileUtils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sim import PinholeCameraCfg

# Add day3/ and day3/task/ to path
_day3_dir = os.path.abspath(os.path.dirname(__file__))
if _day3_dir not in sys.path:
    sys.path.insert(0, _day3_dir)
_task_dir = os.path.join(_day3_dir, "task")
if _task_dir not in sys.path:
    sys.path.insert(0, _task_dir)

# legacy(normalize=True 재현) 전용 유틸. day3_5_eval_answer.py 와 동일 모듈을 공유한다.
import day3_5_legacy_img_norm as legacy

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

# 중요: "original"/"wide" 는 day3_3.4_mimic_datagenerator_rollout_answer.py 의
#       SPAWN_RANDOMIZATION_PRESETS 와 축·범위가 정확히 일치해야 한다.
#       그래야 "wide 로 증강 학습한 모델이 wide 평가에서 유리한가?" 라는 질문에
#       답할 수 있다. 이름만 같고 회전축이 다르면(예: pitch vs yaw) 학습 분포 밖이 되어
#       증강 모델조차 학습 중 본 적 없는 장면에서 평가돼 실패하고, 결과가 정책 성능처럼 오독된다.
PICKPLACE_SPAWN_PRESETS = {
    "original": {
        "object_x": (-0.1, 0.1), "object_y": (-0.1, 0.1),
        "object_yaw": (-math.pi / 4, math.pi / 4),
        "bin_pose_range": None,
    },
    "wide": {
        "object_x": (-0.18, 0.18), "object_y": (-0.18, 0.18),
        "object_yaw": (-math.pi / 4, 3 * math.pi / 4),
        "bin_pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "pitch": (-math.pi / 6, math.pi / 6)},
    },
    # "extreme" 은 학습 분포 밖(OOD) 을 일부러 보기 위한 범위다. 증강 모델도 실패하는 것이
    # 정상이며, 모델 간 우열 비교용이 아니라 "어디서 무너지는가" 관찰용이다.
    "extreme": {
        "object_x": (-0.25, 0.25), "object_y": (-0.25, 0.25),
        "object_yaw": (-math.pi / 2, math.pi),
        "bin_pose_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "pitch": (-math.pi / 4, math.pi / 4)},
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
    ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=checkpoint_path)
    # 다중 데이터셋(예: teleop+DR)으로 학습한 체크포인트는 shape_metadata 가
    # list[dict] 로 저장된다. robomimic 로더는 dict 를 기대하므로 첫 항목으로 펼친다.
    if isinstance(ckpt_dict.get("shape_metadata"), list):
        n = len(ckpt_dict["shape_metadata"])
        ckpt_dict["shape_metadata"] = ckpt_dict["shape_metadata"][0]
        print(f"[GEN-EVAL] shape_metadata list({n}) -> 첫 항목으로 펼침 (다중 데이터셋)")
    dim = getattr(args_cli, "ddim_steps", None)
    if dim is not None:
        # 추론 샘플러를 DDIM 으로 전환(재학습 불필요, 같은 ε-네트워크). 스케줄러는 정책
        # 생성 시 config 로 정해지므로 여기서 flip 해야 한다. (day3_5_eval_answer.py 와 동일)
        cfg = json.loads(ckpt_dict["config"])
        cfg["algo"]["ddim"]["enabled"] = True
        cfg["algo"]["ddpm"]["enabled"] = False
        cfg["algo"]["ddim"]["num_inference_timesteps"] = dim
        ckpt_dict["config"] = json.dumps(cfg)
        print(f"[GEN-EVAL] 추론 샘플러 -> DDIM (num_inference_timesteps={dim})")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=True,
    )
    return policy


# =====================================================================
#  이미지 인코딩 
#  raw   : normalize=False -> 카메라 uint8 [0,255] 그대로
#  legacy: normalize=True + day3_4.99_preprocess 로 만든 데이터.
#          env 출력을 학습 때와 같은 상수/식으로 재현해야 한다.
# =====================================================================
OBS_ENCODING = "raw"
IMG_STATS = {}

DEFAULT_IMG_STATS = {
    "pusht": {
        "top_cam":   (-0.7432959675788879, 0.2146419882774353),
        "wrist_cam": (-0.8750622868537903, 0.7452652454376221),
    },
    "pickplace": {
        "top_cam":   (-0.8524356484, 0.1419889331),
        "wrist_cam": (-0.8809610605, 0.3209683299),
    },
}


def encode_obs_image(img: torch.Tensor, obs_key: str) -> torch.Tensor:
    global OBS_ENCODING
    if img.shape[-1] == 4:
        img = img[..., :3]
    env_is_uint8 = (img.dtype == torch.uint8)
    if OBS_ENCODING == "raw":
        if env_is_uint8:
            return img
        # float 반환 = normalize=True = legacy 파이프라인. 상수를 알면 자동 전환한다.
        if IMG_STATS.get(obs_key) is not None:
            OBS_ENCODING = "legacy"
            print("[GEN-EVAL] env 카메라가 float 를 반환 -> obs_encoding 을 'legacy' 로 자동 전환합니다.")
        else:
            raise RuntimeError(
                f"raw 인코딩인데 카메라가 float 반환. '{obs_key}' 상수를 몰라 전환 불가. "
                "legacy 모델이면 사이드카/학습 hdf5 의 정규화 상수가 필요합니다.")
    # legacy 변환(normalize=True 재현)은 day3_5_legacy_img_norm.py 로 분리돼 있다.
    return legacy.encode_legacy(img, obs_key, IMG_STATS, env_is_uint8)


def preprocess_obs(obs_policy: dict, device: torch.device) -> dict:
    out = {}
    for key, val in obs_policy.items():
        out[key] = (encode_obs_image(val.to(device), key) if val.ndim == 3
                    else val.float().to(device))
    return out


class ObsFrameStacker:
    def __init__(self, frame_stack=2):
        self.frame_stack = frame_stack
        self.buffer = None

    def reset(self):
        self.buffer = None

    def add(self, obs):
        self.buffer = [obs] * self.frame_stack if self.buffer is None \
            else self.buffer[1:] + [obs]

    def get_batched(self):
        return {k: torch.stack([f[k] for f in self.buffer], dim=0).unsqueeze(0)
                for k in self.buffer[0]}


ALIGN_ACTIONS = {
    "pusht": torch.tensor([[0.4, 0.0, 0.005, 0.0, 1.0, 0.0, 0.0, -1.0]]),
    "pickplace": torch.tensor([[0.46590596437454224, 4.9243681132793427e-08,
                                0.38296937942504883, 0.0, 1.0, 0.0, 0.0, 1.0]]),
}


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

    # yaw 를 default quaternion 과 합성한다. (bin 과 동일 방식, native reset_root_state_uniform 과 일치)
    # 주의: T-bar 의 default quat 은 identity 가 아니라 테이블에 눕힌 자세이므로,
    #       qw/qz 만 덮어쓰면 qx/qy 가 남아 잘못된(비정규화) orientation 이 되어 물체가 망가진다.
    yaw = torch.FloatTensor(1).uniform_(*preset["object_yaw"]).to(env.device)
    yaw_quat = torch.zeros(1, 4, device=env.device)
    yaw_quat[:, 0] = torch.cos(yaw / 2)  # qw
    yaw_quat[:, 3] = torch.sin(yaw / 2)  # qz (world-z 축 회전)
    from isaaclab.utils.math import quat_mul
    default_quat = obj_state[:, 3:7].clone()  # (qw, qx, qy, qz)
    obj_state[:, 3:7] = quat_mul(yaw_quat, default_quat)
    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

    # ---- Bin ----
    # 학습 데이터 생성기(day3_3.4_mimic_datagenerator_rollout_answer.py)와 **완전히 동일한**
    # 방식으로 적용해야 한다. 그쪽은 roll/pitch/yaw 오일러각을 default quat 에 국소(local)
    # 합성한다: quat_mul(default, delta). 축이나 합성 순서가 다르면 학습 때 본 적 없는
    # 분포가 되어, 넓은 범위로 증강 학습한 모델조차 이점을 보이지 못한다.
    bin_range = preset.get("bin_pose_range")
    if bin_range is not None:
        from isaaclab.utils.math import quat_mul, quat_from_euler_xyz

        bin_obj = env.scene["bin"]
        bin_state = bin_obj.data.default_root_state[env_ids].clone()

        def _sample(key):
            lo, hi = bin_range.get(key, (0.0, 0.0))
            return torch.empty(1, device=env.device).uniform_(float(lo), float(hi))

        bin_state[:, 0] += _sample("x")
        bin_state[:, 1] += _sample("y")
        bin_state[:, 2] += _sample("z")
        delta = quat_from_euler_xyz(_sample("roll"), _sample("pitch"), _sample("yaw"))
        bin_state[:, 3:7] = quat_mul(bin_state[:, 3:7].clone(), delta)

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
    global OBS_ENCODING, IMG_STATS
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    rng = random.Random()

    # ---- 이미지 인코딩 자동 결정 (day3_5_eval_answer.py 와 동일 규칙) ----
    # 우선순위: 사이드카 캐시(데이터셋 불필요) > 체크포인트 자동탐지 > 태스크 기본값
    datasets_dir = os.path.join(_day3_dir, "datasets")
    explicit_stats = legacy.load_stats_sidecar(args_cli.checkpoint)
    src = "명시 지정"
    if explicit_stats:
        src = "사이드카 캐시(데이터셋 미사용)"
        print(f"[GEN-EVAL] 정규화 상수를 캐시에서 불러왔습니다.")
    if not explicit_stats:
        auto_ds = legacy.resolve_train_datasets_from_ckpt(args_cli.checkpoint, datasets_dir)
        if auto_ds:
            explicit_stats = legacy.load_img_stats_multi(auto_ds)
            if explicit_stats:
                tag = auto_ds[0] if len(auto_ds) == 1 \
                    else f"{len(auto_ds)}개 (multi): {', '.join(os.path.basename(p) for p in auto_ds)}"
                print(f"[GEN-EVAL] 학습 데이터셋 자동 탐지: {tag}")
                src = "체크포인트 자동탐지"
                legacy.save_stats_sidecar(args_cli.checkpoint, explicit_stats)
    IMG_STATS = {**DEFAULT_IMG_STATS.get(args_cli.task_type, {}), **explicit_stats}

    OBS_ENCODING = "legacy" if explicit_stats else "raw"
    if not explicit_stats and IMG_STATS:
        src = f"'{args_cli.task_type}' 태스크 기본값"
    print(f"[GEN-EVAL] obs_encoding = {OBS_ENCODING}  (정규화 상수: {src})")
    stacker = ObsFrameStacker(2)

    # ---- Load policy ----
    print(f"\n[GEN-EVAL] Loading checkpoint: {args_cli.checkpoint}")
    policy = load_policy(args_cli.checkpoint, device)

    # ---- 체크포인트 메타데이터: frame_stack, 학습 해상도, 학습 obs 키 ----
    # 두 값 모두 여기서 한 번에 읽는다. (day3_5_eval_answer.py 와 동일)
    ckpt_img_shapes = {}
    ckpt_obs_keys = None   # 학습에 실제로 쓴 obs 키 집합(예: top_cam 단독). None 이면 필터 안 함.
    try:
        ck = FileUtils.load_dict_from_checkpoint(ckpt_path=args_cli.checkpoint)
        cfg = ck["config"]
        cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
        stacker = ObsFrameStacker(int(cfg["train"]["frame_stack"]))

        sm = ck["shape_metadata"]
        if isinstance(sm, list):      # 다중 데이터셋 학습 시 list[dict]
            sm = sm[0]
        ckpt_obs_keys = set(sm["all_shapes"].keys())
        for k, shape in sm["all_shapes"].items():
            if len(shape) == 3 and int(shape[0]) in (1, 3):   # (C, H, W)
                ckpt_img_shapes[k] = (int(shape[1]), int(shape[2]))
        print(f"[GEN-EVAL] frame_stack={stacker.frame_stack}  학습 해상도={ckpt_img_shapes}")
        print(f"[GEN-EVAL] 학습 obs 키={sorted(ckpt_obs_keys)}")
    except Exception as e:
        print(f"[WARN] 체크포인트 메타데이터를 읽지 못했습니다: {e}")

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

    # ---- obs 카메라 해상도를 학습 해상도에 맞춘다 ----
    # 체크포인트가 기록한 학습 해상도를 우선 사용하고, 못 읽으면 env cfg 기본값으로 폴백한다.
    CAM_TO_OBS_KEY = {"camera": "wrist_cam", "top_camera": "top_cam"}
    for cam_name, obs_key in CAM_TO_OBS_KEY.items():
        if hasattr(env_cfg.scene, cam_name):
            cam_cfg = getattr(env_cfg.scene, cam_name)
            if hasattr(cam_cfg, "height"):
                used_ckpt = obs_key in ckpt_img_shapes
                h, w = ckpt_img_shapes.get(obs_key, (cam_cfg.height, cam_cfg.width))
                cam_cfg.height, cam_cfg.width = h, w
                src = "from checkpoint" if used_ckpt else "fallback(env default)"
                print(f"[GEN-EVAL] Obs camera '{cam_name}' ({obs_key}) -> {h}x{w} [{src}]")
                model_uses = (ckpt_obs_keys is None) or (obs_key in ckpt_obs_keys)
                if not used_ckpt and model_uses:
                    print(f"  <-- 경고: '{obs_key}' 는 정책 입력인데 학습 해상도를 확인하지 못해 "
                          f"env 기본값({h}x{w})을 사용합니다. 학습 해상도와 다르면 성능이 무너집니다.")

    # ---- 고해상도 비디오 전용 카메라 ----
    if args_cli.no_video:
        video_camera_name = None
        print("[GEN-EVAL] --no_video: 비디오 녹화 비활성화 (속도 우선)")
    else:
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
        print(f"[GEN-EVAL] Video camera -> {VIDEO_H}x{VIDEO_W}")

    env = ManagerBasedEnv(cfg=env_cfg)

    # Success function
    success_fn = get_success_fn(args_cli.task_type)

    # Visual DR targets
    visual_targets = None
    if args_cli.visual_dr:
        visual_targets = build_visual_dr_targets()

    # Video recording
    save_video = video_camera_name is not None
    if save_video:
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
    os.makedirs(video_dir if save_video else eval_dir, exist_ok=True)

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
    rollout_results = []

    for ep in range(args_cli.num_rollouts):
        # 초기 배치 + visual DR 스타일 재현: torch/numpy 전역 시드와 python rng 를
        # rollout 마다 고정한다. 같은 eval_seed 면 조건이 달라도 동일한 초기상태·DR세트.
        if args_cli.eval_seed is not None:
            s = args_cli.eval_seed + ep
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            np.random.seed(s)
            rng.seed(s)
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

        # ---- 리셋+랜덤화 직후 정렬(Align) ----
        # arm_action 은 절대 pose IK 이므로 데이터 수집과 동일한 정렬 액션을 보내야
        # 정책이 학습 때와 같은 초기 로봇 자세에서 출발한다.
        align = ALIGN_ACTIONS.get(args_cli.task_type)
        if align is not None:
            a = align.to(device).repeat(env.num_envs, 1)
            if a.shape[-1] == env.action_manager.total_action_dim:
                for _ in range(10):
                    obs_full = env.step(a)[0]

        # obs 재계산 (랜덤화/DR 반영)
        obs_full = {"policy": env.observation_manager.compute_group("policy", update_history=False)}

        policy.start_episode()
        stacker.reset()
        ep_success = False
        frames = []

        for step in range(args_cli.max_steps):
            with torch.no_grad():
                obs_policy = obs_full["policy"]
                obs_single = {k: v[0] for k, v in obs_policy.items()}
                # 학습에 쓴 obs 키만 남긴다. env 는 top_cam·wrist_cam 을 모두 내보내지만,
                # top_cam 단독으로 학습한 모델에 wrist_cam 을 넘기면 robomimic 이 그 키를
                # 'assumed low_dim' 으로 잘못 등록해 정책 입력을 오염시킨다.
                if ckpt_obs_keys is not None:
                    obs_single = {k: v for k, v in obs_single.items() if k in ckpt_obs_keys}
                stacker.add(preprocess_obs(obs_single, device))
                obs_batched = stacker.get_batched()

                if ep == 0 and step == 0:
                    missing = (ckpt_obs_keys - set(obs_single.keys())) if ckpt_obs_keys else set()
                    if missing:
                        print(f"[WARN] 학습 obs 키 중 env 에 없는 것: {sorted(missing)}")
                    print(f"[DEBUG] policy 입력 obs 키: {sorted(obs_single.keys())}")
                    print(f"[DEBUG] Observation shapes:")
                    for k, v in obs_single.items():
                        extra = ""
                        if v.ndim == 3 and v.shape[-1] in (1, 3):
                            vf = v.float()
                            nz = (vf.abs() > 1e-6).float().mean().item() * 100
                            extra = f"  nonzero={nz:.1f}% min={vf.min():.3f} max={vf.max():.3f} mean={vf.mean():.3f}"
                        print(f"  {k}: {tuple(v.shape)}{extra}")

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
