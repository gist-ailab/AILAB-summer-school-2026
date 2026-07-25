#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# [ANSWER] day3_5 · Diffusion Policy 평가 (Eval)
#
# 학습된 Diffusion Policy 체크포인트를 Isaac Lab 환경에서 평가합니다.
# - 체크포인트 로드 → 환경 생성 → rollout 실행 → 성공률/보상 집계
# - Isaac Lab obs를 robomimic 형식으로 변환하는 파이프라인 포함
# - 비디오 녹화 및 JSON 결과 저장
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
parser.add_argument("--eval_seed", type=int, default=None,
                    help="지정하면 rollout i 의 초기 배치를 (eval_seed+i) 로 고정한다. "
                         "같은 값이면 epoch/조건이 달라도 동일한 초기 상태 세트에서 평가되어 "
                         "공정한 대응 비교가 된다. 미지정 시 매번 랜덤(재현 불가).")
parser.add_argument("--action_horizon", type=int, default=None,
                    help="추론 시 실행할 액션 수(Ta) 오버라이드. 재학습 불필요. "
                         "작을수록 자주 재계획(반응성↑, 연산↑).")
parser.add_argument("--ddpm_steps", type=int, default=None,
                    help="DDPM 추론 스텝 수 오버라이드 (기본 100). DDPM 은 적은 스텝에서 "
                         "품질이 저하되므로, 빠른 추론이 목적이면 --ddim_steps 를 권장한다.")
parser.add_argument("--ddim_steps", type=int, default=None,
                    help="지정하면 추론 샘플러를 DDIM 으로 전환하고 이 스텝 수로 추론한다"
                         " 미지정 시 학습된 DDPM(기본 100) 유지. "
                         "기존 결과는 DDPM 100 으로 기록됐으므로 재현하려면 이 옵션을 주지 않는다.")
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

import robomimic.utils.file_utils as FileUtils

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

# legacy(normalize=True 재현) 전용 유틸. raw 파이프라인에서는 쓰이지 않는다.
import day3_5_legacy_img_norm as legacy


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
    

def pusht_goal_error(env, goal_pos=(0.4, 0.0), goal_yaw=1.57079632679):
    """
    T-bar 의 목표 대비 위치/회전 오차 (성공 여부와 별개로 '얼마나 가까운지').

    task/lift/mdp_3_2/terminations_answer.py 의 object_pusht_goal 과 동일한 계산.
    성공/실패만 보면 "정밀도가 모자란 것"과 "물체를 아예 안 건드린 것"을 구분할 수 없다.
    """
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    obj = env.scene["object_0"]
    obj_pos = (obj.data.root_pos_w - env.scene.env_origins)[:, :2]
    pos_error = torch.norm(obj_pos - torch.tensor([goal_pos], device=env.device), dim=-1)
    _, _, yaw = euler_xyz_from_quat(obj.data.root_quat_w)
    yaw_error = torch.abs(wrap_to_pi(yaw - torch.tensor([goal_yaw], device=env.device)))
    return float(pos_error.min().item()), float(yaw_error.min().item())


def load_policy(checkpoint_path: str, device: torch.device):
    """
    Load a trained robomimic Diffusion Policy from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Torch device.

    Returns:
        policy: robomimic PolicyAlgo instance in eval mode.
    """
    # 추론 파라미터 오버라이드 (재학습 불필요 — inference 시에만 쓰임).
    # 체크포인트를 먼저 dict 로 읽어 config JSON 을 수정한 뒤 policy 를 만든다.
    ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=checkpoint_path)
    if isinstance(ckpt_dict.get("shape_metadata"), list):
        n = len(ckpt_dict["shape_metadata"])
        ckpt_dict["shape_metadata"] = ckpt_dict["shape_metadata"][0]
        print(f"[EVAL] shape_metadata 가 list({n}) -> 첫 항목으로 (multi-dataset 학습)")
    ah = getattr(args_cli, "action_horizon", None)
    ds = getattr(args_cli, "ddpm_steps", None)
    dim = getattr(args_cli, "ddim_steps", None)
    if ah is not None or ds is not None or dim is not None:
        cfg = json.loads(ckpt_dict["config"])
        if ah is not None:
            old = cfg["algo"]["horizon"]["action_horizon"]
            cfg["algo"]["horizon"]["action_horizon"] = ah
            print(f"[EVAL] action_horizon 오버라이드: {old} -> {ah}")
        if dim is not None:
            # 추론 샘플러를 DDIM 으로 전환한다. 같은 ε-네트워크·같은 beta_schedule 을
            # 쓰므로 재학습이 필요 없다. DDPM(100스텝)과 달리 적은 스텝(10~16)에서도
            # 결정론적으로 품질을 유지한다. 스케줄러는 정책 생성 시 config 로 정해지므로
            # (robomimic diffusion_policy.py 의 _create_networks) 여기서 flip 해야 한다.
            cfg["algo"]["ddim"]["enabled"] = True
            cfg["algo"]["ddpm"]["enabled"] = False
            cfg["algo"]["ddim"]["num_inference_timesteps"] = dim
            print(f"[EVAL] 추론 샘플러 -> DDIM (num_inference_timesteps={dim})")
            if ds is not None:
                print("[EVAL] --ddim_steps 가 지정되어 --ddpm_steps 는 무시합니다.")
        elif ds is not None:
            old = cfg["algo"]["ddpm"]["num_inference_timesteps"]
            cfg["algo"]["ddpm"]["num_inference_timesteps"] = ds
            print(f"[EVAL] ddpm num_inference_timesteps 오버라이드: {old} -> {ds}")
        ckpt_dict["config"] = json.dumps(cfg)

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict,
        device=device,
        verbose=True,
    )
    policy.start_episode()
    return policy




# =====================================================================
#  이미지 인코딩
#  [RAW]    env cfg 의 ObsTerm 에 normalize=False -> 카메라가 uint8 [0,255] 를
#           그대로 내보내고, 그게 곧 학습 데이터다. robomimic 이 /255 만 하면 된다.
#           (권장 파이프라인. 전처리 단계가 필요 없다.)
#
#  [LEGACY] ObsTerm 이 normalize=True (IsaacLab 기본값) 인 상태로 수집한 뒤
#           day3_4.99_preprocess_hdf5.py 로 uint8 로 바꾼 데이터.
#             env 출력  = rgb/255 - 프레임별평균        (float, 음수 포함)
#             학습 데이터 = (x - gmin)/(gmax - gmin)*255 (uint8)
#           평가에서도 **똑같은 상수로 똑같은 식**을 써야 정책이 학습 때와 같은
#           이미지를 본다. 이걸 빠뜨리면 정책은 새까만 화면을 입력받는다.
# =====================================================================

OBS_ENCODING = "raw"        # "raw" | "legacy"
IMG_STATS = {}              # legacy 일 때 카메라별 (gmin, gmax)


def encode_obs_image(img: torch.Tensor, obs_key: str) -> torch.Tensor:
    """
    env 카메라 한 프레임을 학습 데이터와 동일한 인코딩으로 변환한다.

    env 출력은 ObsTerm 의 normalize 설정에 따라 둘 중 하나다:
      - uint8            : normalize=False
      - float(평균제거됨) : normalize=True
    목표 인코딩은 OBS_ENCODING 이 정한다. 네 조합을 모두 처리한다.
    """
    if img.shape[-1] == 4:                 # RGBA -> RGB
        img = img[..., :3]
    env_is_uint8 = (img.dtype == torch.uint8)

    if OBS_ENCODING == "raw":  # 학습 데이터 형식
        if env_is_uint8:
            return img                     
        # env 가 normalize=True 인데 raw uint8 로 학습된 모델을 평가하려는 경우,
        # 프레임별 평균 제거가 이미 되어 있어 원본 RGB 를 완전히 복원할 수는 없다.
        # 다만 평가가 멈추지 않도록 float -> uint8 로 범위를 재표준화해 정책 입력을 가능한 범위로 맞춘다.
        x = img.float()

        # 이미지가 이미 0~1 범위면 그 상태로 사용하고,
        # 음수/중심화된 값은 프레임 내 min-max 로 재정규화한다.
        if x.min() < 0:
            x = x - x.min()
            x = x / (x.max() + 1e-6)
        else:
            x = x.clamp(0.0, 1.0)

        return (x * 255.0).clamp(0, 255).to(torch.uint8)

    # ---- LEGACY (normalize=True 재현) ----
    # 이 경로는 옛 데이터로 학습한 체크포인트 재현 전용이며, 변환 로직은
    # day3_5_legacy_img_norm.py 로 분리돼 있다.
    return legacy.encode_legacy(img, obs_key, IMG_STATS, env_is_uint8)


# 리셋 직후 정렬 액션 — 데이터 수집 스크립트가 쓰는 값과 동일해야 한다.
# arm_action 은 절대 pose IK 이므로 zeros(8) 은 "EE 를 원점으로, 쿼터니언 (0,0,0,0)"
# 이라는 잘못된 명령이 된다. 형식: [x, y, z, qw, qx, qy, qz, gripper]
#   pusht     : day3_2_pusht_teleop_collect_data_answer.py
#   pickplace : day3_1_pickplace_teleop_collect_data_answer.py
ALIGN_ACTIONS = {
    "pusht": torch.tensor([[0.4, 0.0, 0.005, 0.0, 1.0, 0.0, 0.0, -1.0]]),
    "pickplace": torch.tensor([[0.46590596437454224, 4.9243681132793427e-08,
                                0.38296937942504883, 0.0, 1.0, 0.0, 0.0, 1.0]]),
}


def preprocess_obs(obs_policy: dict, device: torch.device) -> dict:
    """env obs 한 스텝을 robomimic 단일 프레임 형식으로 변환 (배치/시간축 없음)."""
    out = {}
    for key, val in obs_policy.items():
        if val.ndim == 3:                       # (H, W, C) 이미지
            out[key] = encode_obs_image(val.to(device), key)
        else:
            out[key] = val.float().to(device)
    return out


class ObsFrameStacker:
    """
    최근 frame_stack 개의 관측을 유지해 (1, T, ...) 텐서를 만든다.

    학습 시 SequenceDataset 은 서로 다른 연속 프레임을 쌓아 넣는다.
    현재 프레임을 T번 복제하면 정책이 움직임 정보를 전혀 못 보므로,
    실제 과거 프레임을 버퍼에 보관한다.
    에피소드 시작 시엔 첫 프레임으로 채운다 (robomimic pad_frame_stack=True 와 동일).
    """

    def __init__(self, frame_stack: int = 2):
        self.frame_stack = frame_stack
        self.buffer = None

    def reset(self):
        self.buffer = None

    def add(self, obs: dict):
        self.buffer = [obs] * self.frame_stack if self.buffer is None \
            else self.buffer[1:] + [obs]

    def get_batched(self) -> dict:
        assert self.buffer is not None, "add() 를 먼저 호출하세요."
        return {k: torch.stack([f[k] for f in self.buffer], dim=0).unsqueeze(0)
                for k in self.buffer[0]}


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
    global OBS_ENCODING, IMG_STATS
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda:0")

    IMG_STATS = legacy.resolve_img_stats(args_cli.checkpoint,
                                         os.path.join(_day3_dir, "datasets"))
    OBS_ENCODING = "legacy" if IMG_STATS else "raw"
    why = ("stats 있음(사이드카/데이터셋) -> legacy" if IMG_STATS
           else "stats 없음 -> normalize=False 로 수집된 데이터로 간주")
    print(f"[EVAL] obs_encoding -> '{OBS_ENCODING}' ({why})")
    if OBS_ENCODING == "legacy":
        for k, (lo, hi) in IMG_STATS.items():
            print(f"[EVAL]   {k:12s} gmin={lo:.6f} gmax={hi:.6f}")

    # ---- Load policy ----
    print(f"[EVAL] Loading checkpoint: {args_cli.checkpoint}")
    policy = load_policy(args_cli.checkpoint, device)
    print(f"[EVAL] Policy loaded successfully.")

    # ---- 체크포인트에서 frame_stack 과 학습 해상도를 읽는다 ----
    frame_stack, ckpt_img_shapes = 2, {}
    ckpt_obs_keys = None   # 학습에 실제로 쓴 obs 키(예: top_cam 단독). None 이면 필터 안 함.
    try:
        ck = FileUtils.load_dict_from_checkpoint(ckpt_path=args_cli.checkpoint)
        cfg = ck["config"]
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        frame_stack = int(cfg["train"]["frame_stack"])
        # multi-dataset 학습 체크포인트는 shape_metadata 가 list(데이터셋별 1개)다.
        # 모든 항목의 obs shape 은 동일하므로 첫 항목을 사용한다. (load_policy 와 동일 처리)
        sm = ck["shape_metadata"]
        if isinstance(sm, list):
            sm = sm[0]
        ckpt_obs_keys = set(sm["all_shapes"].keys())
        for k, shape in sm["all_shapes"].items():
            if len(shape) == 3:
                ckpt_img_shapes[k] = (int(shape[1]), int(shape[2]))
        print(f"[EVAL] frame_stack={frame_stack}  학습 해상도={ckpt_img_shapes}")
        print(f"[EVAL] 학습 obs 키={sorted(ckpt_obs_keys)}")
    except Exception as e:
        print(f"[WARN] 체크포인트 메타데이터를 읽지 못했습니다: {e}")
    stacker = ObsFrameStacker(frame_stack)

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
    CAM_TO_OBS_KEY = {"camera": "wrist_cam", "top_camera": "top_cam"}
    for cam_name, obs_key in CAM_TO_OBS_KEY.items():
        if not hasattr(env_cfg.scene, cam_name):
            continue
        cam_cfg = getattr(env_cfg.scene, cam_name)
        if not hasattr(cam_cfg, "height"):
            continue
        used_ckpt = obs_key in ckpt_img_shapes
        h, w = ckpt_img_shapes.get(obs_key, (cam_cfg.height, cam_cfg.width))
        cam_cfg.height, cam_cfg.width = h, w
        src = "from checkpoint" if used_ckpt else "fallback(env default)"
        print(f"[EVAL] Obs camera '{cam_name}' ({obs_key}) → {h}x{w}  [{src}]")
        # (ckpt_obs_keys 가 None = 메타데이터를 아예 못 읽음 → 모든 카메라가 의심 대상)
        model_uses = (ckpt_obs_keys is None) or (obs_key in ckpt_obs_keys)
        if not used_ckpt and model_uses:
            print(f"  <-- 경고: '{obs_key}' 는 정책 입력인데 학습 해상도를 확인하지 못해 "
                  f"env 기본값({h}x{w})을 사용합니다. 학습 해상도와 다르면 성능이 무너집니다.")

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
        print(f"[EVAL] 경고: 성공 판정 함수가 없습니다. 성공률이 항상 0%로 나옵니다.")
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
    _algo_cfg = policy.policy.algo_config
    _Ta = _algo_cfg.horizon.action_horizon
    _sampler = (f"ddim{_algo_cfg.ddim.num_inference_timesteps}" if _algo_cfg.ddim.enabled
                else f"ddpm{_algo_cfg.ddpm.num_inference_timesteps}")
    eval_dir = os.path.join(run_dir, f"eval_{args_cli.task_type}_{ckpt_stem}_Ta{_Ta}_{_sampler}")
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
    # 평가 지표는 성공률뿐이다. 이 환경(ManagerBasedEnv)은 보상을 산출하지 않으며,
    # DP 평가의 표준(IsaacLab robomimic play.py)도 성공 판정만 사용한다.
    success_count = 0
    rollout_results = []

    for ep in range(args_cli.num_rollouts):
        # 초기 배치 재현: reset_tbar_left_right 가 torch.randint/torch.rand 로
        # 좌우·y오프셋·x노이즈·yaw 를 뽑으므로, reset 직전 전역 시드를 고정하면
        # rollout i 는 항상 같은 초기 상태가 된다. (eval_seed 미지정 시 랜덤 유지)
        if args_cli.eval_seed is not None:
            s = args_cli.eval_seed + ep
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            np.random.seed(s)
        obs_full, _ = env.reset()

        # ---- 리셋 직후 정렬(Align) ----
        # arm_action 은 use_relative_mode=False 인 절대 pose IK 다.
        # 데이터 수집 스크립트도 리셋 후 같은 정렬 액션을 10스텝 보낸 뒤 녹화를 시작하므로,
        # 평가에서도 동일하게 해야 정책이 학습 때와 같은 초기 상태에서 출발한다.
        align = ALIGN_ACTIONS.get(args_cli.task_type)
        if align is not None:
            a = align.to(device).repeat(env.num_envs, 1)
            if a.shape[-1] == env.action_manager.total_action_dim:
                for _ in range(10):
                    obs_full = env.step(a)[0]
            elif ep == 0:
                print(f"[WARN] align action dim {a.shape[-1]} != "
                      f"env {env.action_manager.total_action_dim} -> 정렬 생략")

        policy.start_episode()
        stacker.reset()

        ep_success = False
        # 목표 접근도 추적: 성공/실패만으로는 "정밀도 부족"인지 "물체를 안 건드림"인지 모른다.
        start_pos_err = start_yaw_err = None
        best_pos_err = best_yaw_err = float("inf")
        last_pos_err = last_yaw_err = float("nan")
        frames = []

        for step in range(args_cli.max_steps):
            with torch.no_grad():
                # Get observations from env (dict mode)
                obs_policy = obs_full["policy"]
                # Use first env only (index 0)
                obs_single = {k: v[0] for k, v in obs_policy.items()}
                # 학습에 쓴 obs 키만 남긴다. env 는 top_cam·wrist_cam 을 모두 내보내므로,
                # top_cam 단독으로 학습한 모델에 wrist_cam 을 넘기면 robomimic 이 그 키를
                # 'assumed low_dim' 으로 잘못 등록해 정책 입력을 오염시킨다.
                if ckpt_obs_keys is not None:
                    obs_single = {k: v for k, v in obs_single.items() if k in ckpt_obs_keys}

                # Debug: print obs info on first step of first rollout
                if ep == 0 and step == 0:
                    missing = (ckpt_obs_keys - set(obs_single.keys())) if ckpt_obs_keys else set()
                    if missing:
                        print(f"[WARN] 학습 obs 키 중 env 에 없는 것: {sorted(missing)}")
                    print(f"[EVAL] policy 입력 obs 키: {sorted(obs_single.keys())}")
                    print(f"\n[DEBUG] === Observation Debug Info ===")
                    for k, v in obs_single.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                              f"min={v.min().item():.4f}, max={v.max().item():.4f}")

                    # ---- env 의 normalize 설정과 인코딩 정합성 확인 ----
                    img_dtypes = [v.dtype for v in obs_single.values() if v.ndim == 3]
                    if img_dtypes:
                        env_uint8 = all(dt == torch.uint8 for dt in img_dtypes)
                        env_norm = "False(raw uint8)" if env_uint8 else "True(float, 평균제거)"
                        print(f"[EVAL] env 카메라 normalize={env_norm}  ↔  obs_encoding={OBS_ENCODING}")
                        if OBS_ENCODING == "raw" and not env_uint8:
                            print("  <-- 경고: raw 인코딩인데 env 가 float 를 출력합니다. "
                                  "env cfg 의 이미지 obs 를 normalize=False 로 맞추거나, "
                                  "legacy 모델이면 학습 hdf5/사이드카의 정규화 상수가 필요합니다.")

                # Convert to robomimic batched format
                stacker.add(preprocess_obs(obs_single, device))
                obs_batched = stacker.get_batched()

                if ep == 0 and step == 0:
                    print(f"[DEBUG] === After encode({OBS_ENCODING}) + frame stacking ===")
                    for k, v in obs_batched.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                              f"min={v.min().item():.4f}, max={v.max().item():.4f}")
                    # 인코딩이 맞는지 즉시 확인
                    for k, v in obs_batched.items():
                        if v.dtype == torch.uint8:
                            nz = (v > 0).float().mean().item() * 100
                            warn = "   <-- 경고: 인코딩 불일치 확인" if nz < 50 else ""
                            print(f"  [CHECK] {k}: nonzero={nz:.1f}% "
                                  f"mean={v.float().mean():.1f}{warn}")

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
                # 목표까지의 오차를 매 스텝 기록 (성공 여부와 무관)
                if args_cli.task_type == "pusht":
                    try:
                        pe, ye = pusht_goal_error(env)
                        if start_pos_err is None:
                            start_pos_err, start_yaw_err = pe, ye
                        last_pos_err, last_yaw_err = pe, ye
                        best_pos_err = min(best_pos_err, pe)
                        best_yaw_err = min(best_yaw_err, ye)
                    except Exception as e:
                        if ep == 0 and step == 0:
                            print(f"[WARN] 목표 오차 계산 실패: {e}")

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

        status = "\u2713 SUCCESS" if ep_success else "\u2717 FAIL"
        print(f"  Rollout {ep+1:3d}/{args_cli.num_rollouts} | "
              f"Steps: {step+1:4d} | {status}")

        entry = {
            "rollout": ep + 1,
            "steps": step + 1,
            "success": ep_success,
        }
        if start_pos_err is not None:
            moved = start_pos_err - last_pos_err   # 양수면 목표에 가까워진 것
            print(f"      목표오차 | 시작 pos={start_pos_err:.4f} yaw={start_yaw_err:.4f}")
            print(f"               | 최종 pos={last_pos_err:.4f} yaw={last_yaw_err:.4f}"
                  f"   (물체 이동 {moved:+.4f}m)")
            print(f"               | 최소 pos={best_pos_err:.4f} yaw={best_yaw_err:.4f}"
                  f"   [성공 기준 pos<0.05 yaw<0.15]")
            entry.update({
                "start_pos_error": start_pos_err, "start_yaw_error": start_yaw_err,
                "final_pos_error": last_pos_err, "final_yaw_error": last_yaw_err,
                "min_pos_error": best_pos_err, "min_yaw_error": best_yaw_err,
                "pos_improvement": moved,
            })
        rollout_results.append(entry)

        # Save video
        if save_video and len(frames) > 0:
            vid_path = os.path.join(video_dir, f"rollout_{ep:03d}.mp4")
            imageio.mimsave(vid_path, frames, fps=args_cli.video_fps)

    # ---- Summary ----
    success_rate = success_count / args_cli.num_rollouts * 100

    summary = {
        "checkpoint": ckpt_abs,
        "task": args_cli.task_type,
        "num_rollouts": args_cli.num_rollouts,
        "max_steps": args_cli.max_steps,
        "success_count": success_count,
        "success_rate": success_rate,
        "rollouts": rollout_results,
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Task:       {args_cli.task_type}")
    print(f"  Rollouts:     {args_cli.num_rollouts}")
    print(f"  Success Rate: {success_count}/{args_cli.num_rollouts} ({success_rate:.1f}%)")

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
