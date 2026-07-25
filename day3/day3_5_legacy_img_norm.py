#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# [LEGACY] normalize=True 데이터의 이미지 정규화 상수 재현 유틸
#
# 권장 파이프라인(수집 시 normalize=False, uint8 RGB)에서는 이 모듈이
# **전혀 필요 없다**. 평가 스크립트는 정규화 상수(img_norm_min/max)를
# 실제로 찾았을 때에만 이 모듈의 함수를 사용한다.
#
# 왜 이 코드가 필요한가:
#   ObsTerm 이 normalize=True (IsaacLab 기본값) 인 상태로 수집한 뒤
#   day3_4.99_preprocess_hdf5.py 로 uint8 로 바꾼 데이터는
#     env 출력  = rgb/255 - 프레임별평균        (float, 음수 포함)
#     학습 데이터 = (x - gmin)/(gmax - gmin)*255 (uint8)
#   이라, 평가에서도 **똑같은 상수로 똑같은 식**을 재현해야 정책이 학습 때와
#   같은 이미지를 본다. 상수는 데이터셋 attrs 에만 있으므로, 체크포인트가
#   기억하는 학습 hdf5 를 되찾거나, 한 번 읽은 값을 사이드카에 캐시해 둔다.
#
# 이 모듈은 IsaacLab 에 의존하지 않으므로(순수 torch/h5py) 일반 python 으로도
# import 할 수 있다(예: 사이드카 사전 생성 스크립트).
# =====================================================================

import os
import json
import glob

import torch

import robomimic.utils.file_utils as FileUtils


# =====================================================================
#  학습 데이터셋 attrs 에서 정규화 상수 읽기
# =====================================================================

def load_img_stats(train_dataset: str = None, stats_json: str = None):
    """학습 데이터의 정규화 상수(카메라별 (gmin, gmax))를 읽는다."""
    stats = {}
    if train_dataset:
        try:
            import h5py
            with h5py.File(train_dataset, "r") as f:
                demo0 = f["data"][sorted(f["data"].keys())[0]]
                for key in demo0["obs"]:
                    d = demo0["obs"][key]
                    if "img_norm_min" in d.attrs and "img_norm_max" in d.attrs:
                        stats[key] = (float(d.attrs["img_norm_min"]),
                                      float(d.attrs["img_norm_max"]))
        except Exception as e:
            print(f"[WARN] {train_dataset} 에서 정규화 상수를 읽지 못했습니다: {e}")
    if stats_json:
        stats.update({k: (float(v[0]), float(v[1]))
                      for k, v in json.loads(stats_json).items()})
    return stats


def _dataset_has_img_norm(path: str) -> bool:
    """해당 hdf5 의 첫 데모 이미지에 img_norm attrs 가 있는지(=상수 자기기술) 확인한다."""
    try:
        return bool(load_img_stats(path, None))
    except Exception:
        return False


def resolve_train_datasets_from_ckpt(checkpoint_path: str, datasets_dir: str):
    """체크포인트가 저장해 둔 학습 데이터셋 경로들을 되찾는다.

    robomimic 체크포인트의 config["train"]["data"] 에는 학습에 쓴 hdf5 경로가
    그대로 들어 있다. **단일 파일**(문자열/1개 목록)일 수도, 데이터셋을 합치지 않고
    **여러 개를 그대로 지정한 multi-dataset**([{"path": a}, {"path": b}, ...]) 일 수도 있다.
    각 hdf5 attrs 에 정규화 상수(img_norm_min/max)가 있으므로, 사용자가 경로를
    직접 주지 않아도 여기서 자동으로 찾을 수 있다.

    경로가 상대경로이거나 다른 머신에서 학습한 경우를 대비해, 원본 경로가 없으면
    파일명만 떼어 datasets_dir 아래에서도 찾아본다.

    Args:
        checkpoint_path: 체크포인트(.pth) 경로.
        datasets_dir:    파일명으로 재탐색할 로컬 datasets 디렉토리.

    Returns:
        존재하는 데이터셋 경로들의 리스트(학습 config 에 적힌 순서 유지). 없으면 빈 리스트.
    """
    try:
        ck = FileUtils.load_dict_from_checkpoint(ckpt_path=checkpoint_path)
        cfg = ck["config"]
        cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
        data = cfg["train"]["data"]
        if isinstance(data, list):
            candidates = [d["path"] if isinstance(d, dict) else d for d in data]
        else:
            candidates = [data]
    except Exception as e:
        print(f"[WARN] 체크포인트에서 학습 데이터 경로를 읽지 못했습니다: {e}")
        return []

    found = []
    for path in candidates:
        if not path:
            continue
        if os.path.isfile(path):
            found.append(path)
            continue
        # 원본 절대경로가 사라졌으면 파일명으로 로컬 datasets/ 에서 재탐색
        alt = os.path.join(datasets_dir, os.path.basename(path))
        if os.path.isfile(alt):
            found.append(alt)
            continue
        # 파일명이 바뀐 경우(예: 전처리 후 접미사 추가 ..._merged.hdf5 -> ..._merged_resized.hdf5)
        # 원본 stem 으로 시작하는 파일을 datasets/ 에서 찾는다.
        stem = os.path.splitext(os.path.basename(path))[0]
        matches = sorted(glob.glob(os.path.join(datasets_dir, stem + "*.hdf5")))
        if len(matches) > 1:
            # 후보가 여럿이면(예: uint8 정식본과 float32 중복본) 정규화 상수 attrs 를
            # 가진 것을 우선한다 — 그게 '자기 상수를 갖춘' 의도된 데이터다.
            with_attrs = [m for m in matches if _dataset_has_img_norm(m)]
            if len(with_attrs) == 1:
                matches = with_attrs
        if len(matches) == 1:
            print(f"[WARN] 학습 데이터 '{os.path.basename(path)}' 없음 -> 이름이 바뀐 "
                  f"'{os.path.basename(matches[0])}' 로 대체합니다.")
            found.append(matches[0])
        elif len(matches) > 1:
            print(f"[WARN] 학습 데이터 '{os.path.basename(path)}' 후보가 여럿이라 "
                  f"자동 선택하지 않습니다: {[os.path.basename(m) for m in matches]}")
    if not found:
        print(f"[WARN] 체크포인트가 가리키는 학습 데이터셋을 찾지 못했습니다: {candidates}")
    return found


def load_img_stats_multi(paths):
    """여러 데이터셋에서 정규화 상수를 읽어 병합한다(multi-dataset 지원).

    카메라 키가 여러 데이터셋에 다른 값으로 존재하면 경고하고 **첫 데이터셋 값**을 쓴다.
    (평가 env 의 렌더링은 보통 첫 데이터셋 = 원본 teleop 과 일치하므로 이쪽이 안전하다.)
    """
    merged = {}
    for p in paths:
        for key, val in load_img_stats(p, None).items():
            if key in merged and merged[key] != val:
                print(f"[WARN] '{key}' 정규화 상수가 데이터셋마다 다릅니다: "
                      f"{merged[key]} vs {val}. 첫 값({merged[key]})을 사용합니다.")
                continue
            merged.setdefault(key, val)
    return merged


# =====================================================================
#  정규화 상수 사이드카 캐시
#  상수(img_norm_min/max)는 데이터셋 attrs 에만 있어, 데이터셋이 삭제되면(증강
#  데이터는 용량이 커서 흔히 지워진다) 체크포인트만으로는 되살릴 수 없다. 한 번 읽은
#  상수를 체크포인트 run 디렉토리에 img_norm_stats.json 으로 저장해 두면, 이후에는
#  데이터셋 없이도 평가할 수 있다. run 디렉토리는 .gitignore 대상이라 저장소를
#  오염시키지 않는다.
# =====================================================================
SIDECAR_NAME = "img_norm_stats.json"


def _ckpt_run_dir(checkpoint_path: str) -> str:
    """체크포인트 경로에서 run 디렉토리(.../models 의 부모)를 구한다."""
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    return os.path.dirname(parent) if os.path.basename(parent) == "models" else parent


def load_stats_sidecar(checkpoint_path: str) -> dict:
    """run 디렉토리에 캐시된 정규화 상수를 읽는다(없으면 빈 dict)."""
    path = os.path.join(_ckpt_run_dir(checkpoint_path), SIDECAR_NAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return {k: (float(v[0]), float(v[1])) for k, v in json.load(f).items()}
    except Exception as e:
        print(f"[WARN] 사이드카 {path} 를 읽지 못했습니다: {e}")
        return {}


def save_stats_sidecar(checkpoint_path: str, stats: dict):
    """정규화 상수를 run 디렉토리에 캐시한다(다음 평가부터 데이터셋 불필요)."""
    if not stats:
        return
    path = os.path.join(_ckpt_run_dir(checkpoint_path), SIDECAR_NAME)
    try:
        with open(path, "w") as f:
            json.dump({k: [float(v[0]), float(v[1])] for k, v in stats.items()}, f, indent=2)
        print(f"[EVAL] 정규화 상수를 캐시했습니다: {path} (다음부터 데이터셋 없이 평가 가능)")
    except Exception as e:
        print(f"[WARN] 사이드카 저장 실패({path}): {e}")


# =====================================================================
#  고수준 진입점: 상수 자동 해결 + 인코딩 변환
# =====================================================================

def resolve_img_stats(checkpoint_path: str, datasets_dir: str) -> dict:
    """정규화 상수를 자동으로 찾는다(legacy 여부 판별용).

    우선순위:
      1) 체크포인트 옆 사이드카 캐시 (img_norm_stats.json) — 데이터셋 불필요
      2) 체크포인트가 기억하는 학습 데이터셋 경로 자동 탐지 → 읽은 뒤 사이드카에 캐시

    Returns:
        상수 dict(카메라별 (gmin, gmax)). 비어 있으면 상수가 없는 것 = raw 로 간주.
    """
    stats = load_stats_sidecar(checkpoint_path)
    if stats:
        print("[EVAL] 정규화 상수를 캐시(사이드카)에서 불러왔습니다 (데이터셋 미사용).")
        return stats
    train_datasets = resolve_train_datasets_from_ckpt(checkpoint_path, datasets_dir)
    if train_datasets:
        tag = train_datasets[0] if len(train_datasets) == 1 \
            else f"{len(train_datasets)}개 (multi): {', '.join(os.path.basename(p) for p in train_datasets)}"
        print(f"[EVAL] 학습 데이터셋 자동 탐지: {tag}")
        stats = load_img_stats_multi(train_datasets)
        save_stats_sidecar(checkpoint_path, stats)
    return stats


def encode_legacy(img: torch.Tensor, obs_key: str, stats: dict,
                  env_is_uint8: bool) -> torch.Tensor:
    """env 카메라 한 프레임을 legacy(normalize=True) 학습 데이터와 동일하게 인코딩한다.

    학습 전처리와 동일한 (x - gmin)/(gmax - gmin)*255 를 uint8 로 잘라 반환한다.
    env 가 normalize=False(uint8) 로 출력하면, normalize=True 가 하던 프레임별
    평균 제거를 여기서 먼저 재현한 뒤 상수 정규화를 적용한다.

    Args:
        img:          (H, W, 3) 이미지 텐서(RGBA 는 호출 전에 RGB 로 잘라둔다).
        obs_key:      카메라 키(예: "top_cam").
        stats:        카메라별 (gmin, gmax) 상수.
        env_is_uint8: env 카메라가 uint8 를 출력하는지 여부.
    """
    gmin, gmax = stats.get(obs_key, (None, None))
    if gmin is None:
        raise RuntimeError(
            f"legacy 인코딩인데 '{obs_key}' 의 정규화 상수를 모릅니다. "
            f"체크포인트 옆 사이드카(img_norm_stats.json) 또는 학습 hdf5 의 "
            f"img_norm attrs 를 확인하십시오."
        )
    x = img.float()
    if env_is_uint8:
        # env 는 normalize=False 지만 모델은 legacy 로 학습됨 ->
        # normalize=True 가 하던 일을 여기서 재현한다.
        x = x / 255.0
        x = x - x.mean(dim=(0, 1), keepdim=True)   # (H, W, C) 의 H,W 평균
    # 이제 x 는 env 가 normalize=True 일 때와 같은 값. 학습 전처리와 동일 식 적용.
    x = (x - gmin) / max(gmax - gmin, 1e-6) * 255.0
    # preprocess 가 (arr*255).clip().astype(uint8) 로 잘라내므로 truncation 으로 맞춘다.
    return x.clamp(0, 255).to(torch.uint8)
