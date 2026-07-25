#!/usr/bin/env python3
"""
학습 전 config ↔ 데이터셋 정합성 검사 (pre-flight check).

Diffusion Policy 학습(robomimic train.py)을 실행하기 전에, 작성한 config가
데이터셋과 실제로 일치하는지 먼저 확인합니다. config 빈칸을 잘못 채우면 학습이
즉시 오류로 중단되거나, 잘못된 입력(검은 화면)으로 학습이 진행되어 시간을
낭비할 수 있습니다.

검사 항목
---------
  [ERROR]  학습이 실패하거나 무의미해지는 항목 — 발견 시 exit 1
    - "???" 미입력 빈칸
    - "_" 로 시작하는 힌트 키 잔존 (robomimic key-lock 이 거부)
    - config 의 low_dim / rgb key 가 데이터셋 obs 에 존재하지 않음
    - crop 크기가 이미지보다 큼
  [WARN]   동작은 하나 성능/공정성에 영향을 주는 항목 — 학습은 진행
    - 이미지가 float32 (normalize=True 데이터). uint8 변환 또는 normalize=False 재수집 권장
    - crop 이 이미지의 90% 내외가 아님
    - pretrained=false (소규모 데이터에서 성능 저하)

마지막에 "이 config의 학습 구성" 요약(카메라·pretrained·horizon·
샘플러·프레임당 학습 횟수)을 출력합니다.

Usage
-----
    python check_train_config.py --config configs/xxx.json --dataset datasets/yyy.hdf5
"""

import argparse
import json
import sys

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")


G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; X = "\033[0m"


def is_image_shape(shape):
    """(T,H,W,C) or (T,H,W) 카메라 obs 판별."""
    if len(shape) == 4:
        return shape[-1] in (1, 3, 4)
    return len(shape) == 3 and shape[1] > 8 and shape[2] > 8


def find_placeholder_keys(obj, path=""):
    """'_' 로 시작하는 힌트 키와 '???' 값을 재귀적으로 찾는다."""
    bad_keys, blanks = [], []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if isinstance(k, str) and k.startswith("_"):
                bad_keys.append(p)
                continue  # 삭제 대상 subtree — 내부 "???" 는 검사하지 않음
            bk, bl = find_placeholder_keys(v, p)
            bad_keys += bk; blanks += bl
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bk, bl = find_placeholder_keys(v, f"{path}[{i}]")
            bad_keys += bk; blanks += bl
    elif isinstance(obj, str) and "???" in obj:
        blanks.append(path)
    return bad_keys, blanks


def _open_ro(path):
    """읽기 전용 open. 다른 프로세스가 파일을 열고 있어도(학습/전처리 중) 락 충돌 없이 연다."""
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        # 구버전 h5py: locking 인자 미지원 → 환경변수로 대체
        import os as _os
        _os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        return h5py.File(path, "r")


def inspect_dataset(path):
    """데이터셋에서 obs key / 이미지 shape·dtype / 총 프레임 수를 뽑는다."""
    with _open_ro(path) as f:
        if "data" not in f:
            sys.exit(f"{R}데이터셋에 'data' 그룹이 없습니다: {path}{X}")
        demos = sorted(f["data"].keys())
        d0 = f["data"][demos[0]]
        obs = d0["obs"]
        low_dim_keys, img_info = [], {}
        for k in obs:
            shp = obs[k].shape
            if is_image_shape(shp):
                img_info[k] = {"shape": tuple(shp[1:]), "dtype": str(obs[k].dtype)}
            else:
                low_dim_keys.append(k)
        total_frames = 0
        for dm in demos:
            total_frames += int(f["data"][dm]["actions"].shape[0])
        action_dim = int(d0["actions"].shape[-1])
    return {
        "n_demos": len(demos),
        "low_dim_keys": low_dim_keys,
        "img_info": img_info,     # {key: {shape:(H,W,C), dtype}}
        "total_frames": total_frames,
        "action_dim": action_dim,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="학습 config JSON")
    ap.add_argument("--dataset", required=True, help="학습 데이터셋 HDF5")
    args = ap.parse_args()

    with open(args.config) as fp:
        raw = fp.read()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"{R}config JSON 파싱 실패: {e}{X}")

    ds = inspect_dataset(args.dataset)

    errors, warns = [], []

    # ── 1. 미입력 빈칸 / 힌트 키 잔존 ────────────────────────────
    bad_keys, blanks = find_placeholder_keys(cfg)
    for p in blanks:
        errors.append(f'미입력 빈칸 "???" 이 존재합니다: {p}')
    for p in bad_keys:
        errors.append(f'힌트 키가 잔존합니다 (robomimic 이 거부): {p}  → 해당 항목을 삭제하십시오')

    # ── config 에서 필요한 값 추출 (없으면 안전하게) ──────────────
    obs_mod = cfg.get("observation", {}).get("modalities", {}).get("obs", {})
    cfg_low_dim = [k for k in obs_mod.get("low_dim", []) if "???" not in str(k)]
    cfg_rgb = [k for k in obs_mod.get("rgb", []) if "???" not in str(k)]

    rgb_enc = cfg.get("observation", {}).get("encoder", {}).get("rgb", {})
    crop_kw = rgb_enc.get("obs_randomizer_kwargs", {})
    crop_h = crop_kw.get("crop_height")
    crop_w = crop_kw.get("crop_width")
    pretrained = rgb_enc.get("core_kwargs", {}).get("backbone_kwargs", {}).get("pretrained")

    # ── 2. obs key 존재 확인 ────────────────────────────────────
    ds_obs_all = set(ds["low_dim_keys"]) | set(ds["img_info"].keys())
    for k in cfg_low_dim:
        if k not in ds["low_dim_keys"]:
            hint = " (이미지 key 를 low_dim 에 지정함)" if k in ds["img_info"] else ""
            errors.append(f'config low_dim 의 "{k}" 가 데이터셋에 존재하지 않습니다{hint}. '
                          f'데이터셋 low_dim: {ds["low_dim_keys"]}')
    for k in cfg_rgb:
        if k not in ds["img_info"]:
            hint = " (저차원 key 를 rgb 에 지정함)" if k in ds["low_dim_keys"] else ""
            errors.append(f'config rgb 의 "{k}" 가 데이터셋 카메라에 존재하지 않습니다{hint}. '
                          f'데이터셋 카메라: {list(ds["img_info"].keys())}')

    # ── 3. crop vs 이미지 크기 / dtype ──────────────────────────
    used_imgs = [k for k in cfg_rgb if k in ds["img_info"]]
    for k in used_imgs:
        H, W = ds["img_info"][k]["shape"][0], ds["img_info"][k]["shape"][1]
        dt = ds["img_info"][k]["dtype"]
        if crop_h and crop_w:
            if crop_h > H or crop_w > W:
                errors.append(f'crop({crop_h}×{crop_w}) 가 "{k}" 이미지({H}×{W}) 보다 큽니다.')
            else:
                rh, rw = crop_h / H, crop_w / W
                if not (0.8 <= rh <= 0.98 and 0.8 <= rw <= 0.98):
                    warns.append(f'crop({crop_h}×{crop_w}) 가 "{k}"({H}×{W}) 의 '
                                 f'{rh*100:.0f}%×{rw*100:.0f}% 입니다 — 일반적으로 90% 내외를 사용합니다.')
        if dt != "uint8":
            warns.append(f'"{k}" 이미지가 {dt} 입니다(uint8 아님). normalize=True 로 수집된 데이터이므로, '
                         f'day3_4.99 로 uint8 변환하거나 normalize=False 로 재수집하는 것을 권장합니다.')

    # ── 4. pretrained ──────────────────────────────────────────
    if pretrained is False and ds["n_demos"] <= 100:
        warns.append(f'데모 수가 {ds["n_demos"]}개로 적은데 pretrained=false 입니다. '
                     f'true 로 설정 시 성공률이 크게 향상됩니다.')

    # ── 5. action dim ──────────────────────────────────────────
    # (env action_dim 은 여기서 알 수 없으므로 데이터셋 기준만 보고)

    # ── 결과 출력 ───────────────────────────────────────────────
    print(f"\n{B}══ 학습 전 검사 ══{X}")
    print(f"  config : {args.config}")
    print(f"  dataset: {args.dataset}")
    print(f"           demos={ds['n_demos']}, frames={ds['total_frames']:,}, action_dim={ds['action_dim']}")
    print(f"           low_dim={ds['low_dim_keys']}")
    print(f"           cameras={ {k: v['shape'] for k, v in ds['img_info'].items()} }")

    if errors:
        print(f"\n{R}{B}✗ ERROR {len(errors)}건 — 학습을 진행할 수 없습니다{X}")
        for e in errors:
            print(f"  {R}✗{X} {e}")
    if warns:
        print(f"\n{Y}{B}⚠ WARN {len(warns)}건 — 학습은 가능하나 확인이 필요합니다{X}")
        for w in warns:
            print(f"  {Y}⚠{X} {w}")
    if not errors and not warns:
        print(f"\n{G}{B}✓ 문제 없음{X}")

    # ── 학습 계획 요약 ─────────────────────────────────────────
    if not errors:
        hz = cfg.get("algo", {}).get("horizon", {})
        To = hz.get("observation_horizon"); Ta = hz.get("action_horizon"); Tp = hz.get("prediction_horizon")
        ddpm = cfg.get("algo", {}).get("ddpm", {})
        ddim = cfg.get("algo", {}).get("ddim", {})
        if ddim.get("enabled"):
            sampler = f"DDIM ({ddim.get('num_inference_timesteps')} steps)"
        else:
            sampler = f"DDPM ({ddpm.get('num_inference_timesteps')} steps)"
        tr = cfg.get("train", {})
        bs = tr.get("batch_size"); n_ep = tr.get("num_epochs")
        eps = cfg.get("experiment", {}).get("epoch_every_n_steps")

        print(f"\n{B}══ 이 config의 학습 구성 ══{X}")
        print(f"  입력 관측 : 카메라 {used_imgs}  +  low_dim {cfg_low_dim}")
        print(f"  시각 인코더: ResNet18  (pretrained={pretrained})"
              + ("   ← 소규모 데이터에서 가장 큰 성능 결정 요인" if pretrained else ""))
        print(f"  horizon   : To={To} 프레임 관찰 → Tp={Tp} 액션 예측 → Ta={Ta} 실행 후 재계획")
        print(f"  샘플러    : {sampler}")
        if bs and n_ep and eps:
            passes = n_ep * eps * bs
            per_frame = passes / ds["total_frames"] if ds["total_frames"] else 0
            print(f"  학습량    : {n_ep} epochs × {eps} steps × batch {bs} = {passes:,} 프레임패스")
            print(f"              = 프레임당 평균 {per_frame:.1f} 회 학습 "
                  f"(데이터 증가 시 epoch_every_n_steps 도 비례하여 상향해야 공정한 비교가 가능)")
        print()

    if errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
