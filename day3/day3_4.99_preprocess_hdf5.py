#!/usr/bin/env python3
"""
Isaac Lab HDF5 데이터셋 전처리: 이미지 Resize + float32→uint8 변환.

Isaac Lab RecorderManager는 카메라 이미지를 raw float32로 저장하지만,
robomimic은 uint8 [0, 255]를 전제합니다. 이 스크립트가 두 단계를 한 번에 처리:

  1) 이미지를 지정 해상도로 resize (예: 480×640 → 240×320)
  2) float32 → uint8 [0, 255] min-max 정규화 변환

나머지 데이터(actions, joint_pos, joint_vel, states, attrs)는 그대로 복사합니다.
!! 변환 전에 데이터셋 구조를 먼저 확인하세요. !!

Usage
-----
# 구조 확인 (dry-run):
    python day3_4.99_preprocess_hdf5.py -i datasets/raw.hdf5 --dry-run

# resize + uint8 변환:
    python day3_4.99_preprocess_hdf5.py \\
        -i datasets/tbar_pickplace_teleop.hdf5 \\
        -o datasets/tbar_pickplace_teleop_resized.hdf5 \\
        --height 240 --width 320

# uint8 변환만 (원본 해상도):
    python day3_4.99_preprocess_hdf5.py \\
        -i datasets/raw.hdf5 \\
        -o datasets/processed.hdf5 \\
        --no_resize
"""

import argparse
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")

try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ──────────────────────────────────────────────────
#  Utility functions
# ──────────────────────────────────────────────────

def is_image(dset):
    """Camera obs are stored as (T, H, W, C) or (T, H, W)."""
    if not isinstance(dset, h5py.Dataset):
        return False
    if dset.ndim == 4:
        return dset.shape[-1] in (1, 3, 4)
    return dset.ndim == 3 and dset.shape[1] > 8 and dset.shape[2] > 8


def pick_interp(src_hw, dst_hw):
    return cv2.INTER_AREA if (dst_hw[0] * dst_hw[1]) < (src_hw[0] * src_hw[1]) else cv2.INTER_CUBIC


def resize_frames(arr, dst_hw):
    """Resize a (T, H, W[, C]) array frame-by-frame."""
    T = arr.shape[0]
    interp = pick_interp(arr.shape[1:3], dst_hw)
    dsize = (dst_hw[1], dst_hw[0])  # cv2 wants (width, height)
    if arr.ndim == 3:
        out = np.empty((T, dst_hw[0], dst_hw[1]), dtype=arr.dtype)
        for i in range(T):
            out[i] = cv2.resize(arr[i], dsize, interpolation=interp)
    else:
        C = arr.shape[3]
        out = np.empty((T, dst_hw[0], dst_hw[1], C), dtype=arr.dtype)
        for i in range(T):
            r = cv2.resize(arr[i], dsize, interpolation=interp)
            out[i] = r[..., None] if r.ndim == 2 else r
    return out


def float_to_uint8(arr, global_min, global_max):
    """Normalize float32 array to uint8 [0, 255] using global min/max."""
    arr = arr.astype(np.float32)
    drange = global_max - global_min
    if drange > 1e-6:
        arr = (arr - global_min) / drange
    else:
        arr = np.zeros_like(arr)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────
#  Global min/max 계산 (float→uint8 용)
# ──────────────────────────────────────────────────

def compute_global_stats(f, img_keys):
    """Compute global min/max per image key across all demos."""
    demos = sorted(f["data"].keys())
    stats = {k: {"min": float("inf"), "max": float("-inf")} for k in img_keys}

    for demo_name in tqdm(demos, desc="scanning min/max"):
        obs = f["data"][demo_name]["obs"]
        for key in img_keys:
            arr = obs[key]
            # Sample every 20th frame for speed
            indices = list(range(0, arr.shape[0], max(1, arr.shape[0] // 20)))
            sample = arr[indices]
            stats[key]["min"] = min(stats[key]["min"], float(sample.min()))
            stats[key]["max"] = max(stats[key]["max"], float(sample.max()))

    return stats


# ──────────────────────────────────────────────────
#  Core: copy + resize + uint8 변환
# ──────────────────────────────────────────────────

def copy_group(src, dst, dst_hw, cameras, do_resize, img_stats, stats, in_obs=False):
    """Recursively copy, resizing and converting images to uint8."""
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.create_group(name)
            copy_group(item, child, dst_hw, cameras, do_resize, img_stats, stats,
                       in_obs=in_obs or name in ("obs", "next_obs"))
        else:
            is_img = in_obs and is_image(item) and (cameras is None or name in cameras)
            if is_img:
                arr = item[()].astype(np.float32)

                # Step 1: Resize
                old_shape = arr.shape[1:]
                if do_resize:
                    arr = resize_frames(arr, dst_hw)
                new_shape = arr.shape[1:]

                # Step 2: float32 → uint8
                if name in img_stats and arr.dtype != np.uint8:
                    s = img_stats[name]
                    arr = float_to_uint8(arr, s["min"], s["max"])

                # Step 3: Save with compression to prevent file size bloat
                comp = item.compression if item.compression else "gzip"
                comp_opts = item.compression_opts if item.compression_opts else 4

                d = dst.create_dataset(
                    name, 
                    data=arr, 
                    chunks=(1,) + arr.shape[1:],
                    compression=comp,
                    compression_opts=comp_opts
                )
                for k, v in item.attrs.items():
                    d.attrs[k] = v
                stats.setdefault(name, {
                    "old_shape": old_shape,
                    "new_shape": new_shape,
                    "old_dtype": str(item.dtype),
                    "new_dtype": str(arr.dtype),
                })
            else:
                src.copy(name, dst, name)


# ──────────────────────────────────────────────────
#  Inspect (dry-run)
# ──────────────────────────────────────────────────

def inspect(path):
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        if "data" not in f:
            print("  ! no top-level 'data' group")
            print("    top-level keys:", list(f.keys()))
            return
        demos = list(f["data"].keys())
        print(f"  demos: {len(demos)}   (e.g. {demos[:3]})")
        d0 = f["data"][demos[0]]
        print(f"  structure of '{demos[0]}':")

        def show(item, indent):
            pad = "  " * indent
            for name, child in item.items():
                if isinstance(child, h5py.Group):
                    print(f"{pad}{name}/")
                    show(child, indent + 1)
                else:
                    tag = "   <-- image" if is_image(child) else ""
                    print(f"{pad}{name:16s} {str(child.shape):24s} {child.dtype}{tag}")

        show(d0, 2)
        if "mask" in f:
            print("  filter keys (mask/):", list(f["mask"].keys()))


# ──────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Input HDF5 file.")
    ap.add_argument("-o", "--output", help="Output HDF5 file.")
    ap.add_argument("--height", type=int, default=240, help="Target height (default: 240).")
    ap.add_argument("--width", type=int, default=320, help="Target width (default: 320).")
    ap.add_argument("--no_resize", action="store_true",
                    help="Skip resize, only do float32→uint8 conversion.")
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="Specific camera keys to process (default: auto-detect).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print structure and exit without writing.")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input not found: {args.input}")

    inspect(args.input)
    if args.dry_run:
        return

    if not args.output:
        sys.exit("Provide -o/--output (or use --dry-run to only inspect)")
    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"Output exists: {args.output}   (pass --overwrite to replace)")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    do_resize = not args.no_resize
    dst_hw = (args.height, args.width)
    cameras = set(args.cameras) if args.cameras else None

    # ---- Step 1: Find image keys & compute global min/max ----
    with h5py.File(args.input, "r") as f:
        demos = sorted(f["data"].keys())
        d0 = f["data"][demos[0]]
        img_keys = []
        if "obs" in d0:
            for key in d0["obs"]:
                if is_image(d0["obs"][key]) and (cameras is None or key in cameras):
                    img_keys.append(key)

        img_stats = {}
        needs_conversion = False
        for key in img_keys:
            if d0["obs"][key].dtype != np.uint8:
                needs_conversion = True
                break

        if needs_conversion:
            print(f"\n[1/2] Computing global min/max for float32→uint8 conversion...")
            img_stats = compute_global_stats(f, img_keys)
            for key in img_keys:
                if key in img_stats:
                    s = img_stats[key]
                    print(f"  {key}: min={s['min']:.4f}, max={s['max']:.4f}")
        else:
            print(f"\nImages already uint8, skipping float conversion.")

    # ---- Step 2: Resize + Convert ----
    actions = []
    if do_resize:
        actions.append(f"resize→{dst_hw[0]}×{dst_hw[1]}")
    if needs_conversion:
        actions.append("float32→uint8")
    print(f"\n[2/2] Processing: {' + '.join(actions)}")

    stats = {}
    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        for name, item in tqdm(list(fin.items()), desc="processing"):
            if isinstance(item, h5py.Group):
                grp = fout.create_group(name)
                copy_group(item, grp, dst_hw, cameras, do_resize, img_stats, stats)
            else:
                fin.copy(name, fout, name)
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

    # ---- Summary ----
    if stats:
        print(f"\nProcessed image observations:")
        for k, s in stats.items():
            print(f"  {k:16s} {s['old_shape']} ({s['old_dtype']}) → "
                  f"{s['new_shape']} ({s['new_dtype']})")

    old_mb = os.path.getsize(args.input) / 1e6
    new_mb = os.path.getsize(args.output) / 1e6
    ratio = 100 * new_mb / old_mb if old_mb > 0 else 0
    print(f"\nFile size: {old_mb:,.0f} MB → {new_mb:,.0f} MB  ({ratio:.0f}%)")

    # Verify
    with h5py.File(args.output, "r") as f:
        demos = list(f["data"].keys())
        d0 = f["data"][demos[0]]
        print(f"\nVerify ({demos[0]}):")
        for key in img_keys:
            if key in d0["obs"]:
                ds = d0["obs"][key]
                sample = ds[0]
                print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}, "
                      f"min={sample.min()}, max={sample.max()}")
        if "actions" in d0:
            print(f"  actions: {d0['actions'].shape} ✓")

    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
