#!/usr/bin/env bash
# ============================================================
# Docker build-time setup for the AILAB summer-school image.
#
# Installs the project Python deps + robomimic and downloads the course data and
# local model checkpoints
# data, ALL baked into the image so it is self-contained: the image can be
# `docker save`d and run on fresh workstations with no host repo and no
# downloads (see docker/isaaclab/README.md).
#
# The repo-root setup.sh is the bare-metal host installer and is left untouched.
# This is the Docker-adapted version (no git submodule, no host assumptions),
# called by the Dockerfile in stages so code edits don't re-run the heavy
# dependency / data layers:
#   setup_docker.sh deps | robomimic | data | verify | all
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root inside the image (/workspace/AILAB-summer-school-2026)

DRIVE_URL="https://drive.google.com/drive/folders/1R9UEEVVQ4NwvMMGxt6rcmUoqW5ILYktq"
CONTACT_GRASP_CKPT="data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth"
SAM3_CKPT="data/checkpoint/sam3/sam3.1_multiplex.pt"
ZIP_DIR="data/_zips"
DAY3_DATASET_ID="1dxN5yS4Ixa45hXilRxHyFdi0T4-aYCJZ"
DAY3_DATASET_NAME="tbar_pickplace_teleop_0719_240x320.hdf5"

# "파일ID:파일명". 각 zip 은 unzip 시 최상위에 자기 이름의 디렉토리를 갖는다 (-> data/<이름>/).
ZIPS=(
    "1U2Lx7C60gnC9REaJobkBmLk3KeOXJAlg:assets.zip"        # day2/day3 YCB assets -> data/assets/
    "1KtkR46L-ZlPS5KAeujb8FhfPA6EFnCes:checkpoint.zip"    # cgnet + SAM3 checkpoints -> data/checkpoint/
    "1ESUhUw3F39mbOeK2eFudkJAWRupB6bHK:handeye_data.zip"  # day1_4.3.1/4.3.2 -> data/handeye_data/
    "1nFmfcubM0Su2aa-08BPNx7z5SWES4aBg:slam_map_data.zip" # day1_4.3.3 -> data/slam_map_data/
    "1oS9YpR__J8qD60Mv9VOYQi5WH8w6h476:PennFudanPed.zip"  # day1 object detection -> data/PennFudanPed/
    "1ttTD9ZaWo7F-OWi9Y-_kaS1T-5h1gYpy:sam3_practice.zip" # day2 SAM3 example inputs -> data/sam3_practice/
)

do_deps() {
    echo "==> deps (requirements.txt: PyPI + sam3 via git + ./cgnet/graspnetAPI + gdown)"
    pip install -r requirements.txt
    # graspnetAPI pins transforms3d==0.3.1, which uses np.float (removed in numpy>=1.24);
    # under the pinned numpy 1.26 that breaks `import graspnetAPI`.
    pip install -U "transforms3d>=0.4.2"
}

do_robomimic() {
    echo "==> robomimic (git clone + editable install + Isaac Lab compat patch)"
    if [ ! -d day3/robomimic/.git ]; then
        git clone --depth 1 https://github.com/ARISE-Initiative/robomimic.git day3/robomimic
    fi
    pip install -e day3/robomimic
    # Isaac Lab datasets may lack env_kwargs; guard the lookup so robomimic doesn't KeyError.
    sed -i 's/if "env_lang" in env_meta\["env_kwargs"\]/if "env_kwargs" in env_meta and "env_lang" in env_meta["env_kwargs"]/' \
        day3/robomimic/robomimic/utils/file_utils.py
}

do_data() {
    command -v unzip >/dev/null || { echo "ERROR: unzip is required"; exit 1; }
    command -v gdown >/dev/null || { echo "ERROR: gdown is required (run 'deps' first)"; exit 1; }

    echo "==> data and local checkpoint zips"
    mkdir -p "$ZIP_DIR"
    for entry in "${ZIPS[@]}"; do
        id="${entry%%:*}"; name="${entry##*:}"
        if [ -f "$ZIP_DIR/$name" ]; then echo "    reuse: $name"; else echo "    download: $name"; gdown "$id" -O "$ZIP_DIR/$name"; fi
    done

    echo "==> unzip"
    for entry in "${ZIPS[@]}"; do
        name="${entry##*:}"
        unzip -qo "$ZIP_DIR/$name" -d data
    done

    # Fail in the data layer (rather than near the end of the Docker build) if
    # Google Drive points to an outdated or incorrectly packaged checkpoint zip.
    for checkpoint in "$CONTACT_GRASP_CKPT" "$SAM3_CKPT"; do
        if [ ! -f "$checkpoint" ]; then
            echo "ERROR: checkpoint.zip에 필요한 파일이 없음: $checkpoint"
            echo "checkpoint.zip 내용:"
            unzip -Z1 "$ZIP_DIR/checkpoint.zip" | sed -n '1,120p'
            exit 1
        fi
    done

    echo "==> day3 demo dataset (~3.6GB)"
    mkdir -p day3/datasets
    if [ -f "day3/datasets/$DAY3_DATASET_NAME" ]; then
        echo "    reuse: $DAY3_DATASET_NAME"
    else
        gdown "$DAY3_DATASET_ID" -O "day3/datasets/$DAY3_DATASET_NAME"
    fi

    rm -rf "$ZIP_DIR"
}

do_verify() {
    echo "==> verify data"
    # day1 notebooks run in day1/ and reference ./data, so link root data -> day1/data.
    [ -d day1 ] && ln -sfn ../data day1/data
    local missing=0
    for d in data/assets data/checkpoint; do [ -d "$d" ] || { echo "    MISSING: $d"; missing=1; }; done
    [ -f "$CONTACT_GRASP_CKPT" ] || { echo "    MISSING: $CONTACT_GRASP_CKPT"; missing=1; }
    [ -f "$SAM3_CKPT" ] || { echo "    MISSING: $SAM3_CKPT"; missing=1; }
    [ -f "day3/data/assets/t_bar/T_bar.usd" ] || { echo "    MISSING: day3/data/assets/t_bar/T_bar.usd"; missing=1; }
    [ -f "day3/data/assets/t_bar/T_bar_goal.usd" ] || { echo "    MISSING: day3/data/assets/t_bar/T_bar_goal.usd"; missing=1; }
    [ -d "data/PennFudanPed/PNGImages" ] || { echo "    MISSING: data/PennFudanPed/PNGImages"; missing=1; }
    for d in data/handeye_data data/slam_map_data data/sam3_practice; do [ -d "$d" ] || { echo "    MISSING: $d"; missing=1; }; done
    [ -f "day3/datasets/$DAY3_DATASET_NAME" ] || { echo "    MISSING: day3/datasets/$DAY3_DATASET_NAME"; missing=1; }
    if [ "$missing" -ne 0 ]; then
        echo "데이터가 온전하지 않다. 직접 받아 data/ 에 배치할 것: $DRIVE_URL"
        exit 1
    fi
    echo "    OK  assets dirs: $(find data/assets -mindepth 1 -maxdepth 1 -type d | wc -l), data total: $(du -sh data | cut -f1)"
}

case "${1:-all}" in
    deps)      do_deps ;;
    robomimic) do_robomimic ;;
    data)      do_data ;;
    verify)    do_verify ;;
    all)       do_deps; do_robomimic; do_data; do_verify ;;
    *) echo "usage: $0 {deps|robomimic|data|verify|all}"; exit 1 ;;
esac
