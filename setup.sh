#!/usr/bin/env bash
# AILAB summer school 2026 - 의존성 설치 및 데이터 다운로드
#
# 사전 조건: Isaac Lab 이 먼저 설치되어 있어야 함 (README 의 Isaac Lab 섹션 참고)
# 사용법: 저장소 루트에서
#   bash setup.sh
#
# SAM3를 포함한 checkpoint.zip과 에셋, day3 데모 hdf5를 다운로드한다.
set -euo pipefail

cd "$(dirname "$0")"

DRIVE_URL="https://drive.google.com/drive/folders/1R9UEEVVQ4NwvMMGxt6rcmUoqW5ILYktq"
CONTACT_GRASP_CKPT="data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth"
SAM3_CKPT="data/checkpoint/sam3/sam3.1_multiplex.pt"
ZIP_DIR="data/_zips"

# day3 teleop 데모 데이터셋 (단일 hdf5, 압축 아님) -> day3/datasets/
DAY3_DATASET_ID="1dxN5yS4Ixa45hXilRxHyFdi0T4-aYCJZ"
DAY3_DATASET_NAME="tbar_pickplace_teleop_0719_240x320.hdf5"

# 다운로드할 zip 들. "파일ID:파일명" 형식. 각 zip 은 unzip 시 data/<이름>/ 으로 풀린다.
ZIPS=(
    "1U2Lx7C60gnC9REaJobkBmLk3KeOXJAlg:assets.zip"              # day2/day3 YCB 에셋 -> data/assets/
    "1KtkR46L-ZlPS5KAeujb8FhfPA6EFnCes:checkpoint.zip"          # cgnet + SAM3 체크포인트 -> data/checkpoint/
    "1ESUhUw3F39mbOeK2eFudkJAWRupB6bHK:handeye_data.zip"        # day1_4.3.1/4.3.2 -> data/handeye_data/
    "1nFmfcubM0Su2aa-08BPNx7z5SWES4aBg:slam_map_data.zip"       # day1_4.3.3 -> data/slam_map_data/
    "1oS9YpR__J8qD60Mv9VOYQi5WH8w6h476:PennFudanPed.zip"        # day1 객체 검출 -> data/PennFudanPed/
    "1ttTD9ZaWo7F-OWi9Y-_kaS1T-5h1gYpy:sam3_practice.zip"       # day2 SAM3 예제 입력 -> data/sam3_practice/
)

echo "==> 패키지 설치"
pip install -r requirements.txt

# day3 robomimic 설치
echo "==> robomimic 서브모듈 초기화 및 설치"
git submodule update --init --recursive
pip install -e day3/robomimic

# Isaac Lab 데이터셋에는 env_kwargs가 없을 수 있어 안전하게 패치
echo "==> robomimic Isaac Lab 호환 패치"
sed -i 's/if "env_lang" in env_meta\["env_kwargs"\]/if "env_kwargs" in env_meta and "env_lang" in env_meta["env_kwargs"]/' \
    day3/robomimic/robomimic/utils/file_utils.py

command -v unzip >/dev/null || { echo "unzip 이 필요하다: sudo apt install unzip"; exit 1; }

echo "==> 데이터 및 로컬 체크포인트 다운로드"
mkdir -p "$ZIP_DIR"
for entry in "${ZIPS[@]}"; do
    id="${entry%%:*}"
    name="${entry##*:}"
    # checkpoint.zip 이 Google Drive에서 갱신되기 전에 받은 구버전이면
    # SAM3 로컬 체크포인트가 없으므로 새 zip으로 다시 받는다.
    if [ "$name" = "checkpoint.zip" ] && [ -f "$ZIP_DIR/$name" ] && \
       ! unzip -Z1 "$ZIP_DIR/$name" | grep -Eq '(^|/)sam3/sam3\.1_multiplex\.pt$'; then
        echo "    구버전 checkpoint.zip 감지 (SAM3 없음): 다시 다운로드"
        gdown "$id" -O "$ZIP_DIR/$name"
    elif [ -f "$ZIP_DIR/$name" ]; then
        echo "    받아둔 파일 재사용: $name"
    else
        echo "    다운로드: $name"
        gdown "$id" -O "$ZIP_DIR/$name"
    fi
done

echo "==> 압축 해제"
for entry in "${ZIPS[@]}"; do
    name="${entry##*:}"
    # -o 로 덮어쓰기. 각 zip 은 최상위에 자기 이름의 디렉토리를 갖는다.
    unzip -qo "$ZIP_DIR/$name" -d data
done

# day1 노트북들은 day1/ 에서 실행되며 './data' 를 참조하므로, 루트 data/ 로 링크해준다.
ln -sfn ../data day1/data

echo "==> day3 데모 데이터셋 다운로드 (약 3.6GB)"
mkdir -p day3/datasets
if [ -f "day3/datasets/$DAY3_DATASET_NAME" ]; then
    echo "    받아둔 파일 재사용: $DAY3_DATASET_NAME"
else
    gdown "$DAY3_DATASET_ID" -O "day3/datasets/$DAY3_DATASET_NAME"
fi

echo "==> Jupyter 커널 등록 (day1 노트북 실습용)"
# 현재 conda 환경(isaaclab)을 노트북 커널로 등록. 등록해두면 VSCode/Jupyter 에서
# 'isaaclab' 커널을 바로 선택할 수 있다. ipykernel 은 requirements.txt 에서 설치됨.
python -m ipykernel install --user --name isaaclab --display-name "isaaclab (Python 3.11.15)"

echo "==> 결과 확인"
missing=0
for d in data/assets data/checkpoint; do
    [ -d "$d" ] || { echo "    누락: $d"; missing=1; }
done
[ -f "$CONTACT_GRASP_CKPT" ] || { echo "    누락: $CONTACT_GRASP_CKPT"; missing=1; }
[ -f "$SAM3_CKPT" ] || { echo "    누락: $SAM3_CKPT"; missing=1; }
[ -d "data/PennFudanPed/PNGImages" ] || { echo "    누락: data/PennFudanPed/PNGImages"; missing=1; }
for d in data/handeye_data data/slam_map_data data/sam3_practice; do
    [ -d "$d" ] || { echo "    누락: $d"; missing=1; }
done
[ -f "day3/datasets/$DAY3_DATASET_NAME" ] || { echo "    누락: day3/datasets/$DAY3_DATASET_NAME"; missing=1; }

if [ "$missing" -ne 0 ]; then
    echo
    echo "데이터가 온전하지 않다. $ZIP_DIR 를 지우고 다시 실행하거나,"
    echo "아래에서 직접 받아 data/ 에 배치할 것:"
    echo "  $DRIVE_URL"
    exit 1
fi

echo "    assets  : $(find data/assets -mindepth 1 -maxdepth 1 -type d | wc -l) 개 디렉토리"
echo "    전체    : $(du -sh data | cut -f1)"
echo
echo "완료. zip 을 지우려면: rm -rf $ZIP_DIR"
echo "테스트: python day2/day2_4.0.sam3_inference.py"
