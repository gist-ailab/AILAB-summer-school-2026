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

# ------------------------------------------------------------------
# day3 diffusion policy 학습 완료 체크포인트 -> day3/robomimic/diffusion_policy_trained_models/
#   (day3_5_eval_pickplace.sh 가 robomimic/diffusion_policy_trained_models/... 를 참조)
#
# 80명이 동시에 gdown 하면 Google Drive 의 파일당 24h 다운로드 쿼터에 걸려
# "Too many users have downloaded this file recently" 로 실패한다. 이를 완화하기 위해:
#   1) 미러 우선   : CKPT_MIRROR_URL 이 있으면 쿼터 없는 미러에서 curl 로 받는다(권장·최선).
#   2) 단일 zip    : CKPT_ZIP_ID(단일 diffusion_policy_trained_models.zip) 가 폴더보다 훨씬 안정적.
#   3) 지연·재시도 : 동시 접속을 무작위 지연으로 분산 + 지수 백오프 재시도 + 이어받기.
# 위 1),2) 를 비워두면 폴더 링크를 gdown --folder 로 받는다(동시성에 가장 취약, 폴백용).
#
# ▶ 강사 준비(권장): diffusion_policy_trained_models/ 를 zip 하나로 묶어
#   (a) Google Drive 에 올리고 그 파일 ID 를 CKPT_ZIP_ID 에,
#   (b) 쿼터 없는 미러(GitHub Release 에셋 / Hugging Face / 웹서버)에 올리고 직링크를 CKPT_MIRROR_URL 에.
#   zip 최상위에 diffusion_policy_trained_models/ 디렉토리가 있어야 한다(그래야 day3/robomimic/ 로 바로 풀림).
CKPT_MIRROR_URL="${CKPT_MIRROR_URL:-}"                                  # 예: https://github.com/gist-ailab/ailab-summer-school-2026/releases/download/v1/diffusion_policy_trained_models.zip
CKPT_ZIP_ID="${CKPT_ZIP_ID:-}"                                         # 예: 단일 zip 의 Google Drive 파일 ID
CKPT_FOLDER_ID="${CKPT_FOLDER_ID:-18qNEoygnG4YsnhFHJATnpX2jXhMzzW6Z}" # (폴백) 폴더 링크 ID
CKPT_ZIP_NAME="diffusion_policy_trained_models.zip"
DP_CKPT_DIR="day3/robomimic"                        # 여기로 압축 해제 -> day3/robomimic/diffusion_policy_trained_models/
DP_CKPT_MARKER="$DP_CKPT_DIR/diffusion_policy_trained_models"          # 존재하면 이미 설치된 것으로 보고 건너뜀

# 대량 동시 다운로드 대비 파라미터 (환경변수로 조정 가능)
DL_MAX_RETRY="${DL_MAX_RETRY:-6}"       # 최대 재시도 횟수
DL_STAGGER_MAX="${DL_STAGGER_MAX:-90}"  # 첫 시도 전 0..N 초 무작위 대기(동시성 분산). 0 이면 대기 없음

# 동시에 몰리지 않도록 무작위 지연 (Google Drive 다운로드 직전 1회)
_stagger() {
    [ "${DL_STAGGER_MAX:-0}" -gt 0 ] || return 0
    local s=$(( RANDOM % (DL_STAGGER_MAX + 1) ))
    echo "    (동시 접속 분산) ${s}s 대기 후 시작..."
    sleep "$s"
}

# download_single <file_id> <out_path> [mirror_url]
#   mirror_url 이 있으면 우선 curl(쿼터 없음), 실패 시 Google Drive gdown 으로 폴백. 이어받기 지원.
download_single() {
    local id="$1" out="$2" mirror="${3:-}" attempt delay
    if [ -f "$out" ]; then echo "    재사용: $(basename "$out")"; return 0; fi
    mkdir -p "$(dirname "$out")"

    if [ -n "$mirror" ]; then
        for attempt in $(seq 1 "$DL_MAX_RETRY"); do
            echo "    미러 다운로드 (시도 $attempt/$DL_MAX_RETRY): $mirror"
            if curl -fL --retry 3 --retry-delay 2 -C - -o "$out" "$mirror"; then
                echo "    완료(미러): $(basename "$out")"; return 0
            fi
            delay=$(( 2 ** attempt + RANDOM % 5 ))
            echo "    미러 실패. ${delay}s 후 재시도..."; sleep "$delay"
        done
        echo "    미러가 계속 실패 → Google Drive 로 폴백"
    fi

    _stagger
    for attempt in $(seq 1 "$DL_MAX_RETRY"); do
        echo "    gdown (시도 $attempt/$DL_MAX_RETRY): $id"
        if gdown --continue "$id" -O "$out"; then
            echo "    완료(gdown): $(basename "$out")"; return 0
        fi
        delay=$(( 2 ** attempt + RANDOM % 10 ))
        echo "    실패(쿼터/네트워크 가능). ${delay}s 후 재시도..."; sleep "$delay"
    done
    echo "ERROR: 다운로드 실패: $out"
    echo "  Google Drive 쿼터일 수 있다. 잠시 후 다시 실행하거나 CKPT_MIRROR_URL(미러)을 설정하라."
    return 1
}

# $DP_CKPT_MARKER 아래에 하위 디렉토리가 있으면(=압축 해제 완료) 0 반환
_ckpt_extracted() {
    [ -d "$DP_CKPT_MARKER" ] && \
        [ -n "$(find "$DP_CKPT_MARKER" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]
}

# 폴더 안에 받아진 zip(pickplace.zip, pusht.zip 등)들을 제자리에 해제.
# zip 이 이미 자기 이름 폴더로 감싸져 있으면 그대로, 아니면 zip 이름 폴더로 감싼다.
_unzip_inner_zips() {
    shopt -s nullglob
    local z base top
    for z in "$DP_CKPT_MARKER"/*.zip; do
        base="$(basename "${z%.zip}")"
        top="$(unzip -Z1 "$z" 2>/dev/null | head -1 | cut -d/ -f1 || true)"
        if [ "$top" = "$base" ]; then
            echo "    압축 해제: $(basename "$z") -> $DP_CKPT_MARKER/"
            unzip -qo "$z" -d "$DP_CKPT_MARKER"
        else
            echo "    압축 해제: $(basename "$z") -> $DP_CKPT_MARKER/$base/"
            unzip -qo "$z" -d "$DP_CKPT_MARKER/$base"
        fi
        rm -f "$z"
    done
    shopt -u nullglob
}

# 대용량 hdf5 를 이어받기(--continue)+재시도+무결성(h5py open) 검증으로 안전하게 받는다.
# 잘린/손상 파일은 재사용하지 않고 다시 받으며, 끝내 열리지 않으면 실패한다(이미지에 잘린 파일이 구워지는 것 방지).
download_hdf5() {
    local id="$1" out="$2" attempt
    for attempt in $(seq 1 "$DL_MAX_RETRY"); do
        if [ -f "$out" ] && python -c "import h5py,sys; h5py.File(sys.argv[1],'r').close()" "$out" >/dev/null 2>&1; then
            echo "    무결성 통과, 재사용: $(basename "$out")"; return 0
        fi
        echo "    다운로드/이어받기 (시도 $attempt/$DL_MAX_RETRY): $(basename "$out")"
        gdown --continue "$id" -O "$out" || true
    done
    if [ -f "$out" ] && python -c "import h5py,sys; h5py.File(sys.argv[1],'r').close()" "$out" >/dev/null 2>&1; then
        echo "    무결성 통과: $(basename "$out")"; return 0
    fi
    echo "ERROR: 데이터셋이 잘렸거나 손상됨(무결성 실패): $out"
    echo "  Google Drive 쿼터/네트워크로 다운로드가 끊겼을 수 있다. 잠시 후 다시 실행하라."
    return 1
}

# day3 학습 완료 체크포인트를 day3/robomimic/ 로 설치
download_dp_ckpt() {
    if _ckpt_extracted; then
        echo "    재사용: $DP_CKPT_MARKER"
        return 0
    fi
    mkdir -p "$DP_CKPT_DIR"
    if [ -n "$CKPT_MIRROR_URL" ] || [ -n "$CKPT_ZIP_ID" ]; then
        # 단일 zip 방식 (권장): zip 하나에 전체 트리(diffusion_policy_trained_models/...) 포함
        local zip="$ZIP_DIR/$CKPT_ZIP_NAME"
        download_single "$CKPT_ZIP_ID" "$zip" "$CKPT_MIRROR_URL" || return 1
        echo "    압축 해제: $CKPT_ZIP_NAME -> $DP_CKPT_DIR/"
        unzip -qo "$zip" -d "$DP_CKPT_DIR"
    else
        # 폴더 방식 (폴백): 폴더 안의 zip 들을 받아 각각 제자리 해제.
        echo "    폴더 다운로드(gdown --folder) — 동시성에 취약. 가능하면 CKPT_ZIP_ID/CKPT_MIRROR_URL 사용을 권장."
        mkdir -p "$DP_CKPT_MARKER"
        _stagger
        local attempt delay ok=0
        for attempt in $(seq 1 "$DL_MAX_RETRY"); do
            echo "    gdown --folder (시도 $attempt/$DL_MAX_RETRY)"
            # gdown --folder 는 이미 받은 파일은 건너뛰므로 재시도/재실행 시 이어받기 효과
            if gdown --folder "https://drive.google.com/drive/folders/$CKPT_FOLDER_ID" -O "$DP_CKPT_MARKER"; then
                ok=1; break
            fi
            delay=$(( 2 ** attempt + RANDOM % 10 ))
            echo "    폴더 다운로드 실패. ${delay}s 후 재시도..."; sleep "$delay"
        done
        if [ "$ok" -ne 1 ]; then
            echo "ERROR: 체크포인트 폴더 다운로드 실패"
            echo "  Google Drive 쿼터일 수 있다. 잠시 후 다시 실행하거나 CKPT_MIRROR_URL(미러)을 설정하라."
            return 1
        fi
        _unzip_inner_zips
    fi
    _ckpt_extracted || { echo "ERROR: 체크포인트 압축 해제 결과가 비어있다: $DP_CKPT_MARKER"; return 1; }
}

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

echo "==> day3 데모 데이터셋 다운로드 (대용량 hdf5)"
mkdir -p day3/datasets
download_hdf5 "$DAY3_DATASET_ID" "day3/datasets/$DAY3_DATASET_NAME" || exit 1

echo "==> day3 학습 완료 체크포인트 다운로드 -> $DP_CKPT_DIR/"
download_dp_ckpt

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
_ckpt_extracted || { echo "    누락(또는 미해제): $DP_CKPT_MARKER"; missing=1; }

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
