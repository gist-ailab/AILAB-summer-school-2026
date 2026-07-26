#!/usr/bin/env bash
# Diffusion Policy 학습 — 학습 전 config↔데이터셋 검사 후 train.py 실행.
#
# 사용법:
#   ./train.sh <config.json> <dataset.hdf5> [train.py 추가인자...]
# 예:
#   ./train.sh configs/day3_4_pusht_teleop_dp_config_practice.json \
#              datasets/tbar_pusht_teleop_practice.hdf5
#
# 검사에서 ERROR 발생 시 학습을 시작하지 않고 중단합니다.
# 검사만 실행하려면:  python day3_4_check_train_config.py --config <cfg> --dataset <ds>
#
# 사용할 python 은 PYTHON 환경변수로 지정합니다(미지정 시 `python`).
# robomimic 이 설치된 환경을 지정해야 합니다. 예:
#   PYTHON=$HOME/anaconda3/envs/env_isaaclab/bin/python ./train.sh <config> <dataset>

set -euo pipefail

# 위치 인자(<config> <dataset>) 와 named 플래그(--config/--dataset) 를 모두 받는다.
CONFIG=""; DATASET=""; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config)  CONFIG="$2";  shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        *)
            if   [ -z "$CONFIG" ];  then CONFIG="$1";  shift
            elif [ -z "$DATASET" ]; then DATASET="$1"; shift
            else EXTRA+=("$1"); shift; fi;;
    esac
done

if [ -z "$CONFIG" ] || [ -z "$DATASET" ]; then
    echo "사용법: ./train.sh <config.json> <dataset.hdf5> [train.py 추가인자...]"
    echo "   또는: ./train.sh --config <config.json> --dataset <dataset.hdf5>"
    exit 2
fi
set -- ${EXTRA[@]+"${EXTRA[@]}"}

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/2] 학습 전 검사 ==="
if ! "$PY" "$HERE/day3_4_check_train_config.py" --config "$CONFIG" --dataset "$DATASET"; then
    echo ""
    echo "✗ 검사 실패 — config/데이터셋을 수정한 뒤 다시 실행하십시오. 학습을 시작하지 않습니다."
    exit 1
fi

echo ""
echo "=== [2/2] 학습 시작 ==="
exec "$PY" "$HERE/robomimic/robomimic/scripts/train.py" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    "$@"
