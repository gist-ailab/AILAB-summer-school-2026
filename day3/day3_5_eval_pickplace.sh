#!/bin/bash
# =====================================================================
# day3_5_eval_pickplace.sh — PickPlace Diffusion Policy 자동 평가 예시
#
# 한 run 디렉토리 안의 여러 epoch 체크포인트(또는 단일 .pth)를 **동일 seed**로
# 자동 평가하고, epoch별 성공률을 표로 요약합니다.
#   - 평가 스크립트: day3_5_eval_answer.py (이미지 인코딩/정규화 상수 자동 결정)
#   - 성공 판정: object_pickplace_goal (bin 실시간 pose 사용)
#   - 공정 비교를 위해 --eval_seed 를 고정합니다.
#
# 사용법:
#   bash day3_5_eval_pickplace.sh <run_dir | checkpoint.pth> [num_rollouts]
#
#   # 예1) run 디렉토리의 모든 epoch 평가 (n=20)
#   bash day3_5_eval_pickplace.sh \
#     robomimic/diffusion_policy_trained_models/pickplace/tbar_dp_teleop10_aug40/20260721032044
#
#   # 예2) 단일 체크포인트를 n=50 으로 평가
#   bash run_eval_pickplace.sh <...>/models/model_epoch_450.pth 50
#
# 환경 변수(선택):
#   EVAL_SEED      초기 배치 고정 시드 (기본: 1000)
#   MAX_STEPS      에피소드당 최대 스텝 (기본: 300)
#   DDIM_STEPS     지정 시 DDIM 샘플러로 전환 (예: 16). 미지정 시 학습된 DDPM 100 유지
# =====================================================================
set -e
cd "$(dirname "$0")"

ISAAC="../../IsaacLab/isaaclab.sh"
EVAL_SCRIPT="day3_5_eval_answer.py"

TARGET="${1:?사용법: bash day3_5_eval_pickplace.sh <run_dir|ckpt.pth> [num_rollouts]}"
N="${2:-20}"
SEED="${EVAL_SEED:-1000}"
STEPS="${MAX_STEPS:-300}"

# DDIM 옵션(선택)
EXTRA=()
[[ -n "$DDIM_STEPS" ]] && EXTRA+=(--ddim_steps "$DDIM_STEPS")

# ---- 평가 대상 체크포인트 수집 ----
# 디렉토리면 models/ 아래 정상 크기(>1MB)의 model_epoch_*.pth 를 epoch 순으로.
# (학습 중 잘린 0바이트/수십KB 손상 파일은 크기 필터로 자동 제외)
if [[ -d "$TARGET" ]]; then
    MODELS_DIR="$TARGET/models"; [[ -d "$MODELS_DIR" ]] || MODELS_DIR="$TARGET"
    mapfile -t CKPTS < <(find "$MODELS_DIR" -name "model_epoch_*.pth" -size +1M | sort -V)
    RUN_DIR="$TARGET"
else
    CKPTS=("$TARGET")
    RUN_DIR="$(dirname "$(dirname "$TARGET")")"
fi
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "체크포인트를 찾지 못했습니다: $TARGET"; exit 1; }

echo "=================================================================="
echo " PickPlace 자동 평가 | 대상 ${#CKPTS[@]}개 | n=$N seed=$SEED max_steps=$STEPS ${DDIM_STEPS:+| DDIM=$DDIM_STEPS}"
echo "=================================================================="

for CKPT in "${CKPTS[@]}"; do
    echo ""
    echo ">> $(basename "$CKPT")"
    "$ISAAC" -p "$EVAL_SCRIPT" \
        --task_type pickplace \
        --checkpoint "$CKPT" \
        --num_rollouts "$N" \
        --max_steps "$STEPS" \
        --eval_seed "$SEED" 
        "${EXTRA[@]}" \
        2>&1 | tr '\r' '\n' | grep -E "obs_encoding|Obs camera|경고|Success Rate|Results log"
done

# ---- epoch별 성공률 요약 (eval_results.json 파싱) ----
echo ""
echo "=================================================================="
echo " 요약 (성공률 순)"
echo "=================================================================="
python3 - "$RUN_DIR" <<'PYEOF'
import sys, os, json, glob
run_dir = sys.argv[1]
rows = []
for jf in glob.glob(os.path.join(run_dir, "eval_pickplace_*", "eval_results.json")):
    try:
        d = json.load(open(jf))
        rows.append((os.path.basename(os.path.dirname(jf)),
                     d["success_count"], d["num_rollouts"], float(d["success_rate"])))
    except Exception:
        pass
if not rows:
    print("  eval_results.json 을 찾지 못했습니다."); raise SystemExit
for name, sc, n, sr in sorted(rows, key=lambda r: -r[3]):
    print(f"  {name:52s} {sc:2d}/{n:2d}  ({sr:5.1f}%)")
best = max(rows, key=lambda r: r[3])
print(f"\n  ▶ 최고: {best[0]}  {best[3]:.1f}%")
PYEOF
