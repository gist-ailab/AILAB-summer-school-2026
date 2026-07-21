#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ISAACLAB="${ISAACLAB_PATH:-}"
if [[ -z "$ISAACLAB" ]]; then
    if [[ -x /workspace/IsaacLab/isaaclab.sh ]]; then
        ISAACLAB=/workspace/IsaacLab
    else
        ISAACLAB="$HOME/IsaacLab"
    fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
run_step="${1:-all}"

run_1_1() { TERM=xterm "$ISAACLAB/isaaclab.sh" -p day3/day3_3.1.1_pusht_state_rerender_answer.py; }
run_1_2() { TERM=xterm "$ISAACLAB/isaaclab.sh" -p day3/day3_3.1.2_pusht_visual_dr_replay_answer.py; }
run_1() { run_1_1; run_1_2; }
run_2_1() { TERM=xterm "$ISAACLAB/isaaclab.sh" -p day3/day3_3.2.1_action_replay_answer.py; }
run_2_2() { TERM=xterm "$ISAACLAB/isaaclab.sh" -p day3/day3_3.2.2_replay_mimic_ready_data_answer.py; }
run_2() { run_2_1; run_2_2; }
run_3() { TERM=xterm "$ISAACLAB/isaaclab.sh" -p day3/day3_3.3_object_centric_transform_answer.py; }
run_4() { TERM=xterm "$ISAACLAB/isaaclab.sh" -p day3/day3_3.4_mimic_datagenerator_rollout_answer.py; }
run_5() { "$PYTHON_BIN" day3/day3_3.5_2subtask_generation_answer.py; }
run_6() { "$PYTHON_BIN" day3/day3_3.6_multisubtask_generation_answer.py; }

case "$run_step" in
    1) run_1 ;;
    1.1) run_1_1 ;;
    1.2) run_1_2 ;;
    2) run_2 ;;
    2.1) run_2_1 ;;
    2.2) run_2_2 ;;
    3) run_3 ;;
    4) run_4 ;;
    5) run_5 ;;
    6) run_6 ;;
    all) run_1; run_2; run_3; run_4; run_5; run_6 ;;
    *) echo "Usage: $0 [1|1.1|1.2|2|2.1|2.2|3|4|5|6|all]" >&2; exit 2 ;;
esac
