# Day 3 - Pick & Place와 PushT 데이터 수집, 증강 및 학습/평가 실습

이 디렉토리는 2026 AILAB 여름학교 Day 3 과정인 **Isaac Lab 환경 구축 및 시뮬레이션 기반 데이터 수집, 증강 및 학습/평가** 실습을 위한 코드를 담고 있습니다.

> **모든 명령어는 프로젝트 저장소 루트에서 실행합니다.**

---

## 📂 디렉토리 구조

```
day3/
├── README.md
│
├── [1교시: Pick & Place 데이터 수집 (State Machine & Teleop)]
│   ├── task/lift/
│   │   ├── custom_pickplace_env_cfg_3_1.1_tbar_practice.py             # 문제 1: T-bar 오브젝트 배치
│   │   ├── custom_pickplace_env_cfg_3_1.2_camera_observation_practice.py # 문제 2: 카메라 관측 설정
│   │   └── custom_pickplace_env_cfg_3_1_answer.py                      # 최종 완성본 (참고용)
│   │
│   ├── task/lift/config/
│   │   ├── ik_abs_env_cfg_3_1.3_action_controller_practice.py          # 문제 3: IK 액션 컨트롤러 교체
│   │   ├── ik_abs_env_cfg_3_1.py                                       # 문제 3 정답
│   │   ├── joint_pos_env_cfg_3_1.4_control_gripper_practice.py         # 문제 4: 이진 그리퍼 제어
│   │   └── joint_pos_env_cfg_3_1.py                                    # 문제 4 정답
│   │
│   ├── day3_1_pickplace_statemachine_collect_data_practice.py          # 문제 5: 성공 에피소드만 저장
│   ├── day3_1_pickplace_statemachine_collect_data_answer.py            # State Machine 최종 완성본
│   │
│   ├── day3_2_pickplace_teleop_collect_data_practice.py                # 문제 6: 델타 적분 및 액션 조립
│   └── day3_1_pickplace_teleop_collect_data_answer.py                  # Teleop 최종 완성본
│
├── [2교시: PushT 데이터 수집 (Teleop)]
│   ├── task/lift/
│   │   ├── custom_pusht_env_cfg_3_2.7_reset_practice.py                # 문제 7: 커스텀 리셋 (도메인 랜덤화)
│   │   └── custom_pusht_env_cfg_3_2_answer.py                          # 최종 완성본 (참고용)
│   │
│   ├── day3_2_pusht_teleop_collect_data_practice.py                    # 문제 8: 성공 판정 구현
│   └── day3_2_pusht_teleop_collect_data_answer.py                      # 최종 완성본 (참고용)
│
├── [3교시: 데이터 증강 (Visual DR & IsaacLab Mimic)]
│   ├── day3_3.1.1_pusht_state_rerender_practice.py              # 문제 1.1: PushT state re-render
│   ├── day3_3.1.2_pusht_visual_dr_replay_practice.py            # 문제 1.2: PushT visual domain randomization
│   ├── day3_3.2.1_action_replay_practice.py                     # 문제 2.1: PickPlace action replay
│   ├── day3_3.2.2_replay_mimic_ready_data_practice.py           # 문제 2.2: mimic-ready datagen_info 기록
│   ├── day3_3.3_object_centric_transform_practice.py            # 문제 3: object-centric trajectory transform
│   ├── day3_3.4_mimic_datagenerator_rollout_practice.py         # 문제 4: isaaclab_mimic DataGenerator rollout
│   ├── day3_3.5_2subtask_generation_practice.py                 # 문제 5: 2-subtask source + generation
│   ├── day3_3.6_multisubtask_generation_practice.py             # 문제 6: multi-subtask source + generation
│   ├── day3_3_utils.py                                          # 3교시 공통 유틸
│   ├── run_day3_3_answer_defaults.sh                            # 3교시 answer 실행 스크립트
│   └── isaaclab_mimic_reference/                                # isaaclab_mimic 내부 구현 참고용
│
├── [4교시: Diffusion Policy 모델 학습]
│   ├── configs/
│   │   ├── day3_4_pusht_teleop_dp_config_practice.json          # PushT 학습 config (실습용)
│   │   ├── pickplace_dp_config_resized.json                     # Pick&Place 학습 config
│   │   └── pusht_teleop_dp_config_resized.json                  # PushT Teleop 학습 config
│   │
│   ├── day3_4.99_preprocess_hdf5.py                          # HDF5 전처리 (resize / float→uint8)
│   └── robomimic/                       # git submodule (학습 프레임워크)
│       └── robomimic/scripts/train.py
│
├── [5교시: Diffusion Policy 모델 평가]
│   ├── day3_5.1_eval_practice.py        # 문제 1: 체크포인트 로드 + 환경 생성
│   ├── day3_5.1_eval_answer.py          # 문제 1 정답
│   ├── day3_5.2_eval_practice.py        # 문제 2: obs 변환 + 전체 Rollout
│   ├── day3_5.2_eval_answer.py          # 문제 2 정답
│   ├── day3_5_eval_answer.py            # Full eval (참고용)
│   └── day3_5_eval_generalization.py    # 일반화 성능 평가 (Visual DR / spawn 범위)

├── datasets/
│   ├── tbar_pusht_teleop_practice.hdf5      # 3교시 Visual DR 입력
│   ├── tbar_pickpalce_teleop_practice.hdf5 # 3교시 Mimic 입력
│   └── tbar_pickplace_statemachine_practice.hdf5 # 1교시 state-machine 수집 결과
│
└── data/
    └── assets/
        ├── basket/basket.usd   # 바구니(Bin) USD 에셋
        └── t_bar/T_bar.usd     # T-bar USD 에셋
```

---

## 🚀 Answer 코드 실행 가이드

아래 명령어는 모두 프로젝트 저장소 루트에서 실행합니다.

---

### 1교시 — Pick & Place 데이터 수집 (State Machine & Teleop)

1교시에서는 T-bar를 바구니에 담는 Pick & Place 작업을 자동(State Machine)과 수동(Teleop)으로 수집합니다.

#### 1. 자동 수집 (State Machine)

State Machine 기반으로 로봇이 사전 정의된 상태(REST → PREGRASP → GRASP → LIFT 등)를 전이하며 **자동으로 물체를 잡고 바구니에 담는 시연** 데이터를 수집합니다.

#### 🔹 최종 완성본 실행

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_1_pickplace_statemachine_collect_data_answer.py \
    --num_envs 4 \
    --num_demos 50 \
    --dataset_file day3/datasets/tbar_pickplace_statemachine_practice.hdf5
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num_envs` | `4` | 병렬 환경 개수 |
| `--num_demos` | `50` | 수집할 성공 데모 수 (0 = 무한) |
| `--max_steps` | `2000` | 환경별 타임아웃 스텝 수 |
| `--dataset_file` | `day3/datasets/tbar_pickplace_statemachine_practice.hdf5` | 저장 경로 |

---

#### 2. 수동 수집 (Teleop)

키보드로 로봇을 직접 조종하며 Pick & Place 데모를 수집합니다.

#### 🔹 최종 완성본 실행

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_1_pickplace_teleop_collect_data_answer.py \
    --num_demos 50 \
    --dataset_file day3/datasets/tbar_pickpalce_teleop_practice.hdf5 \
    --enable_cameras
```

**키보드 조작법**

| 키 | 동작 |
|----|------|
| `W` / `S` | X축 이동 (앞 / 뒤) |
| `A` / `D` | Y축 이동 (좌 / 우) |
| `Q` / `E` | Z축 이동 (위 / 아래) |
| `Z` / `X` | Yaw 회전 |
| `K` | 그리퍼 토글 (열기/닫기) |
| `R` | 현재 에피소드 버리고 리셋 |

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num_demos` | `50` | 수집할 성공 데모 수 (0 = 무한) |
| `--max_steps` | `2000` | 에피소드 타임아웃 스텝 수 |
| `--linear_speed` | `0.4` | 이동 속도 (m/s) |
| `--align_steps` | `45` | 초기 자세 정렬 스텝 수 (이 구간은 데이터 미수집) |
| `--dataset_file` | `day3/datasets/tbar_pickpalce_teleop_practice.hdf5` | 저장 경로 |

---

### 2교시 — PushT 데이터 수집 (Teleop)

2교시에서는 T-bar를 밀어 목표 위치(x=0.4, y=0.4)에 정렬하는 PushT 작업을 수집합니다.

#### 🔹 최종 완성본 실행

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_2_pusht_teleop_collect_data_answer.py \
    --task Template-PushT-Franka-v0 \
    --teleop_device keyboard \
    --enable_cameras \
    --dataset_file day3/datasets/tbar_pusht_teleop_practice.hdf5 \
    --num_demos 50
```

**키보드 조작법**

| 키 | 동작 |
|----|------|
| `W` / `S` | X축 이동 |
| `A` / `D` | Y축 이동 |
| `Z` / `X` | Yaw 회전 |
| `R` | 현재 에피소드 버리고 리셋 |

> **성공 조건**: T-bar 위치 오차 < **1cm** AND Yaw 오차 < **0.1 rad** 상태가 15스텝 연속 유지

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num_demos` | `0` | 수집할 성공 데모 수 (0 = 무한) |
| `--num_success_steps` | `15` | 성공 판정에 필요한 연속 성공 스텝 수 (30Hz 기준 0.5초 = 15스텝) |
| `--step_hz` | `30` | 제어 주파수 (Hz) |
| `--dataset_file` | `day3/datasets/tbar_pusht_teleop_practice.hdf5` | 저장 경로 |

---

### 3교시 — 데이터 증강 (Visual DR & IsaacLab Mimic)

3교시에서는 HDF5 내부 숫자를 직접 바꾸지 않습니다. 저장 state를 IsaacLab에서 다시 렌더링하거나, `isaaclab_mimic` DataGenerator rollout으로 새 궤적을 생성합니다.

모든 3교시 명령은 **프로젝트 루트**에서 실행합니다.

```bash
cd /workspace/AILAB-summer-school-2026
conda activate isaaclab
export ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}
```

#### 입력 데이터

| 용도 | 입력 파일 | 사용 문제 |
|---|---|---|
| Visual re-render / DR | `day3/datasets/tbar_pusht_teleop_practice.hdf5` | 1.1, 1.2 |
| Mimic trajectory augmentation | `day3/datasets/tbar_pickpalce_teleop_practice.hdf5` | 2.1, 2.2 |
| State-machine 수집 결과 | `day3/datasets/tbar_pickplace_statemachine_practice.hdf5` | 1교시 수집 결과 |

#### 문제 흐름

| 문제 | 구현하는 핵심 | 실행 결과 |
|---|---|---|
| 1.1 | HDF5 state 복원 후 camera re-render | PushT re-render HDF5 |
| 1.2 | episode별 object/table/ground/light style 샘플링 | PushT visual DR HDF5 |
| 2.1 | 첫 state reset 후 저장 action replay | PickPlace replay HDF5 |
| 2.2 | state replay 중 `eef/object/target/gripper/signal` 기록 | mimic-ready HDF5와 signal source |
| 3 | `T_new_eef = T_new_object @ inv(T_source_object) @ T_source_eef` | wide 범위의 T-bar로 접근하는 화면 |
| 4 | Mimic success/recorder/action queue rollout 연결 | generated HDF5 |
| 5 | 물체 높이로 2-subtask boundary 생성 | 2-subtask source와 generated HDF5 |
| 6 | 접근/닫힘/lift/bin 근처 boundary 생성 | multi-subtask source와 generated HDF5 |

#### 정답 실행

```bash
# 1.1, 1.2, 2.1, 2.2, 3, 4, 5, 6 중 하나를 지정
./day3/run_day3_3_answer_defaults.sh 1.1
```

개별 실행도 가능합니다. 5·6은 내부적으로 4번 DataGenerator rollout을 호출합니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_3.1.2_pusht_visual_dr_replay_answer.py
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_3.5_2subtask_generation_answer.py --generation_num_trials 3
```

#### 기본 결과 경로

- PushT 재렌더링/visual DR: `day3/datasets/pusht/`
- PickPlace replay, mimic-ready, source, generated data: `day3/datasets/pickplace/`
- 3번은 HDF5를 추가 저장하지 않고 IsaacLab 화면에서 object-centric 접근을 확인합니다.

#### Spawn Randomization

4~6번의 `--spawn_randomization original`은 T-bar를 x/y ±0.1 m, yaw ±45도 범위에서 바꾸고 bin은 고정합니다. `wide`는 T-bar를 x/y ±0.18 m, yaw -45~+135도, bin을 x/y ±0.08 m와 pitch ±30도 범위에서 바꿉니다.

#### Subtask 시각화

4~6번에서 `--visualize_subtasks`를 사용하면 T-bar 머리 옆 marker가 subtask에 따라 바뀝니다. 빨강(초기) → 파랑(접근) → 마젠타(닫힘) → 시안(lift) → 초록(bin 근처) 순서입니다. marker가 camera image에 포함될 수 있으므로, 이 옵션으로 생성한 HDF5는 **검증용**으로만 사용하고 학습 데이터 생성 시에는 옵션을 생략합니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_3.6_multisubtask_generation_answer.py \
  --generation_num_trials 1 \
  --visualize_subtasks
```

### 4교시 — Diffusion Policy 학습

4교시에서는 1~3교시에서 수집·증강한 HDF5 데이터셋으로 Diffusion Policy를 학습합니다. (실습 Task: Push-T)

#### 1. 데이터셋 구조 확인

학습 전 obs key와 이미지 크기를 반드시 확인합니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/robomimic/robomimic/scripts/get_dataset_info.py --dataset <데이터셋.hdf5>
```

#### 2. 데이터셋 전처리

Isaac Lab RecorderManager는 카메라 이미지를 raw float32로 저장하지만, robomimic은 uint8 `[0, 255]`를 사용합니다. 
학습 전에 데이터 전처리를 수행합니다.

```bash
# 구조 확인 (dry-run)
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_4.99_preprocess_hdf5.py --input day3/datasets/tbar_pusht_teleop_practice.hdf5 --dry-run

# resize 또는 float32→uint8 변환
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_4.99_preprocess_hdf5.py \
    --input day3/datasets/tbar_pusht_teleop_practice.hdf5 \
    --output day3/datasets/tbar_pusht_teleop_practice_dtype.hdf5 \
    --no_resize
# resize 하지 않고 float32→uint8만 변환

```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` | (필수) | 입력 HDF5 경로 |
| `--output` | - | 출력 HDF5 경로 |
| `--height` | `240` | 타겟 높이 |
| `--width` | `320` | 타겟 너비 |
| `--no_resize` | OFF | resize 없이 float→uint8만 수행 |
| `--dry-run` | OFF | 구조만 확인하고 종료 |

#### 3. Config 작성

`configs/` 디렉토리에 학습용 JSON config가 준비되어 있습니다. 데이터셋의 obs key와 이미지 크기에 맞게 수정합니다.

| 파일 | 용도 |
|---|---|
| `day3_4_pusht_teleop_dp_config_practice.json` | PushT 학습 config 작성 실습 |
| `pickplace_dp_config_resized.json` | Pick&Place 학습 config |
| `pusht_teleop_dp_config_resized.json` | PushT Teleop 학습 config |

> **데이터셋에 맞게 수정해야 할 항목:**
> - `observation.modalities.obs.low_dim` → HDF5의 obs key와 일치
> - `observation.modalities.obs.rgb` → HDF5의 카메라 key와 일치
> - `CropRandomizer` crop 크기 : 이미지 크기의 **약 90%**
> - `batch_size` → GPU VRAM에 따라 조절

#### 4. 학습 실행

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day3/robomimic/robomimic/scripts/train.py \
    --config day3/configs/day3_4_pusht_teleop_dp_config_practice.json \
    --dataset day3/datasets/tbar_pusht_teleop_practice_dtype.hdf5
```

---

### 5교시 — Diffusion Policy 평가

5교시에서는 학습된 Diffusion Policy를 Isaac Lab 환경에서 rollout하여 성공률을 측정합니다. 5.1에서 환경을 정상 생성하는지 확인한 뒤, 5.2에서 obs 변환과 전체 rollout을 수행합니다.

#### 문제 흐름

| 문제 | 파일 | TODO | 구현하는 핵심 |
|---|---|---|---|
| 5.1 | `day3_5.1_eval_practice.py` | 2개 | `importlib`으로 env_cfg 로드, `concatenate_terms=False` 설정 |
| 5.2 | `day3_5.2_eval_practice.py` | 2개 | Low-dim `(dim,)→(1,T,dim)`, Image `(H,W,C)→(1,T,H,W,C)` 변환 |

#### 성공 판정

성공 판정은 1·2교시 데이터 수집과 동일한 함수를 사용합니다.

| task_type | 판정 함수 | 기준 |
|---|---|---|
| `pickplace` | `mdp_3_1.terminations.object_pickplace_goal` | XY < 15cm, Z < 10cm |
| `pusht` | `mdp_3_2.terminations_answer.object_pusht_goal` | XY < 1cm, Yaw < 0.1 rad |

#### 🔹 정답 실행

```bash
# 5.1: 환경 생성 실습
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_5.1_eval_practice.py \
    --task_type pusht \
    --checkpoint <체크포인트.pth>

# 5.2: rollout 실습
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_5.2_eval_practice.py \
    --task_type pusht \
    --checkpoint <체크포인트.pth> \
    --num_rollouts 10 --max_steps 300

# Full eval (참고용)
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_5_eval_answer.py \
    --task_type pusht \
    --checkpoint <체크포인트.pth> \
    --num_rollouts 10 --max_steps 300
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--task_type` | `pusht` | 평가할 태스크: `pickplace` 또는 `pusht` |
| `--checkpoint` | (필수) | 학습된 체크포인트 `.pth` 경로 |
| `--num_rollouts` | `10` | 평가 에피소드 수 |
| `--max_steps` | `300` | 에피소드당 최대 스텝 수 |

---

### 5교시 (심화) — 일반화 성능 평가

텔레옵 데이터로만 학습한 모델과 증강 데이터로 학습한 모델의 **일반화 성능**을 비교합니다.

| 태스크 | 증강 전략 | 일반화 테스트 |
|---|---|---|
| PushT | Visual DR (3교시 1.2) | 색상/조명을 랜덤화한 환경에서 평가 |
| PickPlace | Trajectory 증강 (3교시 4~6) | 학습 범위보다 넓은 초기 위치에서 평가 |

#### 🔹 실행

```bash
# PushT: Visual DR 일반화 테스트
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_5_eval_generalization.py \
    --task_type pusht --visual_dr \
    --checkpoint <체크포인트.pth> --num_rollouts 20

# PickPlace: 넓은 spawn 범위 일반화 테스트 
"$ISAACLAB_PATH/isaaclab.sh" -p day3/day3_5_eval_generalization.py \
    --task_type pickplace --spawn_range wide \ 
    --checkpoint <체크포인트.pth> --num_rollouts 20 --max_steps 600
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--spawn_range` | `original` | 초기 위치 범위: `original`, `wide`, `extreme` |
| `--visual_dr` | OFF | Visual DR 활성화 |

> `original`은 데이터 수집과 동일한 범위, `wide`는 mimic 증강에서 사용한 확장 범위, `extreme`은 학습 범위 밖(OOD) 테스트입니다.
