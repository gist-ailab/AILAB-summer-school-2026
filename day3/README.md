# Day 3 - Pick & Place와 PushT 데이터 수집, 증강 및 학습/평가 실습

이 디렉토리는 2026 AILAB 여름학교 Day 3 과정인 **Isaac Lab 환경 구축 및 시뮬레이션 기반 데이터 수집, 증강 및 학습/평가** 실습을 위한 코드를 담고 있습니다.

> **1·2교시 명령어는 `day3/`에서, 3교시 이후 명령어는 프로젝트 루트에서 실행합니다.**

---

## 📂 디렉토리 구조

```
day3/
├── README.md
│
├── [1교시: Pick & Place 데이터 수집 (State Machine & Teleop)]
│   ├── task/lift/
│   │   ├── custom_pickplace_env_cfg_3_1_practice_1.py                  # 문제 1: T-bar 오브젝트 배치
│   │   ├── custom_pickplace_env_cfg_3_1_practice_2.py                  # 문제 2: 카메라 관측 설정
│   │   └── custom_pickplace_env_cfg_3_1_answer.py                      # 최종 완성본 (참고용)
│   │
│   ├── task/lift/config/
│   │   ├── ik_abs_env_cfg_3_1_practice.py                              # 문제 3: IK 액션 컨트롤러 교체
│   │   ├── ik_abs_env_cfg_3_1_answer.py                               # 문제 3 정답
│   │   ├── joint_pos_env_cfg_3_1_practice.py                           # 문제 4: 이진 그리퍼 제어
│   │   └── joint_pos_env_cfg_3_1_answer.py                            # 문제 4 정답
│   │
│   ├── day3_1_pickplace_statemachine_collect_data_practice.py          # 문제 5: 성공 에피소드만 저장
│   ├── day3_1_pickplace_statemachine_collect_data_answer.py            # State Machine 최종 완성본
│   │
│   ├── day3_1_pickplace_teleop_collect_data_practice.py                # 문제 6: 델타 적분 및 액션 조립
│   └── day3_1_pickplace_teleop_collect_data_answer.py                  # Teleop 최종 완성본
│
├── [2교시: PushT 데이터 수집 (Teleop)]
│   ├── task/lift/
│   │   ├── custom_pusht_env_cfg_3_2_practice.py                        # 문제 7: 커스텀 리셋 (도메인 랜덤화)
│   │   └── custom_pusht_env_cfg_3_2_answer.py                          # 최종 완성본 (참고용)
│   │
│   ├── task/lift/mdp_3_2/
│   │   ├── terminations_practice.py                         # 문제 8: PushT 성공 판정 구현
│   │   └── terminations_answer.py                           # 문제 8 정답
│   │
│   └── day3_2_pusht_teleop_collect_data_answer.py           # 데이터 수집 실행 스크립트
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
│   ├── day3_4_check_train_config.py      # 학습 전 config↔데이터셋 정합성 검사
│   ├── train.sh                          # 검사 통과 시 학습 실행 (권장)
│   ├── day3_4.99_preprocess_hdf5.py      # 이미지 resize + (레거시) float→uint8 변환
│   └── robomimic/                       # git submodule (학습 프레임워크)
│       └── robomimic/scripts/train.py
│
├── [5교시: Diffusion Policy 모델 평가]
│   ├── day3_5.1_eval_practice.py        # 문제 1: 체크포인트 로드 + 환경 생성
│   ├── day3_5.1_eval_answer.py          # 문제 1 정답
│   ├── day3_5.2_eval_practice.py        # 문제 2: obs 변환 + 전체 Rollout
│   ├── day3_5.2_eval_answer.py          # 문제 2 정답
│   ├── day3_5_eval_answer.py            # Full eval (참고용)
│   ├── day3_5_eval_generalization.py    # 일반화 성능 평가 (Visual DR / spawn 범위)
│   └── day3_5_legacy_img_norm.py        # (레거시) normalize=True 데이터 재현용 인코딩 유틸

├── datasets/
│   ├── tbar_pusht_teleop_practice.hdf5      # 3,4교시 Visual DR 입력
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

아래 명령어는 모두 **`day3/` 디렉토리**에서 실행합니다.

---

### 1교시 — Pick & Place 데이터 수집 (State Machine & Teleop)

1교시에서는 T-bar를 바구니에 담는 Pick & Place 작업을 자동(State Machine)과 수동(Teleop)으로 수집합니다.

#### 1. 자동 수집 (State Machine)

State Machine 기반으로 로봇이 사전 정의된 상태(REST → PREGRASP → GRASP → LIFT 등)를 전이하며 **자동으로 물체를 잡고 바구니에 담는 시연** 데이터를 수집합니다.

#### 🔹 최종 완성본 실행

```bash
python day3_1_pickplace_statemachine_collect_data_answer.py \
    --num_envs 4 \
    --num_demos 50 \
    --dataset_file ./datasets/tbar_pickplace_statemachine_practice.hdf5
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num_envs` | `4` | 병렬 환경 개수 |
| `--num_demos` | `50` | 수집할 성공 데모 수 (0 = 무한) |
| `--max_steps` | `2000` | 환경별 타임아웃 스텝 수 |
| `--dataset_file` | `./datasets/tbar_pickplace_statemachine_practice.hdf5` | 저장 경로 |

---

#### 2. 수동 수집 (Teleop)

키보드로 로봇을 직접 조종하며 Pick & Place 데모를 수집합니다.

#### 🔹 최종 완성본 실행

```bash
python day3_1_pickplace_teleop_collect_data_answer.py \
    --num_demos 50 \
    --dataset_file ./datasets/tbar_pickpalce_teleop_practice.hdf5 \
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
| `--dataset_file` | `./datasets/tbar_pickpalce_teleop_practice.hdf5` | 저장 경로 |

---

### 2교시 — PushT 데이터 수집 (Teleop)

2교시에서는 T-bar를 밀어 목표 위치(x=0.4, y=0.4)에 정렬하는 PushT 작업을 수집합니다.

#### 🔹 최종 완성본 실행

```bash
python day3_2_pusht_teleop_collect_data_answer.py \
    --task Template-PushT-Franka-v0 \
    --teleop_device keyboard \
    --enable_cameras \
    --dataset_file ./datasets/tbar_pusht_teleop_practice.hdf5 \
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
| `--dataset_file` | `./datasets/tbar_pusht_teleop_practice.hdf5` | 저장 경로 |

---

### 3교시 — 데이터 증강 (Visual DR & IsaacLab Mimic)

3교시에서는 HDF5 내부 숫자를 직접 바꾸지 않습니다. 저장 state를 IsaacLab에서 다시 렌더링하거나, `isaaclab_mimic` DataGenerator rollout으로 새 궤적을 생성합니다.

모든 3교시 명령은 **프로젝트 루트**에서 실행합니다.

```bash
cd /workspace/AILAB-summer-school-2026-private
conda activate env_isaaclab
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
$ISAACLAB_PATH/isaaclab.sh -p day3/day3_3.1.2_pusht_visual_dr_replay_answer.py
python day3/day3_3.5_2subtask_generation_answer.py --generation_num_trials 3
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
python day3/day3_3.6_multisubtask_generation_answer.py \
  --generation_num_trials 1 \
  --visualize_subtasks
```

### 4교시 — Diffusion Policy 학습

4교시에서는 1~3교시에서 수집·증강한 HDF5 데이터셋으로 Diffusion Policy(DP)를 학습합니다. (실습 Task: Push-T)
실습 순서: **① 데이터셋 구조 확인 → ② Config 작성(빈칸 채우기) → ③ 학습**

#### 0. 사전 확인 — 이미지 인코딩

<details>
<summary>(`ObsTerm`)의 `base_mdp.image` 가 기본값 `normalize=True` 인 경우</summary>

```
normalize=True  →  이미지 = rgb/255 - 프레임별 평균   (음수 float, RGB 의미 손실) 
```
상태로 저장하면 robomimic이 전제하는 uint8과 일치하지 않아 **별도 변환(day3_4.99)** 이 필요해지며, 해당 변환은 데이터셋마다 다른 상수를 생성하여 **학습/평가 불일치(검은 화면) 오류**의 원인이 됩니다.

| | 수집 시 설정 | 저장 형식 | 전처리 | 평가 인코딩 |
|---|---|---|---|---|
| **권장** | `ObsTerm(..., params={..., "normalize": False})` | uint8 RGB | **불필요** | 자동(raw) |
| (레거시) | `normalize=True` (기본값) | float32 | `day3_4.99` 로 uint8 변환 | 자동(legacy, 상수 필요) |

> 평가 스크립트(`day3_5_eval_*.py`)는 인코딩과 정규화 상수를 **자동 결정**하므로 별도 인자가 필요 없습니다. 체크포인트 옆 사이드카(`img_norm_stats.json`) → 체크포인트가 기억하는 학습 hdf5 → (없으면) raw 순으로 판별합니다.
> **단, 이미지 resize 는** — 수집 해상도(예: 480×640)가 학습 config 해상도(예: `*_resized` config 의 240×320)보다 크면, raw·legacy 여부와 무관하게 `day3_4.99` 로 미리 축소해야 crop 설정과 맞습니다.
</details>


<details>
<summary>데이터 전처리 (day3_4.99): 이미지 resize+ [레거시:optional] float32→uint8 변환</summary>

```bash
# 구조 확인 (dry-run)
python day3_4.99_preprocess_hdf5.py --input datasets/tbar_pusht_teleop_practice.hdf5 --dry-run

# 이미지 resize (예: 480×640 → 240×320). float(legacy) 데이터면 uint8 변환도 자동 수행
python day3_4.99_preprocess_hdf5.py \
    --input datasets/tbar_pusht_teleop_practice.hdf5 \
    --output datasets/tbar_pusht_teleop_practice_resized.hdf5 \
    --height 240 --width 320

# resize 없이 dtype 변환만 (원본 해상도 유지 · legacy float 데이터 전용)
python day3_4.99_preprocess_hdf5.py \
    --input datasets/tbar_pusht_teleop_practice.hdf5 \
    --output datasets/tbar_pusht_teleop_practice_dtype.hdf5 \
    --no_resize
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` / `-i` | (필수) | 입력 HDF5 경로 |
| `--output` / `-o` | `./resized.hdf5` | 출력 HDF5 경로 |
| `--height` / `--width` | 240 / 320 | resize 목표 해상도 |
| `--no_resize` | OFF (기본은 resize 수행) | resize 를 건너뛰고 (레거시) float→uint8 변환만 |
| `--dry-run` | OFF | 구조만 확인하고 종료 |

</details>

#### 1. 데이터셋 구조 확인

Config를 작성하기 **전에** obs key 이름과 이미지 크기를 확인합니다. 이 값들이 config에 그대로 들어갑니다.

```bash
python robomimic/robomimic/scripts/get_dataset_info.py --dataset <데이터셋.hdf5>
```


확인할 것:
- **obs key 이름** — `joint_pos`, `top_cam`, `wrist_cam` … (config의 `low_dim`/`rgb`에 그대로)
- **이미지 크기** — 예: `(480, 640, 3)` → crop은 이 크기의 약 90% (예: 432×576)
- **이미지 dtype** — uint8이면 raw, float32면 레거시(전처리 필요)

#### 2. Config 작성 (빈칸 채우기)

`configs/day3_4_pusht_teleop_dp_config_practice.json` 의 `"???"` **3개**를 채웁니다.

| 빈칸 | 항목 | 중요성 |
|---|---|---|
| **`low_dim`** | 저차원 obs key | 데이터셋 구조(①단계)를 확인해야 채울 수 있음 |
| **`rgb`** | 카메라 key | 사용하는 카메라 수가 학습 속도와 안정성에 영향을 줌.  |
| **`pretrained`** | ImageNet 가중치 | **소규모 데이터(수십 개)**|

| 항목 | 의미 |
|---|---|
| `pretrained` | 시각 인코더 사전학습 |
| `observation_horizon` (To=2) | 관찰 프레임 수 | t-1, t 두 프레임의 **차이**로 움직임을 파악 |
| `prediction_horizon` (Tp=16) | 한 번에 예측하는 액션 수 | 16개 예측 후 8개만 실행하고 재계획 |
| `ddpm` / `ddim` | denoising 샘플러 |
| `epoch_every_n_steps` (100) | epoch당 배치 수 |

이 외 config 파일(`pickplace_dp_config_resized.json`, `pusht_teleop_dp_config_resized.json`)은 빈칸 없이 완성된 참고용입니다.

#### 3. 학습 실행 (검사 → 학습)

```bash
./train.sh configs/day3_4_pusht_teleop_dp_config_practice.json datasets/<학습데이터.hdf5>
```
config를 잘못 작성하면 학습이 즉시 오류로 중단되거나, 잘못된 입력으로 학습이 진행되어 시간을 낭비할 수 있습니다.
`train.sh` 는 학습에 앞서 **config↔데이터셋 정합성 검사**를 먼저 수행하고, 검사를 통과한 경우에만 학습을 시작합니다.

or 
> Config 체크: 
`python day3_4_check_train_config.py --config configs/<학습config.json>  --dataset datasets/<학습데이터.hdf5>`

> Train 실행: `python robomimic/robomimic/scripts/train.py --config <cfg> --dataset <ds>`

- 학습 로그 확인
```bash
 tensorboard --logdir <experiment-log-dir> 
 ```


---

### 5교시 — Diffusion Policy 평가

5교시에서는 학습된 Diffusion Policy를 Isaac Lab 환경에서 rollout하여 성공률을 측정합니다. 5.1에서 환경을 정상 생성하는지 확인한 뒤, 5.2에서 obs 변환과 전체 rollout을 수행합니다.

#### 문제 흐름

| 문제 | 파일 | TODO | 구현하는 핵심 |
|---|---|---|---|
| 5.1 | `day3_5.1_eval_practice.py` | 2개 | `importlib`으로 env_cfg 로드, `concatenate_terms=False` 설정 |
| 5.2 | `day3_5.2_eval_practice.py` | 2개 | Low-dim `(dim,)→(1,T,dim)`, Image `(H,W,C)→(1,T,H,W,C)` 변환 |

#### 성공 판정 기준

성공 판정은 1·2교시 데이터 수집과 동일한 함수를 사용합니다.
| task_type | 판정 함수 | 기준 |
|---|---|---|
| `pickplace` | `mdp_3_1.terminations.object_pickplace_goal` | XY < 15cm, Z < 10cm |
| `pusht` | `mdp_3_2.terminations_answer.object_pusht_goal` | XY < 1cm, Yaw < 0.1 rad |

#### 🔹 정답 실행

```bash
# 5.1: 환경 생성 실습
$ISAACLAB_PATH/isaaclab.sh -p day3_5.1_eval_practice.py \
    --task_type pusht \
    --checkpoint <체크포인트.pth>

# 5.2: rollout 실습
$ISAACLAB_PATH/isaaclab.sh -p day3_5.2_eval_practice.py \
    --task_type pusht \
    --checkpoint <체크포인트.pth> \
    --num_rollouts 20 --max_steps 300

# Full eval (참고용) — 공정 비교용 인자 포함
$ISAACLAB_PATH/isaaclab.sh -p day3_5_eval_answer.py \
    --task_type pusht \
    --checkpoint <체크포인트.pth> \
    --num_rollouts 20 --max_steps 300 \
    --eval_seed 1000 

# (선택) 빠른 추론: DDIM 샘플러로 전환 (재학습 불필요, 권장 10~16스텝)
$ISAACLAB_PATH/isaaclab.sh -p day3_5_eval_answer.py \
    --task_type pusht --checkpoint <체크포인트.pth> \
    --num_rollouts 20 --eval_seed 1000 --ddim_steps 16
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--task_type` | `pusht` | 평가할 태스크: `pickplace` 또는 `pusht` |
| `--checkpoint` | (필수) | 학습된 체크포인트 `.pth` 경로 |
| `--num_rollouts` | `20` | 평가 에피소드 수 (확인용 20, 최종 비교는 50~100 권장) |
| `--max_steps` | `300` | 에피소드당 최대 스텝 수 |
| `--eval_seed` | - | 초기 배치 고정. **여러 모델·epoch를 동일 조건에서 공정하게 비교** |
| `--ddim_steps` | - | 추론 샘플러를 **DDIM**으로 전환(권장 10~16). 미지정 시 학습된 DDPM 100 유지. 기존 결과 재현 시 생략 |

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
$ISAACLAB_PATH/isaaclab.sh -p day3_5_eval_generalization.py \
    --task_type pusht --visual_dr \
    --checkpoint <체크포인트.pth> --num_rollouts 20

# PickPlace: 넓은 spawn 범위 일반화 테스트 
$ISAACLAB_PATH/isaaclab.sh -p day3_5_eval_generalization.py \
    --task_type pickplace --spawn_range wide \ 
    --checkpoint <체크포인트.pth> --num_rollouts 20 --max_steps 600
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--spawn_range` | `original` | 초기 위치 범위: `original`, `wide`, `extreme` |
| `--visual_dr` | OFF | Visual DR 활성화 |

> `original`은 데이터 수집과 동일한 범위, `wide`는 mimic 증강에서 사용한 확장 범위, `extreme`은 학습 범위 밖(OOD) 테스트입니다.
---