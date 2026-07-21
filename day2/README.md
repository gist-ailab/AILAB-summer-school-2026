# Day2: Isaac Sim and Isaac Lab Tabletop Simulation

이 폴더는 2026 인공지능 여름학교 Day2 실습 자료입니다. Isaac Sim GUI에서 tabletop scene을 구성한 뒤, 같은 흐름을 Isaac Lab 코드로 옮겨 object, robot, camera observation, semantic mask, bbox 데이터를 다루는 과정을 학습합니다.

## Target Version

- Isaac Sim: 5.1.0
- Isaac Lab: 2.3.x 계열
- 실행 위치: 프로젝트 저장소 루트
- Camera sensor 사용 시 `--enable_cameras` 옵션 필요

## Folder Structure

```text
day2/
  assets/          # robot, object USD asset 및 texture
  scenes/          # GUI에서 저장한 tabletop/object/observation scene
  observations/    # camera observation 저장 결과
  *.py             # Isaac Lab answer/practice scripts
```

## Lecture Flow

### 1. Tabletop Scene 생성

GUI에서 Ground, Table, Light, Camera를 만들던 과정을 Isaac Lab config 코드로 구성합니다.

- Answer: `day2_2.1_custom_tabletop_answer.py`
- Practice: `day2_2.1_custom_tabletop_practice.py`
- 주요 개념: `AppLauncher`, `@configclass`, `InteractiveSceneCfg`, `AssetBaseCfg`, `RigidObjectCfg`, `CameraCfg`, `SimulationContext`
- 확인 결과: Stage Tree에서 `/World/envs/env_0/Table`, `/World/envs/env_0/Camera` 확인

실행:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day2/day2_2.1_custom_tabletop_answer.py --enable_cameras
```

### 2. Object와 Custom Robot 추가

이전 시간에 만든 tabletop scene을 불러오고, object asset과 Hand-E gripper가 결합된 custom Franka robot을 배치합니다.

- Answer: `day2_2.2_objects_robots_answer.py`
- Practice: `day2_2.2_objects_robots_practice.py`
- 주요 개념: `UsdFileCfg`, `RigidObjectCfg`, `ArticulationCfg`, `ImplicitActuatorCfg`, `joint_pos`, `actuators`
- 확인 결과: Stage Tree에서 `Object1`, `Object2`, `Object3`, `Robot` 확인

실행:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day2/day2_2.2_objects_robots_answer.py --enable_cameras
```

### 3. Camera Observation 취득

기존 scene의 Camera를 Isaac Lab sensor로 연결하고, RGB, depth, semantic segmentation mask, bbox를 저장합니다.

- Answer: `day2_2.3_get_observations_answer.py`
- Practice: `day2_2.3_get_observations_practice.py`
- 주요 개념: `CameraCfg`, `Camera`, `data_types`, `semantic_segmentation`, `add_labels`, `camera.data.output`, `camera.data.info`, bbox 후처리
- 확인 결과: `observations/` 폴더에 RGB/depth/mask/bbox 파일 생성

실행:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day2/day2_2.3_get_observations_answer.py --enable_cameras
```

Headless 실행:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day2/day2_2.3_get_observations_answer.py --enable_cameras --headless
```

## Expected Observation Outputs

`day2_2.3_get_observations_answer.py` 실행 후 `observations/` 폴더에 다음 파일이 생성됩니다.

```text
observations/
  rgb.png
  depth.npy
  depth_vis.png
  semantic_segmentation.npy
  semantic_segmentation_vis.png
  bbox_2d.json
```

- `rgb.png`: camera RGB image
- `depth.npy`: 실제 depth 값 배열
- `depth_vis.png`: depth 확인용 시각화 이미지
- `semantic_segmentation.npy`: pixel별 semantic id mask
- `semantic_segmentation_vis.png`: semantic mask 확인용 색상 이미지
- `bbox_2d.json`: semantic mask에서 계산한 visible 2D bounding box

## Practice Files

Practice 파일은 answer 코드에서 중요한 Isaac Lab 키워드만 빈칸으로 둔 버전입니다. 학생들은 빈칸을 채우며 함수명과 config 구조에 익숙해지는 것을 목표로 합니다.

추천 진행 방식:

1. 강사가 주요 개념을 먼저 설명
2. Practice 코드의 TODO를 직접 입력
3. 실행 후 GUI Stage Tree 또는 `observations/` 결과 확인
4. Answer 코드와 비교

## Notes

- `observations/`는 실행 결과물이므로 매 실행마다 덮어써질 수 있습니다.
- Camera 관련 코드는 반드시 `--enable_cameras` 옵션과 함께 실행합니다.
- USD asset은 repo 내부 상대 경로를 기준으로 참조되도록 유지하는 것이 좋습니다.
- 대용량 USD와 texture 파일은 저장소에 일반 Git 파일로 포함되어 있습니다.
