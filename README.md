# AILAB Summer School 2026

AILAB Summer School 2026 실습 코드입니다. 교육 일정은 2026년 7월 22일부터 24일까지입니다.

## 지원 환경

| 환경 | 실행 방식 | 상태 |
|---|---|---|
| Ubuntu 22.04 x86_64 + NVIDIA GPU | 로컬 또는 Docker | 지원 |
| NVIDIA DGX Spark aarch64 | Podman 네이티브 빌드 | 지원 |

QEMU를 이용한 ARM64 크로스 빌드는 지원하지 않습니다. DGX Spark 이미지는 Spark에서 직접 빌드합니다.

모든 명령은 별도 안내가 없는 한 저장소 루트에서 실행합니다.

```bash
cd AILAB-summer-school-2026
```

## 1. 로컬 설치

### 사전 준비

- NVIDIA GPU 및 호환 드라이버
- Ubuntu 22.04
- Conda
- Git, CMake, 컴파일 도구, unzip

```bash
sudo apt update
sudo apt install -y git cmake build-essential unzip
```

### Isaac Sim 및 Isaac Lab 설치

```bash
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

python -m pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

cd ~
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
export ISAACLAB_PATH="$HOME/IsaacLab"
```

설치 확인:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p \
  "$ISAACLAB_PATH/scripts/tutorials/00_sim/create_empty.py" \
  --headless
```

### 실습 저장소 설치

```bash
cd ~
git clone https://github.com/gist-ailab/AILAB-summer-school-2026.git
cd AILAB-summer-school-2026
conda activate isaaclab
bash setup.sh
```

`setup.sh`는 Python 패키지와 robomimic을 설치하고 Google Drive에서 에셋, 실습 데이터 및 로컬 체크포인트를 내려받습니다. 다운로드 용량이 크므로 충분한 디스크 공간과 안정적인 네트워크가 필요합니다.

### 다운로드되는 주요 파일

```text
data/
├── assets/
├── checkpoint/
│   ├── contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth
│   └── sam3/sam3.1_multiplex.pt
├── handeye_data/
├── slam_map_data/
└── sam3_practice/

day3/datasets/
└── tbar_pickplace_teleop_0719_240x320.hdf5
```

SAM3는 Hugging Face가 아니라 위 로컬 체크포인트를 사용하므로 Hugging Face 로그인이나 토큰이 필요하지 않습니다.

## 2. Docker 설치

호스트에 NVIDIA 드라이버, Docker, NVIDIA Container Toolkit이 필요합니다.

저장소 루트에서 실행합니다.

```bash
./docker/isaaclab/build.sh
./docker/isaaclab/run.sh
```

GUI 실행:

```bash
xhost +local:root
ENABLE_X11=1 ./docker/isaaclab/run.sh
```

컨테이너 안에서 VS Code 실행:

```bash
code --no-sandbox --disable-gpu --user-data-dir=/tmp/vscode-root
```

종료한 컨테이너 재접속:

```bash
docker start -ai ailab-isaaclab
```

자세한 내용은 [Docker 설치 안내](docker/isaaclab/README.md)를 참고합니다.

## 3. DGX Spark

DGX Spark에서는 Docker 호환 명령이 Podman을 호출하므로 rootful Podman으로 네이티브 빌드합니다.

```bash
sudo env CONTAINER_ENGINE=podman \
  IMAGE_NAME=ailab-isaaclab:2.3.2-isaacsim5.1.0-arm64 \
  ./docker/isaaclab/build.sh

sudo env CONTAINER_ENGINE=podman \
  IMAGE_NAME=ailab-isaaclab:2.3.2-isaacsim5.1.0-arm64 \
  ENABLE_X11=1 \
  ./docker/isaaclab/run.sh
```

Spark에서 `build.sh`로 만든 ARM64 이미지는 모든 컨테이너 프로세스에 다음
`LD_PRELOAD`를 적용합니다. 일반 x86_64 빌드에서는 빈 값으로 설정됩니다.

```bash
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:/opt/conda/envs/isaaclab/lib/python3.11/site-packages/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0:/opt/conda/envs/isaaclab/lib/libstdc++.so.6"
```

Podman 컨테이너 재접속:

```bash
sudo podman start -ai ailab-isaaclab
```

## 실행 원칙

모든 실습 명령은 저장소 루트에서 실행합니다. Isaac Lab 프로그램은 다음 launcher를 사용합니다.

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p day2/day2_2.1_custom_tabletop_answer.py --enable_cameras
```

Docker 내부의 기본값은 다음과 같습니다.

```bash
export ISAACLAB_PATH=/workspace/IsaacLab
```

SAM3 단독 추론:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p \
  day2/day2_4.0.sam3_inference.py \
  --input data/sam3_practice/images/truck.jpg \
  --prompt "truck"
```

Day별 상세 실행 방법:

- [Day 2](day2/README.md)
- [Day 3](day3/README.md)

## 연습 파일과 정답 파일

- `*_practice.py`: 학생이 빈칸을 채우는 파일이며 완성 전에는 실행되지 않을 수 있습니다.
- `*_answer.py`: 실행 가능한 정답 파일입니다.

## 알려진 환경 차이

- DGX Spark에서는 rootful Podman과 NVIDIA CDI 장치를 사용합니다.
- Spark의 최종 Torch/CUDA 버전은 Isaac Sim과 Isaac Lab 설치 과정에서 초기 설치 버전과 달라질 수 있습니다.
- ARM64의 Open3D 버전은 x86_64와 다르게 설치됩니다.
- SAM3와 Contact-GraspNet 체크포인트 및 Day3 HDF5 데이터는 빌드 또는 `setup.sh` 실행 중 다운로드됩니다.
