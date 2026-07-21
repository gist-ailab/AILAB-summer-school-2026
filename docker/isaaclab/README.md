# Docker: Ubuntu 22.04 + Isaac Sim 5.1.0 + Isaac Lab v2.3.2

This Docker setup starts from `ubuntu:22.04`, installs a Python 3.11 conda environment, installs Isaac Sim `5.1.0` through pip, then clones and installs Isaac Lab `v2.3.2`.

This is different from using NVIDIA's prebuilt Isaac Sim container. We use this path because the lecture machine environment is intended to stay on Ubuntu 22.04.

## Build

```bash
cd /path/to/AILAB-summer-school-2026
./docker/isaaclab/build.sh
```

빌드 중 `setup_docker.sh data`가 Google Drive의 `checkpoint.zip`을 받아
`data/checkpoint/sam3/sam3.1_multiplex.pt`에 압축 해제한다. SAM3는 이 로컬
파일을 사용하므로 Hugging Face 로그인이나 토큰은 필요하지 않다.

Default image tag:

```text
ailab-isaaclab:2.3.2-isaacsim5.1.0
```

이 Dockerfile은 QEMU 크로스 빌드를 대상으로 하지 않는다. 일반 x86_64 PC에서는
x86_64 이미지를 직접 빌드하고, DGX Spark에서는 ARM64 이미지를 직접 빌드한다.
PyPI wheel 제공 범위에 따라 Open3D는 x86_64에서 0.19, ARM64에서 0.18을
설치하며, 강의 코드에서 사용하는 API는 두 버전에서 동일하게 지원된다.

DGX Spark에서는 rootful Podman을 지정하여 같은 빌드 스크립트를 실행한다.

```bash
sudo env \
  CONTAINER_ENGINE=podman \
  IMAGE_NAME=ailab-isaaclab:2.3.2-isaacsim5.1.0-arm64 \
  ./docker/isaaclab/build.sh
```

## Run

```bash
./docker/isaaclab/run.sh
```

기본적으로 NVIDIA GPU 0 하나만 컨테이너에 전달하고, NVIDIA Vulkan ICD만
사용한다. 다른 NVIDIA GPU를 선택하려면 `NVIDIA_GPU`를 지정한다.

```bash
NVIDIA_GPU=1 ./docker/isaaclab/run.sh
```

실행 스크립트는 컨테이너 엔진을 자동으로 구분한다.

- 일반 Docker: NVIDIA 런타임 + `NVIDIA_VISIBLE_DEVICES=<번호>`
- DGX Spark의 Podman 호환 Docker: `--device nvidia.com/gpu=gpu<번호>`

일반 Docker에서는 숫자 GPU ID가 AMD CDI 장치로 잘못 해석되지 않도록
NVIDIA 런타임을 명시한다.

Spark에서 rootful Podman을 직접 지정해 실행하려면 다음 명령을 사용한다.

```bash
sudo env \
  CONTAINER_ENGINE=podman \
  IMAGE_NAME=ailab-isaaclab:2.3.2-isaacsim5.1.0-arm64 \
  NVIDIA_GPU=0 \
  ENABLE_X11=1 \
  DISPLAY="$DISPLAY" \
  ./docker/isaaclab/run.sh
```

DGX Spark에서 `build.sh`로 네이티브 ARM64 이미지를 빌드하면 Docker 이미지에
아래 `LD_PRELOAD`가 저장된다. 따라서 `run.sh` 실행뿐 아니라 이후
`podman exec`로 들어간 셸을 포함해 모든 코드가 같은 설정을 사용한다.
x86_64 빌드에서는 빈 값으로 설정되어 ARM64 라이브러리를 로드하지 않는다.

```text
/lib/aarch64-linux-gnu/libgomp.so.1
/opt/conda/envs/isaaclab/lib/python3.11/site-packages/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0
/opt/conda/envs/isaaclab/lib/libstdc++.so.6
```

ARM64 빌드 마지막에는 위 세 파일이 모두 존재하는지 검사한다. scikit-learn
wheel이 바뀌어 해시가 포함된 파일명이 달라지면 빌드가 즉시 실패하므로 해당
경로를 새 wheel에 맞게 갱신해야 한다.

GUI 및 컨테이너 내부 VS Code를 사용하려면 호스트에서 X11을 활성화해 실행합니다.

```bash
xhost +local:root
ENABLE_X11=1 ./docker/isaaclab/run.sh
```

그다음 컨테이너 내부에서 VS Code를 실행합니다.

```bash
code --no-sandbox --disable-gpu --user-data-dir=/tmp/vscode-root
```

프로젝트 코드는 이미지에 포함되며 컨테이너의 다음 경로에서 실행한다.
`MOUNT_REPO`를 지정한 개발 실행에서는 호스트 저장소가 같은 경로에 마운트된다.

```text
/workspace/AILAB-summer-school-2026
```

Inside the container:

```bash
echo $ISAACLAB_PATH
"$ISAACLAB_PATH/isaaclab.sh" -p -c "import isaacsim, isaaclab, isaaclab_mimic; print('ok')"
"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/tutorials/00_sim/create_empty.py" --headless
```

## Re-enter an Existing Container

`run.sh` keeps the container after exit so Cursor can attach to it again.

```bash
docker start -ai ailab-isaaclab
# or, if already running:
docker exec -it ailab-isaaclab bash
```

DGX Spark의 rootful Podman 컨테이너는 다음과 같이 다시 접속한다.

```bash
sudo podman start -ai ailab-isaaclab
# 또는 이미 실행 중이면
sudo podman exec -it ailab-isaaclab bash
```

## Notes

- Host must have NVIDIA driver, Docker, and NVIDIA Container Toolkit installed.
- For Cursor, attach to the running `ailab-isaaclab` container and open `/workspace/AILAB-summer-school-2026`.
- Large HDF5 datasets are ignored by `.dockerignore`; mount or copy datasets separately when needed.
