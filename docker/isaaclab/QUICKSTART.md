# AILAB Summer School 2026 — Docker 이미지 실행 안내

배포된 Docker 이미지에는 강의 코드, 파이썬 의존성, 강의 데이터, day3 체크포인트가 모두
포함되어 있습니다. 아래 절차에 따라 이미지를 내려받아 실행하며, 별도의 데이터 다운로드는
필요하지 않습니다. 이미지를 직접 빌드하려는 경우에는 [README.md](README.md)를 참고합니다.

이 안내에서 "호스트"는 명령을 입력하는 본인 컴퓨터(실습 PC)를 의미하고, "컨테이너"는 그 위에서
실행되는 Docker 환경을 의미합니다. 2번부터 5번까지의 명령은 호스트에서 실행합니다.

## 1. 사전 요구사항
- Ubuntu 22.04
- NVIDIA GPU 및 호환 드라이버 (`nvidia-smi` 정상 동작)
- GUI 사용 시 디스플레이가 연결된 환경

## 2. Docker 설치
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```
설치 후 재로그인하여 docker 그룹을 적용합니다. `docker run --rm hello-world` 로 동작을 확인합니다.

## 3. NVIDIA Container Toolkit 설치
GPU 실행에 필요하며, 설치되지 않은 경우 `nvidia-container-runtime` 오류가 발생합니다.
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
`docker info | grep -i runtimes` 출력에 `nvidia` 가 포함되어야 합니다.

## 4. 이미지 다운로드
```bash
docker pull ghcr.io/gist-ailab/ailab-isaaclab:2.3.2-isaacsim5.1.0
```

## 5. 실행
```bash
git clone https://github.com/gist-ailab/AILAB-summer-school-2026.git
cd AILAB-summer-school-2026
xhost +local:root
IMAGE_NAME=ghcr.io/gist-ailab/ailab-isaaclab:2.3.2-isaacsim5.1.0 ENABLE_X11=1 ./docker/isaaclab/run.sh
```
실행하면 컨테이너 내부로 진입하며, 프롬프트가 `root@...:/workspace/AILAB-summer-school-2026#`
형태로 표시되면 정상입니다.

## 6. VS Code 사용
- 방법 A (컨테이너 내장 VS Code 사용): 컨테이너 내부에서 아래를 실행한 뒤
  `/workspace/AILAB-summer-school-2026` 폴더를 엽니다.
  ```bash
  code --no-sandbox --disable-gpu --user-data-dir=/tmp/vscode-root
  ```
- 방법 B (호스트의 VS Code로 컨테이너에 연결): 본인 컴퓨터(호스트)에 설치한 VS Code의
  Dev Containers 확장에서 "Attach to Running Container" 를 선택하고 `ailab-isaaclab` 에
  접속합니다.

## 7. 재접속 및 문제 해결
- 재접속: `docker start -ai ailab-isaaclab` 또는 `docker exec -it ailab-isaaclab bash`
- 컨테이너 이름 충돌: `docker rm -f ailab-isaaclab` 후 5번을 재실행합니다.
- docker 권한 오류: 재로그인 또는 `newgrp docker` 를 실행합니다.
- `unknown runtime: nvidia` 오류: 3번을 재수행합니다.

## 참고
- day3 체크포인트는 이미지의 `day3/robomimic/diffusion_policy_trained_models/` 에 포함되어
  평가에 바로 사용할 수 있습니다. (용량 문제로 체크포인트 일부만 업로드되어 있습니다. 이 점 참고 바랍니다.)
- day3 teleop 학습용 데이터셋은 이미지에 포함되지 않습니다. 학습 실습은 day3_1 로 직접 수집한
  데이터를 사용합니다.
