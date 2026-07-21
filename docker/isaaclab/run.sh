#!/usr/bin/env bash
set -euo pipefail

# Launch the self-contained AILAB image. Course code + all Python deps + the
# course data and local checkpoints are baked into the image, so a fresh workstation only needs
# the image (docker load) and this script — no repo checkout, no downloads.
#
#   ENABLE_X11=1 ./run.sh          # GUI (Isaac Sim window). Needs `xhost +local:root`.
#   ./run.sh                       # no X (note: Isaac Sim rendering needs a display)
#   MOUNT_REPO=/path/to/repo ...   # dev only: bind-mount a working copy over the
#                                  # baked code. WARNING: this also shadows the
#                                  # baked ./data — you'd need data in that copy.
IMAGE_NAME=${IMAGE_NAME:-ailab-isaaclab:2.3.2-isaacsim5.1.0}
CONTAINER_NAME=${CONTAINER_NAME:-ailab-isaaclab}
CACHE_ROOT=${CACHE_ROOT:-${HOME}/docker/isaac-sim}
ENABLE_X11=${ENABLE_X11:-0}
MOUNT_REPO=${MOUNT_REPO:-}
NVIDIA_GPU=${NVIDIA_GPU:-0}
CONTAINER_ENGINE=${CONTAINER_ENGINE:-docker}

# Docker selects the NVIDIA runtime explicitly so a numeric GPU index is not
# resolved against an AMD CDI device. DGX Spark exposes NVIDIA through Podman
# CDI, where the fully-qualified NVIDIA device name is required.
GPU_ARGS=()
if "${CONTAINER_ENGINE}" --version 2>&1 | grep -qi podman; then
  GPU_ARGS+=(
    --device "nvidia.com/gpu=gpu${NVIDIA_GPU}"
    --security-opt label=disable
  )
else
  GPU_ARGS+=(
    --runtime=nvidia
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_GPU}"
  )
fi

X11_ARGS=()
if [[ "${ENABLE_X11}" == "1" ]]; then
  XAUTHORITY_FILE=${XAUTHORITY:-${HOME}/.Xauthority}
  X11_ARGS+=(
    -e DISPLAY=${DISPLAY:-:0}
    -e XAUTHORITY=/root/.Xauthority
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
  if [[ -f "${XAUTHORITY_FILE}" ]]; then
    X11_ARGS+=(-v "${XAUTHORITY_FILE}:/root/.Xauthority:ro")
  fi
fi

MOUNT_ARGS=()
if [[ -n "${MOUNT_REPO}" ]]; then
  MOUNT_ARGS+=(-v "${MOUNT_REPO}:/workspace/AILAB-summer-school-2026:rw")
fi

mkdir -p \
  "${CACHE_ROOT}/cache/ov" \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/config" \
  "${CACHE_ROOT}/data"

"${CONTAINER_ENGINE}" run -it \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  "${X11_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  -v "${CACHE_ROOT}/cache/ov:/root/.cache/ov:rw" \
  -v "${CACHE_ROOT}/cache/pip:/root/.cache/pip:rw" \
  -v "${CACHE_ROOT}/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "${CACHE_ROOT}/config:/root/.config/ov:rw" \
  -v "${CACHE_ROOT}/data:/root/.local/share/ov:rw" \
  -w /workspace/AILAB-summer-school-2026 \
  "${IMAGE_NAME}" \
  bash
