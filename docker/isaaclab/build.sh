#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-ailab-isaaclab:2.3.2-isaacsim5.1.0}
CONTAINER_ENGINE=${CONTAINER_ENGINE:-docker}
DOCKERFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DOCKERFILE_DIR}/../.." && pwd)"

AILAB_LD_PRELOAD=""
if [[ "$(uname -m)" == "aarch64" ]]; then
  AILAB_LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:/opt/conda/envs/isaaclab/lib/python3.11/site-packages/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0:/opt/conda/envs/isaaclab/lib/libstdc++.so.6"
fi

BUILD_ARGS=(
  build
  --build-arg ISAACLAB_VERSION=v2.3.2
  --build-arg "AILAB_LD_PRELOAD=${AILAB_LD_PRELOAD}"
  -t "${IMAGE_NAME}"
  -f "${DOCKERFILE_DIR}/Dockerfile"
  "${REPO_ROOT}"
)

if "${CONTAINER_ENGINE}" --version 2>&1 | grep -qi podman; then
  # Podman's default OCI format ignores Dockerfile SHELL instructions. This
  # image uses bash features such as pipefail, so preserve Docker semantics.
  "${CONTAINER_ENGINE}" build \
    --format docker \
    "${BUILD_ARGS[@]:1}"
else
  DOCKER_BUILDKIT=1 "${CONTAINER_ENGINE}" "${BUILD_ARGS[@]}"
fi
