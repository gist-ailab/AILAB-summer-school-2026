# Ubuntu 22.04 + Isaac Sim 5.1.0 pip install + Isaac Lab v2.3.2.
# This image intentionally starts from Ubuntu 22.04 instead of NVIDIA Isaac Sim's
# prebuilt container, so the OS version stays fixed for the lecture machines.
FROM ubuntu:22.04

SHELL ["/bin/bash", "-lc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG ISAACLAB_VERSION=v2.3.2
ARG AILAB_LD_PRELOAD=""
ARG ISAACLAB_PATH=/workspace/IsaacLab
ARG CONDA_DIR=/opt/conda
ARG CONDA_ENV=isaaclab

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    CONDA_DIR=${CONDA_DIR} \
    CONDA_DEFAULT_ENV=${CONDA_ENV} \
    PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:${PATH} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8

USER root

# System packages needed by Isaac Sim pip, Isaac Lab, OpenGL/X11, and native Python builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ca-certificates \
    apt-transport-https \
    gpg \
    curl \
    wget \
    bzip2 \
    cmake \
    build-essential \
    pkg-config \
    locales \
    fonts-nanum \
    fonts-noto-cjk \
    libgl1 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libx11-dev \
    libxcb1 \
    libxcursor1 \
    libxcursor-dev \
    libxext6 \
    libxi6 \
    libxi-dev \
    libxinerama1 \
    libxinerama-dev \
    libxrandr2 \
    libxrandr-dev \
    libxrender1 \
    libxtst6 \
    libxt6 \
    libnss3 \
    libasound2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    && locale-gen ko_KR.UTF-8 \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*


# Install the VS Code package matching the native build machine architecture.
RUN NATIVE_ARCH="$(dpkg --print-architecture)" \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor > /tmp/packages.microsoft.gpg \
    && install -D -o root -g root -m 644 /tmp/packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg \
    && echo "deb [arch=${NATIVE_ARCH} signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
        > /etc/apt/sources.list.d/vscode.list \
    && rm /tmp/packages.microsoft.gpg \
    && apt-get update \
    && apt-get install -y --no-install-recommends code \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge for both x86_64 and aarch64 hosts, then create the Isaac Lab env.
RUN set -euo pipefail; \
    case "$(dpkg --print-architecture)" in \
      amd64) MINIFORGE_ARCH="x86_64" ;; \
      arm64) MINIFORGE_ARCH="aarch64" ;; \
      *) echo "Unsupported native architecture: $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac; \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -O /tmp/miniforge.sh; \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}"; \
    rm /tmp/miniforge.sh; \
    conda create -y -n "${CONDA_ENV}" python=3.11; \
    conda clean -afy

# README-aligned Python dependencies.
RUN python -m pip install --upgrade pip "setuptools<81" wheel
RUN pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
RUN pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# flatdict 4.0.1 is an old sdist that imports pkg_resources during setup.
# Use the environment setuptools instead of pip build isolation.
RUN pip install --no-build-isolation flatdict==4.0.1 prettytable==3.3.0

WORKDIR /workspace
RUN git clone --depth 1 --branch ${ISAACLAB_VERSION} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

WORKDIR ${ISAACLAB_PATH}
# On native arm64 (DGX Spark), Isaac Lab's installer requires the system OpenMP
# runtime before libraries bundled with Isaac Sim.
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
      LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ./isaaclab.sh --install; \
    else \
      ./isaaclab.sh --install; \
    fi

# Make the Isaac Lab source packages explicit in the conda environment.
# IsaacLab's helper installs dependencies above; --no-deps prevents pip from
# rebuilding already-handled legacy dependencies during Docker builds.
RUN python -m pip install --no-deps \
    -e "${ISAACLAB_PATH}/source/isaaclab" \
    -e "${ISAACLAB_PATH}/source/isaaclab_assets" \
    -e "${ISAACLAB_PATH}/source/isaaclab_tasks" \
    -e "${ISAACLAB_PATH}/source/isaaclab_rl[all]" \
    -e "${ISAACLAB_PATH}/source/isaaclab_mimic[all]"

# Smoke-test package availability. Full Isaac Sim modules should still be imported after
# SimulationApp in runtime scripts, so this check avoids importing isaacsim directly.
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
      LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
        ./isaaclab.sh -p -c "import isaaclab; import isaaclab_mimic; print('Isaac Lab + Isaac Lab Mimic import OK')"; \
    else \
      ./isaaclab.sh -p -c "import isaaclab; import isaaclab_mimic; print('Isaac Lab + Isaac Lab Mimic import OK')"; \
    fi

# ---------------------------------------------------------------------------
# NVIDIA Vulkan runtime config + unzip (used by the data step below).
# This image is built from ubuntu:22.04 (not NVIDIA's Isaac Sim base), so it
# ships no Vulkan loader or ICD. Without them Isaac Sim's renderer fails with
# VK_ERROR_INCOMPATIBLE_DRIVER ("Vulkan 1.1 is not supported") and PhysX falls
# back to the CPU solver. libGLX_nvidia.so.0 is injected at runtime by the NVIDIA
# Container Toolkit (NVIDIA_DRIVER_CAPABILITIES=all); here we add the Vulkan
# loader plus an ICD file pointing at it. Placed right after the heavy Isaac Sim
# / Isaac Lab layers so those stay cached on rebuilds.
# NOTE: the container must still be launched with an X display (ENABLE_X11=1);
# headless-without-display cannot create a Vulkan instance with this driver.
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 vulkan-tools unzip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/vulkan/icd.d \
    && printf '%s\n' \
       '{' \
       '    "file_format_version" : "1.0.0",' \
       '    "ICD": {' \
       '        "library_path": "libGLX_nvidia.so.0",' \
       '        "api_version" : "1.3.242"' \
       '    }' \
       '}' > /usr/share/vulkan/icd.d/nvidia_icd.json

# ---------------------------------------------------------------------------
# Self-contained project layer: code + all Python deps + course data/checkpoints are
# baked into the image so it can be `docker save`d and run on fresh workstations
# with no host repo and no downloads. Driven by docker/isaaclab/setup_docker.sh.
# Steps are ordered so that editing course code re-runs only the final COPY, not
# the heavy dependency / data-download layers.
# ---------------------------------------------------------------------------
WORKDIR /workspace/AILAB-summer-school-2026

# 1) Python deps first (requirements.txt pulls sam3 via git into ./src/sam3,
#    installs ./cgnet/graspnetAPI, gdown, and the shared PyPI stack).
COPY requirements.txt ./
COPY cgnet ./cgnet
COPY docker/isaaclab/setup_docker.sh ./docker/isaaclab/setup_docker.sh
RUN bash docker/isaaclab/setup_docker.sh deps

# 2) robomimic (day3 imitation learning) — cloned into ./day3/robomimic.
RUN bash docker/isaaclab/setup_docker.sh robomimic

# 3) Course data (asset/checkpoint zips + day3 demo hdf5) baked in.
RUN bash docker/isaaclab/setup_docker.sh data

# Smoke-test the extra project dependencies used in later lectures.
RUN python -c "import cv2, open3d, pycocotools; import graspnetAPI; print('Project dependencies import OK')"

# NVIDIA Container Toolkit injects the host-compatible ICD at
# /etc/vulkan/icd.d/nvidia_icd.json when the container starts. Remove the
# temporary image-baked ICD after the cached data layers so Isaac Sim sees the
# same physical GPU only once while rebuilds can still reuse those layers.
RUN rm -f /usr/share/vulkan/icd.d/nvidia_icd.json

# SAM3 weights are included in the Google Drive checkpoint.zip downloaded by
# setup_docker.sh above. Runtime code loads data/checkpoint/sam3/sam3.1_multiplex.pt
# with load_from_HF=False, so neither an HF token nor an HF cache is required.

# 4) The rest of the repo code, last (code edits don't bust deps/data layers).
#    .dockerignore keeps downloaded ./data, ./src, ./day3/robomimic and
#    ./day3/datasets out of the context. Tracked ./day3/data assets are included.
COPY . .
RUN bash docker/isaaclab/setup_docker.sh verify

WORKDIR /workspace/AILAB-summer-school-2026

# Native ARM64 builds pass the tested system OpenMP / scikit-learn OpenMP /
# Conda C++ preload from build.sh. Fail the build if a pinned runtime library is
# missing instead of producing an image that fails later during Python import.
RUN if [[ -n "${AILAB_LD_PRELOAD}" ]]; then \
      test -f /lib/aarch64-linux-gnu/libgomp.so.1; \
      test -f /opt/conda/envs/isaaclab/lib/python3.11/site-packages/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0; \
      test -f /opt/conda/envs/isaaclab/lib/libstdc++.so.6; \
    fi

# On x86_64 AILAB_LD_PRELOAD is empty, so no ARM-specific library is loaded.
ENV LD_PRELOAD=${AILAB_LD_PRELOAD}

CMD ["bash"]
