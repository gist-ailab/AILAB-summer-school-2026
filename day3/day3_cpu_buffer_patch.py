"""
day3_cpu_buffer_patch.py
========================
DGX Spark 등 제한된 VRAM 환경에서 데이터 수집 시 GPU 메모리 누적을 방지하기 위한 패치.

문제:
    IsaacLab의 EpisodeData.add()는 매 스텝 텐서를 .clone() 하여 GPU list에 쌓는다.
    카메라 이미지(480×640×3) × 에피소드 길이(2000스텝)가 쌓이면 에피소드당 ~수 GB가
    GPU VRAM에 누적되어 OOM → 프로세스 강제 종료가 발생한다.

해결:
    EpisodeData.add()를 monkey-patch하여 텐서를 버퍼에 추가하는 시점에 즉시 .cpu()로 이동.
    IsaacLab 소스는 건드리지 않으며, 이 파일을 import하는 것만으로 패치가 적용된다.

사용법:
    수집 스크립트에서 AppLauncher 이후 다음 한 줄을 추가하면 됩니다:

        import day3_cpu_buffer_patch  # noqa: F401  (GPU→CPU 버퍼 패치)
"""

import torch
from isaaclab.utils.datasets.episode_data import EpisodeData


def _cpu_buffer_add(self, key: str, value: "torch.Tensor | dict") -> None:
    """EpisodeData.add()의 CPU 패치 버전.

    원본과 동일한 로직이지만, 텐서를 list에 추가하기 전 즉시 .detach().cpu() 를 호출하여
    GPU VRAM 누적을 방지한다.
    """
    # dict 타입은 재귀 처리 (원본과 동일)
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            self.add(f"{key}/{sub_key}", sub_value)
        return

    # ── 핵심 패치: 텐서를 즉시 CPU로 이동 ──────────────────────────────────
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    # ───────────────────────────────────────────────────────────────────────

    sub_keys = key.split("/")
    current_dataset_pointer = self._data
    for sub_key_index in range(len(sub_keys)):
        if sub_key_index == len(sub_keys) - 1:
            # list에 추가 (이미 CPU 텐서이므로 .clone() 불필요하나 일관성을 위해 유지)
            if sub_keys[sub_key_index] not in current_dataset_pointer:
                current_dataset_pointer[sub_keys[sub_key_index]] = [value]
            else:
                current_dataset_pointer[sub_keys[sub_key_index]].append(value)
            break
        if sub_keys[sub_key_index] not in current_dataset_pointer:
            current_dataset_pointer[sub_keys[sub_key_index]] = dict()
        current_dataset_pointer = current_dataset_pointer[sub_keys[sub_key_index]]


# ── 패치 적용 ─────────────────────────────────────────────────────────────────
EpisodeData.add = _cpu_buffer_add
# ─────────────────────────────────────────────────────────────────────────────

print("[CPU Buffer Patch] EpisodeData.add() patched: GPU tensors will be moved to CPU immediately.")
