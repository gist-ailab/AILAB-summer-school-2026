#!/usr/bin/env python3
"""Day3-3.1.1 PRACTICE: PushT state re-rendering.

Restore saved simulator states from a PushT teleop HDF5 file and re-render
camera observations. This checks that the saved simulator state is sufficient
to reproduce visual observations before adding domain randomization.

Run from project root:
    $ISAACLAB_PATH/isaaclab.sh -p day3/day3_3.1.1_pusht_state_rerender_practice.py \
        --input_file day3/datasets/tbar_pusht_teleop_practice.hdf5 \
        --output_file day3/datasets/pusht/pusht_rerender_test.hdf5 \
        --num_demos 3 \
        --enable_cameras

수업 연결:
- Day3 2교시 PushT teleoperation에서 저장한 HDF5 state 구조를 그대로 사용합니다.
- Day2 4교시 camera observation 코드처럼 AppLauncher 이후 camera 관련 import를 배치합니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
import sys
import time
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DAY3_ROOT.parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

import os
os.chdir(DAY3_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Re-render PushT demos from saved simulator states.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/tbar_pusht_teleop_practice.hdf5"))
parser.add_argument("--output_file", default=str(DAY3_ROOT / "datasets/pusht/pusht_rerender_test.hdf5"))
parser.add_argument("--num_demos", type=int, default=3, help="0 means replay all demos.")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--progress_interval", type=int, default=50)
# Day2 3/4교시와 같은 AppLauncher 인자 추가 패턴입니다.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 / 데이터 처리 라이브러리 임포트
# ============================================================================
import carb  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from task.lift.custom_pusht_env_cfg_3_2_answer import PushTEnvCfg  # noqa: E402
from day3_3_utils import copy_h5_attrs, copy_h5_item, delete_if_exists, ensure_h5_data_attrs, selected_demo_names  # noqa: E402

# 카메라 렌더링 활성화 (--enable_cameras flag 대체)
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# ============================================================================
# 3. 핵심 함수 및 문제 코드
# ============================================================================

# HDF5 group/dataset을 IsaacLab state 복원에 쓸 torch tensor tree로 변환합니다.
def h5_tree_to_torch(item, device: str, step: int | None = None):
    if isinstance(item, h5py.Group):
        return {key: h5_tree_to_torch(item[key], device, step) for key in item.keys()}
    data = item[()] if step is None else item[step : step + 1]
    return torch.as_tensor(data, device=device)


# 저장된 simulator state를 복원한 뒤 카메라 관측을 다시 계산합니다.
def restore_state_and_render(env: ManagerBasedEnv, state, env_ids: torch.Tensor):
    # [문제 1.1] 저장된 simulator state를 복원하고 policy camera observation을 다시 계산하세요.
    raise NotImplementedError("문제 1.1: state restore + camera re-render를 구현하세요.")


# PushT re-render에 사용할 카메라 포함 환경을 생성합니다.
def make_env() -> ManagerBasedEnv:
    env_cfg = PushTEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.camera.width = args_cli.camera_width
    env_cfg.scene.camera.height = args_cli.camera_height
    env_cfg.scene.top_camera.width = args_cli.camera_width
    env_cfg.scene.top_camera.height = args_cli.camera_height
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
    env_cfg.recorders = None
# Day2 3교시부터 사용한 ManagerBasedEnv 생성 방식입니다.
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.4, 0.0, 0.5])
    return env


# ============================================================================
# 4. 메인 함수
# ============================================================================
# PushT state re-render 실습 파이프라인을 실행합니다.
def main() -> None:
    env = make_env()
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
    input_path = Path(args_cli.input_file)
    output_path = Path(args_cli.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    exported = 0
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_h5_attrs(src, dst)
        src_data = src["data"]
        dst_data = dst.create_group("data")
        ensure_h5_data_attrs(dst_data, src_data, env_name="Template-PushT-Franka-v0")
        names = selected_demo_names(src_data, args_cli.num_demos)
        print(f"[START] state re-render source_demos={len(names)}", flush=True)

        for name in names:
            if not simulation_app.is_running():
                break
            src_demo = src_data[name]
            if "states" not in src_demo:
                print(f"[SKIP] {name}: no states group", flush=True)
                continue
            num_steps = min(int(src_demo["actions"].shape[0]), args_cli.max_steps)
            dst_demo = dst_data.create_group(f"demo_{exported}")
            copy_h5_attrs(src_demo, dst_demo)
            dst_demo.attrs["source_demo"] = name
            dst_demo.attrs["augmentation"] = "pusht_state_rerender"
            for key in src_demo.keys():
                copy_h5_item(src_demo, dst_demo, key)

            top_frames, wrist_frames = [], []
            t0 = time.perf_counter()
            for step_idx in range(num_steps):
                state = h5_tree_to_torch(src_demo["states"], env.device, step=step_idx)
                obs = restore_state_and_render(env, state, env_ids)
                top_frames.append(obs["top_cam"][0].detach().cpu().numpy())
                wrist_frames.append(obs["wrist_cam"][0].detach().cpu().numpy())
                if args_cli.progress_interval > 0 and step_idx > 0 and step_idx % args_cli.progress_interval == 0:
                    print(f"[STEP] {name}: {step_idx}/{num_steps}", flush=True)

            obs_group = dst_demo["obs"]
            delete_if_exists(obs_group, "top_cam")
            delete_if_exists(obs_group, "wrist_cam")
            obs_group.create_dataset("top_cam", data=top_frames, compression="gzip")
            obs_group.create_dataset("wrist_cam", data=wrist_frames, compression="gzip")
            exported += 1
            dst_data.attrs["total"] = sum(int(demo["actions"].shape[0]) for demo in dst_data.values() if "actions" in demo)
            dst.flush()
            print(f"[SUCCESS] {name} -> demo_{exported - 1}, frames={num_steps}, dt={time.perf_counter() - t0:.2f}s", flush=True)

    env.close()
    print(f"[DONE] exported {exported} re-rendered demos to {args_cli.output_file}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
