#!/usr/bin/env python3
"""Day3-3.2.1 ANSWER: PickPlace action replay.

기존 HDF5의 첫 simulator state에서 시작해 저장된 IK pose action을
같은 controller 환경에 다시 입력합니다. action replay가 contact 상황에서
항상 완벽히 재현되지는 않는다는 점을 확인하는 단계입니다.

수업 연결:
- Day3 1교시/2교시에서 저장한 action HDF5를 같은 IK controller 환경에 다시 넣어봅니다.
- Day2 3교시의 ManagerBasedEnv + env.step(actions) 제어 흐름과 같은 방식입니다.
"""

from __future__ import annotations

# ============================================================================
# 1. 명령행 인자 설정 및 앱 실행
# ============================================================================
import argparse
import sys
from pathlib import Path

DAY3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DAY3_ROOT.parent
if str(DAY3_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY3_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay pick-place HDF5 actions in IsaacLab.")
parser.add_argument("--input_file", default=str(DAY3_ROOT / "datasets/tbar_pickpalce_teleop_practice.hdf5"))
parser.add_argument("--replayed_file", default=str(DAY3_ROOT / "datasets/pickplace/tbar_pickplace_action_replay.hdf5"))
parser.add_argument("--num_demos", type=int, default=3)
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
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from day3_3_utils import selected_demo_names  # noqa: E402
from task.lift.config.ik_abs_env_cfg_3_1_answer import FrankaTBarPickPlaceEnvCfg  # noqa: E402

# 카메라 렌더링 활성화 (--enable_cameras flag 대체)
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

# ============================================================================
# 3. 핵심 함수 및 문제 코드
# ============================================================================

# 입력 HDF5 경로가 없을 때 루트 data 폴더의 기본 파일로 대체합니다.
def resolve_input_file(path: str) -> str:
    input_path = Path(path)
    fallback = DAY3_ROOT / "datasets/tbar_pickpalce_teleop_practice.hdf5"
    if input_path.exists():
        return str(input_path)
    if fallback.exists():
        print(f"[INPUT] {input_path} not found. Using fallback: {fallback}", flush=True)
        return str(fallback)
    raise FileNotFoundError(f"No input HDF5 found: {input_path}")


# HDF5에 저장된 특정 step의 simulator state를 reset_to 입력으로 읽습니다.
def load_state_step(group: h5py.Group, step: int, device: str):
    # HDF5 states tree를 env.reset_to()가 받을 nested tensor dict로 바꿉니다.
    state = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            state[key] = torch.as_tensor(item[step], device=device).unsqueeze(0)
        else:
            state[key] = load_state_step(item, step, device)
    return state


# PickPlace action replay에 사용할 IK action 환경과 recorder를 생성합니다.
def make_env(output_file: str) -> ManagerBasedEnv:
    # 원본 데이터가 IK pose action으로 저장되어 있으므로 같은 action controller를 사용합니다.
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    env_cfg = FrankaTBarPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
# Day3 1/2교시 데이터 수집에서 사용한 recorder 설정 방식입니다.
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(output.parent)
    env_cfg.recorders.dataset_filename = output.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
# Day2 3교시부터 사용한 ManagerBasedEnv 생성 방식입니다.
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    return env


# 현재 recorder episode를 성공 episode로 표시하고 HDF5에 내보냅니다.
def export_current_episode(env: ManagerBasedEnv) -> int:
    # 성공한 replay로 표시해서 recorder가 HDF5로 export하게 합니다.
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.ones((1, 1), dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])
    count = int(env.recorder_manager.exported_successful_episode_count)
    env.recorder_manager.reset([0])
    return count


# 저장된 action sequence를 같은 환경에서 다시 실행해 replay dataset을 만듭니다.
def replay_actions(input_file: str, output_file: str, num_demos: int) -> None:
    input_file = resolve_input_file(input_file)
    env = make_env(output_file)
    with h5py.File(input_file, "r") as src:
        for demo_index, demo_name in enumerate(selected_demo_names(src["data"], num_demos), start=1):
            demo = src[f"data/{demo_name}"]
            initial_state = load_state_step(demo["states"], 0, env.device)
            env.reset_to(initial_state, env_ids=torch.tensor([0], dtype=torch.long, device=env.device), is_relative=True)
            actions = demo["actions"][:]
            for step, action_np in enumerate(actions):
                action = torch.as_tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
                env.step(action)
                if (step + 1) % 100 == 0:
                    print(f"[REPLAY] {demo_name}: {step + 1}/{len(actions)}", flush=True)
            count = export_current_episode(env)
            print(f"[EXPORT] {demo_name} -> replay_demo_{count - 1} ({demo_index})", flush=True)
    env.close()


# ============================================================================
# 4. 메인 함수
# ============================================================================
# PickPlace action replay 파이프라인을 실행합니다.
def main() -> None:
    replay_actions(args_cli.input_file, args_cli.replayed_file, args_cli.num_demos)
    print(f"[DONE] action replay dataset: {args_cli.replayed_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
