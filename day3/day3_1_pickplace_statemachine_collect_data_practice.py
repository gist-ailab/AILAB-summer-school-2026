# ============================================================
#  [문제 5] 데이터 기록 · 성공 에피소드만 저장 + 연속성공 디바운스
#  1교시 · Pick&Place  |  저장 위치: day3_1_1_pickplace_statemachine_collect_data.py
#  ── 할 일: 아래 TODO(문제5) 주석의 ??? 2곳을 채우세요.
#     연속 성공 스텝 카운트와, 성공 에피소드만 HDF5 로 저장하는 부분을 완성한다.
#  (이 파일 하나만으로는 실행되지 않습니다: 나머지 프로젝트 코드 필요)
# ============================================================
# T_bar Pick & Place State Machine
import argparse
import torch
from collections.abc import Sequence
import math

from isaaclab.app import AppLauncher
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

parser = argparse.ArgumentParser(description="T-bar pick and place state machine.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4, help="병렬 환경 개수")
parser.add_argument("--num_demos", type=int, default=50, help="수집할 성공 demo 개수")
parser.add_argument("--dataset_file", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "tbar_pickplace_statemachine_practice.hdf5"),
                    help="저장할 HDF5 파일 경로")
parser.add_argument("--max_steps", type=int, default=2000, help="환경별 타임아웃 스텝")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 카메라 렌더링 기본 활성화 (데이터 수집 시 카메라 센서 렌더링 필요)
if not hasattr(args_cli, "enable_cameras") or not args_cli.enable_cameras:
    args_cli.enable_cameras = True
    
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_mul
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
import os

import carb
carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

from task.lift.config.ik_abs_env_cfg_3_1_answer import FrankaTBarPickPlaceEnvCfg
from task.lift.mdp_3_1.terminations import object_pickplace_goal
from isaaclab.envs import ManagerBasedEnv


class GripperState:
    OPEN = 1.0
    CLOSE = -1.0


class PickAndPlaceSmState:
    REST = 0
    PREDICT = 1
    READY = 2
    PREGRASP = 3
    GRASP = 4
    CLOSE = 5
    HOLD = 6          
    LIFT = 7          
    MOVE_TO_BIN = 8   
    LOWER = 9         
    RELEASE = 10      
    BACK = 11         
    BACK_TO_READY = 12 


class PickAndPlaceSmWaitTime:
    REST = 1.0
    PREDICT = 0.0
    READY = 0.5
    PREGRASP = 0.8
    GRASP = 0.3
    CLOSE = 0.8
    HOLD = 0.2       
    LIFT = 0.5
    MOVE_TO_BIN = 0.5
    LOWER = 0.5
    RELEASE = 0.5
    BACK = 0.5
    BACK_TO_READY = 0.5


# 수직 grasp 자세: 그리퍼가 아래를 향함 (qw,qx,qy,qz) = (0,1,0,0)
DOWN_QUAT = [0.0, 1.0, 0.0, 0.0]


class PickAndPlaceSm:
    def __init__(self, dt, num_envs, device="cpu", position_threshold=0.02):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_state = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)

        self.des_ee_pose = torch.zeros((num_envs, 7), device=device)
        self.des_gripper_state = torch.full((num_envs, 1), 0.0, device=device)

        # 준비 자세 (테이블 위에서 대기)
        self.ready_pose = torch.tensor(
            [[0.4, 0.0, 0.5] + DOWN_QUAT], device=device
        ).repeat(num_envs, 1)

        # 바구니 위 / 바구니 안 (환경의 bin 위치 0.2, 0.6 기준)
        self.bin_pose = torch.tensor(
            [[0.2, 0.6, 0.35] + DOWN_QUAT], device=device
        ).repeat(num_envs, 1)
        self.bin_lower_pose = torch.tensor(
            [[0.2, 0.6, 0.15] + DOWN_QUAT], device=device
        ).repeat(num_envs, 1)

        # grasp/pregrasp (PREDICT에서 T_bar 위치로 채워짐)
        self.grasp_pose = torch.zeros((num_envs, 7), device=device)
        self.pregrasp_pose = torch.zeros((num_envs, 7), device=device)
        self.lift_pose = torch.zeros((num_envs, 7), device=device)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndPlaceSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    def set_grasp_from_object(self, env_idx, object_pos_w, object_quat_w, yaw_offset=0.0):
        """T_bar 위치 + 회전(yaw)에 맞춰 수직 grasp pose 설정."""
        x, y, z = object_pos_w[0].item(), object_pos_w[1].item(), object_pos_w[2].item()
        z_base = z - 0.5  # EE 좌표계(base 기준)에 맞춤

        roll, pitch, yaw = euler_xyz_from_quat(object_quat_w.unsqueeze(0))
        yaw = yaw.item() + yaw_offset

        base_down = torch.tensor(DOWN_QUAT, device=self.device).unsqueeze(0)
        yaw_quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device), 
            torch.zeros(1, device=self.device), 
            torch.tensor([yaw], device=self.device)
        )
        grasp_quat = quat_mul(yaw_quat, base_down).squeeze(0)

        gq = grasp_quat.tolist()
        self.grasp_pose[env_idx] = torch.tensor([x, y, z_base + 0.0] + gq, device=self.device)
        self.pregrasp_pose[env_idx] = torch.tensor([x, y, z_base + 0.12] + gq, device=self.device)

    def compute(self, ee_pose):
        ee_pos = ee_pose[:, :3].clone()
        ee_pos[:, 2] -= 0.5  # 로봇 base 높이 보정

        for i in range(self.num_envs):
            state = self.sm_state[i]

            if state == PickAndPlaceSmState.REST:
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.REST:
                    self.sm_state[i] = PickAndPlaceSmState.PREDICT
                    self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREDICT:
                # grasp_pose는 외부(main)에서 set_grasp_from_object로 채워짐
                self.sm_state[i] = PickAndPlaceSmState.PREGRASP
                self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREGRASP:
                self.des_ee_pose[i] = self.pregrasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.PREGRASP:
                        self.sm_state[i] = PickAndPlaceSmState.GRASP
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.GRASP:
                self.des_ee_pose[i] = self.grasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.GRASP:
                        self.sm_state[i] = PickAndPlaceSmState.CLOSE
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.CLOSE:
                self.des_ee_pose[i] = self.grasp_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.CLOSE:
                    self.sm_state[i] = PickAndPlaceSmState.HOLD 
                    self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.HOLD:
                self.des_ee_pose[i] = self.grasp_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.HOLD:
                    self.sm_state[i] = PickAndPlaceSmState.LIFT
                    self.sm_wait_time[i] = 0.0
                    self.lift_pose[i] = self.grasp_pose[i].clone()
                    self.lift_pose[i, 2] += 0.25

            elif state == PickAndPlaceSmState.LIFT:
                self.des_ee_pose[i] = self.lift_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LIFT:
                        self.sm_state[i] = PickAndPlaceSmState.MOVE_TO_BIN
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.MOVE_TO_BIN:
                self.des_ee_pose[i] = self.bin_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.MOVE_TO_BIN:
                        self.sm_state[i] = PickAndPlaceSmState.LOWER
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.LOWER:
                self.des_ee_pose[i] = self.bin_lower_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LOWER:
                        self.sm_state[i] = PickAndPlaceSmState.RELEASE
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.RELEASE:
                self.des_ee_pose[i] = self.bin_lower_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.RELEASE:
                    self.sm_state[i] = PickAndPlaceSmState.BACK
                    self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.BACK:
                self.des_ee_pose[i] = self.bin_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK:
                        self.sm_state[i] = PickAndPlaceSmState.BACK_TO_READY
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.BACK_TO_READY:
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 준비 자세 도달 후 자동으로 다음 에피소드 시작 (REST로 전이)
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    self.sm_state[i] = PickAndPlaceSmState.REST
                    self.sm_wait_time[i] = 0.0

            self.sm_wait_time[i] += self.dt

        return torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)


def main():
    num_envs = args_cli.num_envs
    target_demos = args_cli.num_demos
    dataset_path = args_cli.dataset_file
    max_episode_steps = args_cli.max_steps

    output_dir = os.path.dirname(dataset_path)
    output_file_name = os.path.splitext(os.path.basename(dataset_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = FrankaTBarPickPlaceEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # 성공 판정 term을 분리 (종료는 안 시키되 판정용으로 사용)
    success_term = None
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None:
        success_term = getattr(terminations, "success", None)
        if success_term is not None:
            terminations.success = None
        if hasattr(terminations, "time_out"):
            terminations.time_out = None

    # RecorderManager 설정 - 성공한 에피소드만 HDF5로 저장
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    env.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    device = env.scene.device

    sm = PickAndPlaceSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=num_envs,
        device=device,
        position_threshold=0.02,
    )

    num_success_steps = 10
    # 환경별 성공 카운트 (정수 → 텐서)
    success_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
    # 환경별 경과 스텝 (개별 타임아웃 관리용)
    env_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
    # CLI args 값 사용 (main() 상단에서 이미 할당됨)
    # target_demos = args_cli.num_demos  (238번줄에서 할당)
    # max_episode_steps = args_cli.max_steps  (242번줄에서 할당)
    recorded_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # PREDICT 상태인 환경들의 grasp 목표 설정
            for i in range(num_envs):
                if sm.sm_state[i] == PickAndPlaceSmState.PREDICT:
                    obj = env.scene["object_0"]
                    obj_pos_w = obj.data.root_pos_w[i] - env.scene.env_origins[i]
                    obj_quat_w = obj.data.root_quat_w[i]
                    from isaaclab.utils.math import euler_xyz_from_quat
                    r, p, y = euler_xyz_from_quat(obj_quat_w.unsqueeze(0))
                    import math as m
                    print(f"env{i} 실제 회전: roll={m.degrees(r.item()):.0f} "
                          f"pitch={m.degrees(p.item()):.0f} yaw={m.degrees(y.item()):.0f}")
                    sm.set_grasp_from_object(i, obj_pos_w, obj_quat_w, yaw_offset=-math.pi/2)
                    # 초기 위치(ready_pose) 도달 직후: REST 구간 버퍼를 리셋하여 초기 위치부터 성공까지 저장
                    env.recorder_manager.reset([i])

            ee_frame = env.scene["ee_frame"]
            tcp_pos = ee_frame.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)

            actions = sm.compute(ee_pose)
            env.step(actions)
            env_step_count += 1

            # ── 성공 판정 ──────────────────────────────────────────────────
            done_env_ids = torch.tensor([], dtype=torch.long, device=device)
            if success_term is not None:
                success_flags = success_term.func(env, **success_term.params)  # (num_envs,) bool
            else:
                success_flags = object_pickplace_goal(env)

            # TODO(문제5-a) 순간적인/우연한 성공을 걸러내기 위해 "연속 성공 스텝 수" 를 센다.
            #   성공한 환경은 카운트를 +1, 실패한 환경은 다시 0 으로 되돌려야 한다.
            #   힌트: torch.where(조건, 참일_때_값, 거짓일_때_값)  /  0 으로 되돌릴 때는 torch.zeros_like 사용
            success_step_count = ???

            # 연속 N스텝 성공한 환경 인덱스 추출
            done_env_ids = (success_step_count >= num_success_steps).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.dim() == 0:
                done_env_ids = done_env_ids.unsqueeze(0)  # scalar → 1D

            # 성공한 환경: export 후 리셋 예약
            if len(done_env_ids) > 0:
                ids_list = done_env_ids.tolist()
                # TODO(문제5-b) 성공 에피소드만 HDF5 로 저장하는 3단계 순서다. 아래 (2)의 인자를 채워라.
                #   (1) record_pre_reset      : 리셋 직전까지의 궤적을 버퍼에 확정
                #   (2) set_success_to_episodes: 해당 에피소드들을 "성공" 으로 표시
                #   (3) export_episodes        : 실제 파일로 내보내기
                env.recorder_manager.record_pre_reset(ids_list, force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    ids_list,
                    ???,   # 모양 (len(ids_list), 1), dtype=torch.bool, 값은 모두 성공(True)  / device=env.device
                )
                env.recorder_manager.export_episodes(ids_list)
                recorded_count = env.recorder_manager.exported_successful_episode_count
                print(f">>> {len(ids_list)}개 성공! 누적 demo: {recorded_count}/{target_demos}")

            # ── 타임아웃 판정 (success_term 유무와 무관하게 항상 실행) ───────
            timeout_env_ids = (env_step_count >= max_episode_steps).nonzero(as_tuple=False).squeeze(-1)
            if timeout_env_ids.dim() == 0:
                timeout_env_ids = timeout_env_ids.unsqueeze(0)

            reset_env_ids = torch.cat([done_env_ids, timeout_env_ids]).unique()
            if len(reset_env_ids) > 0:
                print(f"=== RESET 발생 ===")
                print(f"리셋 대상: {reset_env_ids.tolist()}")
                print(f"리셋 전 전체 state: {sm.sm_state.tolist()}")
                env.recorder_manager.reset(reset_env_ids)
                env._reset_idx(reset_env_ids)
                sm.reset_idx(reset_env_ids)
                success_step_count[reset_env_ids] = 0
                env_step_count[reset_env_ids] = 0
                print(f"리셋 후 전체 state: {sm.sm_state.tolist()}")
                print(f"리셋 후 전체 env_step: {env_step_count.tolist()}")

            # 매 100스텝마다 전체 환경 상태 출력 (멈추는지 확인)
            if env_step_count.max().item() % 100 == 0:
                print(f"[모니터] state={sm.sm_state.tolist()} env_step={env_step_count.tolist()}")
                print(f"[모니터] TCP Z pos: {tcp_pos[0, 2].item():.3f}, DES Z pos: {sm.des_ee_pose[0, 2].item():.3f}")

            if recorded_count >= target_demos:
                print(f"=== 목표 {target_demos}개 수집 완료 ===")
                break


if __name__ == "__main__":
    main()
    simulation_app.close()