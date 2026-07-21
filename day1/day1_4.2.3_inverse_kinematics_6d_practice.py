"""
6축 로봇(UR5) 해석적 역기구학 실습.

목표 EE 위치와 자세(roll/pitch/yaw)가 주어지면 6개 관절각을 닫힌 형태(수식)로 풂.
DH 파라미터로 정운동학을 세우고, 역으로 theta1 → theta5 → theta6 → theta3 → theta2/theta4
순서로 각도를 하나씩 분리해 나감.
각 단계에서 부호나 arccos 선택으로 2갈래씩 갈라져(theta1×theta5×theta3) 최대 8개의 해가 나옴.
구한 해를 다시 FK로 풀어 목표와의 위치 오차가 거의 0임을 확인함.
"""
import numpy as np                       # 수치 계산: 삼각함수·배열·행렬 연산
import matplotlib.pyplot as plt          # 그래프 창 생성 및 3D 플롯
from matplotlib.widgets import Slider    # 목표 위치/자세를 실시간으로 조절하는 슬라이더 UI
from mpl_toolkits.mplot3d import Axes3D  # 3D 축(projection='3d') 지원 활성화
from viz_utils import draw_frame_3d, draw_base_3d, add_axis_color_note  # 좌표축·베이스 프레임 등 3D 시각화 헬퍼(직접 만든 모듈)

# UR5 공식 DH 파라미터 (링크 길이/오프셋 상수)
d1 = 0.089159
a2 = -0.425
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

# 6개 관절 각각의 DH 파라미터 (theta 는 매번 바뀌는 값이라 여기엔 넣지 않음)
UR5_PARAMS = [
    {'d': d1, 'a': 0,      'alpha': np.pi/2},
    {'d': 0,  'a': a2,     'alpha': 0},
    {'d': 0,  'a': a3,     'alpha': 0},
    {'d': d4, 'a': 0,      'alpha': np.pi/2},
    {'d': d5, 'a': 0,      'alpha': -np.pi/2},
    {'d': d6, 'a': 0,      'alpha': 0}
]

def dh_matrix(theta, d, a, alpha):
    # 표준 DH 공식: Rz(theta)·Tz(d)·Tx(a)·Rx(alpha) 를 하나의 4x4 행렬로 전개
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_t, -sin_t * cos_a,  sin_t * sin_a, a * cos_t],
        [sin_t,  cos_t * cos_a, -cos_t * sin_a, a * sin_t],
        [0,      sin_a,          cos_a,         d],
        [0,      0,              0,             1]
    ])

def forward_kinematics_ur5(thetas):
    """
    UR5 정운동학. 하나의 DH 누적 루프에서 세 가지를 함께 반환함.
    - joints_pos : Base~EE 각 관절 원점 위치 (7개)
    - R_EE       : End-Effector 자세(회전행렬)
    - frames     : 각 관절까지의 누적 4x4 변환 (7개, frames[0]=base)
    joints_pos·R_EE 는 모두 frames 에서 유도되므로 루프는 한 번만 돈다.
    """
    # Base 에서 출발해 관절마다 DH 변환행렬을 곱해 누적하며 각 관절까지의 프레임을 기록함.
    T = np.eye(4)
    frames = [T.copy()]  # frames[0] = base(world) 프레임

    for i in range(6):
        param = UR5_PARAMS[i]
        A = dh_matrix(thetas[i], param['d'], param['a'], param['alpha'])
        T = T @ A
        frames.append(T.copy())

    joints_pos = [f[:3, 3].copy() for f in frames]  # 각 프레임의 translation = 관절 위치
    R_EE = frames[-1][:3, :3]                        # 마지막 프레임의 회전 = EE 자세
    return joints_pos, R_EE, frames

# x축 기준 회전행렬
def rot_x(theta):
    return np.array([
        [1, 0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

# y축 기준 회전행렬
def rot_y(theta):
    return np.array([
        [np.cos(theta),  0, -np.sin(theta)],
        [0,              1, 0],
        [np.sin(theta),  0, np.cos(theta)]
    ])

# z축 기준 회전행렬
def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

# roll-pitch-yaw(고정축 X-Y-Z) 를 하나의 회전행렬로 합성
def rpy_to_rotation_matrix(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def wrap_angle(angle):
    """ 각도를 [-pi, pi] 범위로 래핑 """
    return np.arctan2(np.sin(angle), np.cos(angle))

def inverse_kinematics_ur5(target_pos, target_rpy):
    """
    UR5 로봇의 해석적(닫힌 형태) 역운동학. 목표 pose 에 대해 최대 8개의 해를 도출함.

    풀이 흐름: theta1 → theta5 → theta6 → theta3 → theta2/theta4.
    theta1, theta5, theta3 단계에서 각각 2갈래로 갈라져 총 2×2×2 = 최대 8개 해가 생김.
    """
    # ──────────────────────────────────────────────────────────
    # [실습] UR5 해석적(닫힌 형태) 역기구학을 구현하세요. 목표 pose → 최대 8개 해.
    #   (구체 수식은 answer 파일 참고 — 여기서는 풀이 '과정'만 안내)
    #
    #   풀이 흐름:  theta1 → theta5 → theta6 → theta3 → theta2/theta4
    #   theta1·theta5·theta3 이 각각 2갈래로 갈려 최대 2×2×2 = 8해.
    #
    #   1) 손목 중심(p05): EE 에서 마지막 링크(d6)만큼 되짚어 손목 위치를 먼저 구함.
    #   2) theta1(어깨 회전, 2해): p05 를 xy평면에 투영해 어깨 각을 정함. (도달 불가면 return [])
    #   3) theta5(손목 굽힘, 2해): 목표 자세로부터 손목 굽힘각을 정함.
    #   4) theta6(손목 회전): frame1 기준 EE 자세에서 남은 손목 회전을 뽑아냄. (손목 특이점 주의)
    #   5) T14 분리: 이미 구한 손목 변환을 전체에서 떼어내 어깨~팔꿈치 구간만 남김.
    #   6) theta3(팔꿈치, 2해): 상완·전완 2링크에 코사인 법칙 적용.
    #   7) theta2·theta4: 평면 2링크 기하로 theta2, 자세 합에서 theta4 를 구함.
    #   → 각 관절각을 wrap_angle 로 정리해 solutions 에 모아 반환.
    # ──────────────────────────────────────────────────────────
    raise NotImplementedError("TODO: inverse_kinematics_ur5 를 구현하세요")

def main():
    # 초기 목표 EE Pose (UR5 작업 공간 내부의 임의 지점)
    init = {'x': 0.35, 'y': 0.25, 'z': 0.45,
            'roll': 20.0, 'pitch': 45.0, 'yaw': 30.0}

    # 최대 8개의 해를 2행 4열 격자에 하나씩 그림.
    fig = plt.figure(figsize=(20, 11))
    plt.subplots_adjust(left=0.04, right=0.97, bottom=0.18, top=0.92, wspace=0.1, hspace=0.15)
    add_axis_color_note(fig)
    grid_axes = [fig.add_subplot(2, 4, k + 1, projection='3d') for k in range(8)]

    def draw_solution(ax, thetas, idx, target_pos):
        """한 칸을 비우고 하나의 IK 해(있으면)를 그림."""
        ax.cla()

        if thetas is not None:
            # 구한 IK 해를 다시 FK 로 풀어 실제 EE 위치를 얻고, 목표와의 위치 오차를 확인함.
            joints_pos, R_EE_calc, frames = forward_kinematics_ur5(thetas)
            p_EE_calc = joints_pos[-1]
            pos_error = np.linalg.norm(p_EE_calc - np.array(target_pos))

            x_coords = [p[0] for p in joints_pos]
            y_coords = [p[1] for p in joints_pos]
            z_coords = [p[2] for p in joints_pos]
            ax.plot(x_coords, y_coords, z_coords, '-o', linewidth=3, markersize=5, label='Robot Arm')

            # 고정 base(world) 좌표계 (패널이 작아 축 라벨은 생략)
            draw_base_3d(ax, (0.0, 0.0, 0.0), size=0.07, label=False)

            # 각 관절의 로컬 좌표축(triad) — 파란 z축이 곧 그 관절의 회전축 방향임.
            # (frame 6 = End-Effector 는 아래에서 EE 좌표축으로 따로 그리므로 여기선 제외)
            for j in range(1, 6):
                draw_frame_3d(ax, frames[j][:3, 3], frames[j][:3, :3], size=0.05, lw=1.2)

            # End-Effector 로컬 프레임 (X: Red, Y: Green, Z: Blue)
            axis_len = 0.1
            u_x = R_EE_calc @ np.array([1.0, 0.0, 0.0])
            u_y = R_EE_calc @ np.array([0.0, 1.0, 0.0])
            u_z = R_EE_calc @ np.array([0.0, 0.0, 1.0])
            ax.quiver(p_EE_calc[0], p_EE_calc[1], p_EE_calc[2], u_x[0], u_x[1], u_x[2], color='red',   length=axis_len, linewidth=1.5)
            ax.quiver(p_EE_calc[0], p_EE_calc[1], p_EE_calc[2], u_y[0], u_y[1], u_y[2], color='green', length=axis_len, linewidth=1.5)
            ax.quiver(p_EE_calc[0], p_EE_calc[1], p_EE_calc[2], u_z[0], u_z[1], u_z[2], color='blue',  length=axis_len, linewidth=1.5)

            # 목표점(빨간 x)을 찍고, 위치 오차를 제목에 표시함.
            ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'rx', markersize=12, label='Target')
            ax.set_title(f"Solution {idx + 1}  (err={pos_error:.1e} m)")
        else:
            # 빈 칸: 목표점만 옅게 표시
            ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'rx', markersize=10)
            ax.set_title("—")

        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.12, 1.0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
        ax.grid(True)

    # 슬라이더 6개 (위치 3 + 자세 3) - 하단 2열 배치
    sld_specs = [
        ('x',     'target x (m)',  -0.85, 0.85),
        ('y',     'target y (m)',  -0.85, 0.85),
        ('z',     'target z (m)',   0.0,  1.0),
        ('roll',  'roll (deg)',    -180,  180),
        ('pitch', 'pitch (deg)',   -180,  180),
        ('yaw',   'yaw (deg)',     -180,  180),
    ]
    sliders = {}
    for i, (key, label, vmin, vmax) in enumerate(sld_specs):
        col = i // 3          # 0: 왼쪽 열(위치), 1: 오른쪽 열(자세)
        row = i % 3
        left = 0.10 + col * 0.45
        bottom = 0.10 - row * 0.035
        ax_s = plt.axes([left, bottom, 0.30, 0.022])
        sliders[key] = Slider(ax_s, label, vmin, vmax, valinit=init[key])

    def update(val=None):
        # 슬라이더에서 목표 위치/자세를 읽어옴 (각도는 라디안으로 변환)
        target_pos = [sliders['x'].val, sliders['y'].val, sliders['z'].val]
        target_rpy = [np.radians(sliders['roll'].val),
                      np.radians(sliders['pitch'].val),
                      np.radians(sliders['yaw'].val)]

        # 역운동학을 풀어 해들을 구하고 8칸에 채움 (해가 부족하면 나머지 칸은 빈 칸).
        sols = inverse_kinematics_ur5(target_pos, target_rpy)

        for k in range(8):
            thetas = sols[k] if k < len(sols) else None
            draw_solution(grid_axes[k], thetas, k, target_pos)

        fig.suptitle(f"UR5 Analytical Inverse Kinematics — {len(sols)} solution(s)   "
                     f"[pos=({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}), "
                     f"rpy=({sliders['roll'].val:.0f}°, {sliders['pitch'].val:.0f}°, {sliders['yaw'].val:.0f}°)]",
                     fontsize=14)
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    update()  # 초기 화면 그리기

    plt.show()

if __name__ == "__main__":
    main()
