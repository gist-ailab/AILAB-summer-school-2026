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
    # 목표 EE 회전행렬(R_EE)과 위치(p_EE)를 준비함.
    R_EE = rpy_to_rotation_matrix(target_rpy[0], target_rpy[1], target_rpy[2])
    p_EE = np.array(target_pos)

    solutions = []

    # 1. 손목 중심(wrist) 위치 p05 계산.
    # EE 에서 마지막 링크 길이 d6 만큼 EE 의 z축 방향을 거슬러 올라간 점임.
    p05 = p_EE - d6 * R_EE[:, 2]
    x5, y5, z5 = p05[0], p05[1], p05[2]

    # 2. theta1 (어깨 회전) 계산 → 2개 해.
    # p05 의 xy평면 거리가 오프셋 d4 보다 작으면 손목이 도달할 수 없음.
    R1 = np.sqrt(x5**2 + y5**2)
    if R1 < np.abs(d4):
        return []  # 도달 불가능 영역

    # 같은 손목 위치를 앞/뒤 어깨 두 자세로 만들 수 있어 theta1 이 2가지로 갈림.
    phi = np.arctan2(y5, x5)
    alpha = np.arcsin(d4 / R1)

    t1_sols = [
        wrap_angle(phi + alpha),
        wrap_angle(phi + np.pi - alpha)
    ]

    for t1 in t1_sols:
        # 3. theta5 (손목 굽힘) 계산 → theta1 하나당 2개 해.
        # arccos 의 +/- 부호가 손목이 위로/아래로 꺾이는 두 자세를 만듦.
        val = (p_EE[0] * np.sin(t1) - p_EE[1] * np.cos(t1) - d4) / d6
        val = np.clip(val, -1.0, 1.0)

        t5_sols = [
            np.arccos(val),
            -np.arccos(val)
        ]

        for t5 in t5_sols:
            # 4. theta6 (손목 회전) 계산 (theta1, theta5 로부터).
            # frame1 기준으로 본 EE 자세 R16 에서 마지막 손목 회전 성분을 뽑아냄.
            T01 = dh_matrix(t1, d1, 0, np.pi/2)
            R01 = T01[:3, :3]
            R16 = R01.T @ R_EE

            # theta5 가 0 에 가까우면 손목 특이점이라 theta6 가 정해지지 않음 → 0 으로 고정.
            sin_t5 = np.sin(t5)
            if np.abs(sin_t5) > 1e-4:
                t6 = np.arctan2(-R16[2, 1] / sin_t5, R16[2, 0] / sin_t5)
            else:
                t6 = 0.0  # 특이점 (Singularity)

            t6 = wrap_angle(t6)

            # 5. T14 구하기.
            # 이미 구한 손목 변환(T46)을 전체 변환에서 떼어내면 어깨~팔꿈치 구간 변환 T14 만 남음.
            T45 = dh_matrix(t5, d5, 0, -np.pi/2)
            T56 = dh_matrix(t6, d6, 0, 0)
            T46 = T45 @ T56

            T06 = np.eye(4)
            T06[:3, :3] = R_EE
            T06[:3, 3] = p_EE

            T14 = np.linalg.inv(T01) @ T06 @ np.linalg.inv(T46)
            p14 = T14[:3, 3]
            R14 = T14[:3, :3]

            X, Y = p14[0], p14[1]

            # 6. theta3 (팔꿈치) 계산 → 2개 해.
            # 상완(a2)·전완(a3) 2링크에 코사인 법칙을 적용. arccos 부호가 팔꿈치 위/아래 두 해.
            cos_t3 = (X**2 + Y**2 - a2**2 - a3**2) / (2 * a2 * a3)
            if np.abs(cos_t3) > 1.0:
                continue

            t3_sols = [
                np.arccos(cos_t3),
                -np.arccos(cos_t3)
            ]

            for t3 in t3_sols:
                # 7. theta2, theta4 계산.
                # theta2: 앞서와 같은 평면 2링크 기하(방향각 - 보정각)로 결정.
                t2 = np.arctan2(Y, X) - np.arctan2(a3 * np.sin(t3), a2 + a3 * np.cos(t3))

                # theta4: R14 에서 얻은 합(theta2+theta3+theta4)에서 theta2, theta3 을 빼서 구함.
                t234 = np.arctan2(R14[1, 0], R14[0, 0])
                t4 = t234 - t2 - t3

                # 6개 관절각을 [-pi, pi] 범위로 정리해 하나의 해로 저장.
                solutions.append((
                    wrap_angle(t1),
                    wrap_angle(t2),
                    wrap_angle(t3),
                    wrap_angle(t4),
                    wrap_angle(t5),
                    wrap_angle(t6)
                ))

    # theta1(2) × theta5(2) × theta3(2) 조합이 모여 최대 8개의 해가 됨.
    return solutions

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
