import numpy as np                       # 수치 계산: 삼각함수·배열·행렬 연산
import matplotlib.pyplot as plt          # 그래프 창 생성 및 3D 플롯
from matplotlib.widgets import Slider    # 6개 관절 각도를 실시간으로 조절하는 슬라이더 UI
from mpl_toolkits.mplot3d import Axes3D  # 3D 축(projection='3d') 지원 활성화
from viz_utils import draw_frame_3d, draw_rot_arc_3d, draw_base_3d, add_axis_color_note  # 좌표축·회전호·베이스 프레임 등 3D 시각화 헬퍼(직접 만든 모듈)

# UR5 DH 파라미터 (d, a, alpha)
d1 = 0.089159
a2 = -0.425
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

# 6개 관절 각각의 DH 파라미터 (theta는 슬라이더로 바꾸는 가변값이라 여기엔 없음)
UR5_PARAMS = [
    {'d': d1, 'a': 0,      'alpha': np.pi/2},
    {'d': 0,  'a': a2,     'alpha': 0},
    {'d': 0,  'a': a3,     'alpha': 0},
    {'d': d4, 'a': 0,      'alpha': np.pi/2},
    {'d': d5, 'a': 0,      'alpha': -np.pi/2},
    {'d': d6, 'a': 0,      'alpha': 0}
]

def dh_matrix(theta, d, a, alpha):
    """
    Denavit-Hartenberg 변환 행렬 (한 관절이 다음 관절 좌표계로 넘어가는 4x4 변환).
    """
    # 표준 DH 공식: Rz(theta)·Tz(d)·Tx(a)·Rx(alpha) 를 하나의 4x4 행렬로 전개한 것
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
    UR5의 정기구학. 하나의 DH 누적 루프에서 세 가지를 함께 반환함.
    - joints_pos : Base~EE 각 관절 원점 위치 (7개)                 ← 6D 중 '위치'
    - R_EE       : End-Effector 자세(회전행렬)                     ← 6D 중 '자세'
    - frames     : 각 관절까지의 누적 4x4 변환 (7개, frames[0]=base) ← 좌표축/회전축 시각화용
    joints_pos·R_EE 는 모두 frames 에서 유도되므로 루프는 한 번만 돈다.
    DH 규칙상 joint i는 frames[i-1]의 z축(파란축)을 회전축으로 삼음.
    """
    # Base(world)에서 출발해 관절 변환을 하나씩 곱해가며 누적 변환 T를 갱신함.
    T = np.eye(4)
    frames = [T.copy()]  # frames[0] = base(world) 프레임

    # 각 관절의 DH 행렬을 차례로 곱하고, 그때마다 누적 프레임을 기록
    for i in range(6):
        param = UR5_PARAMS[i]
        A = dh_matrix(thetas[i], param['d'], param['a'], param['alpha'])
        T = T @ A
        frames.append(T.copy())

    joints_pos = [f[:3, 3].copy() for f in frames]  # 각 프레임의 translation = 관절 위치
    R_EE = frames[-1][:3, :3]                        # 마지막 프레임의 회전 = EE 자세
    return joints_pos, R_EE, frames

def main():
    # 초기 관절 각도 (도) - 각 관절이 확연히 꺾인 형태
    init_deg = [30, -60, 80, -50, 90, 20]

    # 그림 셋업
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.05, right=0.62, bottom=0.05, top=0.95)  # 오른쪽에 슬라이더 공간
    add_axis_color_note(fig)

    def draw(thetas):
        """축을 비우고 현재 6개 관절 각도로 UR5를 다시 그림."""
        ax.cla()

        # 정운동학으로 관절 위치들·EE 자세·각 관절 프레임을 한 번에 계산
        joints_pos, R_EE, frames = forward_kinematics_ur5(thetas)
        p_EE = joints_pos[-1]

        x_coords = [p[0] for p in joints_pos]
        y_coords = [p[1] for p in joints_pos]
        z_coords = [p[2] for p in joints_pos]

        # 관절들을 잇는 링크(파란 선)와 관절 마커
        ax.plot(x_coords, y_coords, z_coords, '-o', linewidth=4, markersize=8,
                color='blue', label='UR5 Links & Joints')

        # 각 관절(=DH 프레임 원점)의 이름: 0=Base, 1=Shoulder, ... 6=End-Effector
        joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3', 'End-Effector']
        for i, p in enumerate(joints_pos):
            if i == 0:
                continue  # Base는 아래 draw_base_3d로 따로 표현
            elif i == 6:
                ax.plot([p[0]], [p[1]], [p[2]], 'ro', markersize=10, label='End-Effector')
            else:
                ax.plot([p[0]], [p[1]], [p[2]], 'go', markersize=7,
                        label='Joints' if i == 1 else "")
            ax.text(p[0], p[1], p[2] + 0.03, joint_names[i], fontsize=8, color='black')

        # 고정된 base(world) 좌표계
        draw_base_3d(ax, joints_pos[0], size=0.1)
        ax.text(joints_pos[0][0], joints_pos[0][1], joints_pos[0][2] - 0.05, 'Base', fontsize=8, color='black')

        # 각 관절의 로컬 좌표축(triad): frame 1~5 표시
        # (frame 6 = End-Effector는 아래에서 EE 좌표축으로 따로 그리므로 여기선 제외)
        for i in range(1, 6):
            draw_frame_3d(ax, frames[i][:3, 3], frames[i][:3, :3], size=0.08)

        # 회전축 표시: joint i는 frame(i-1)의 z축을 중심으로 회전함
        for i in range(1, 7):
            origin = frames[i - 1][:3, 3]
            rot_axis = frames[i - 1][:3, 2]
            draw_rot_arc_3d(ax, origin, rot_axis, radius=0.06)

        # End-Effector 좌표축 (X: Red, Y: Green, Z: Blue)
        # EE 회전행렬 R_EE의 각 열이 곧 X/Y/Z 축의 방향벡터
        axis_len = 0.1
        u_x = R_EE @ np.array([1.0, 0.0, 0.0])
        u_y = R_EE @ np.array([0.0, 1.0, 0.0])
        u_z = R_EE @ np.array([0.0, 0.0, 1.0])
        ax.quiver(p_EE[0], p_EE[1], p_EE[2], u_x[0], u_x[1], u_x[2], color='red',   length=axis_len, linewidth=2)
        ax.quiver(p_EE[0], p_EE[1], p_EE[2], u_y[0], u_y[1], u_y[2], color='green', length=axis_len, linewidth=2)
        ax.quiver(p_EE[0], p_EE[1], p_EE[2], u_z[0], u_z[1], u_z[2], color='blue',  length=axis_len, linewidth=2)

        # 보기 범위/라벨/시점 설정
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.05, 1.0)
        ax.set_xlabel('X Axis (m)')
        ax.set_ylabel('Y Axis (m)')
        ax.set_zlabel('Z Axis (m)')
        ax.view_init(elev=25, azim=45)
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_title(f"UR5 Forward Kinematics\n"
                     f"EE = ({p_EE[0]:.3f}, {p_EE[1]:.3f}, {p_EE[2]:.3f}) m")

    # 6개 관절 슬라이더 생성 (오른쪽 세로 배치)
    sliders = []
    labels = ['J1 (Base)', 'J2 (Shoulder)', 'J3 (Elbow)', 'J4 (Wrist1)', 'J5 (Wrist2)', 'J6 (Wrist3)']
    for i in range(6):
        ax_s = plt.axes([0.72, 0.82 - i * 0.13, 0.22, 0.03])
        s = Slider(ax_s, labels[i], -180, 180, valinit=init_deg[i])
        sliders.append(s)

    def update(val=None):
        # 슬라이더 값(도)을 라디안으로 바꿔 로봇을 다시 그림
        thetas = [np.radians(s.val) for s in sliders]
        draw(thetas)
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)
    update()  # 초기 화면 그리기

    plt.show()

if __name__ == "__main__":
    main()
