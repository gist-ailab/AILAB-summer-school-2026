"""
3D 역기구학(Inverse Kinematics) 실습.

3자유도 공간 로봇(base yaw + shoulder/elbow pitch)의 역운동학을 풂.
핵심 아이디어: yaw(theta1)로 방향을 먼저 정하면 나머지는 세로 평면(r, z')상의
2링크 문제로 바뀌고, 여기에 2D와 똑같은 코사인 법칙을 적용함.
elbow 방향에 따라 Elbow-down / Elbow-up 두 해가 나오는 것을 슬라이더로 관찰함.
"""
import numpy as np                       # 수치 계산: 삼각함수·배열·행렬 연산
import matplotlib.pyplot as plt          # 그래프 창 생성 및 3D 플롯
from matplotlib.widgets import Slider    # 목표 위치를 실시간으로 조절하는 슬라이더 UI
from mpl_toolkits.mplot3d import Axes3D  # 3D 축(projection='3d') 지원 활성화
from viz_utils import draw_frame_3d, draw_rot_arc_3d, draw_base_3d, add_axis_color_note  # 좌표축·회전호·베이스 프레임 등 3D 시각화 헬퍼(직접 만든 모듈)

# z축 기준 회전행렬 (yaw).
def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# y축 기준 회전행렬 (pitch).
def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def joint_frames(theta1, theta2, theta3):
    """각 관절의 로컬 좌표축(회전행렬)과 회전축 방향을 계산함 (시각화용)."""
    # base yaw 좌표계 위에 pitch 회전을 차례로 누적함.
    # 두 pitch 관절의 공통 회전축은 R01 의 y축(tangential) 방향임.
    R01 = _rz(theta1)             # base yaw (회전축: world z)
    R02 = R01 @ _ry(-theta2)      # shoulder pitch (회전축: tangential)
    R03 = R02 @ _ry(-theta3)      # elbow pitch (회전축: tangential)
    tangential = R01[:, 1]
    axes_dirs = {'yaw': np.array([0, 0, 1.0]), 'pitch': tangential}
    return R01, R02, R03, axes_dirs

def inverse_kinematics(x, y, z, L0=0.5, L1=1.0, L2=1.0):
    """
    3자유도 공간 로봇의 기하학적 역기구학.
    """
    # theta1: base yaw. 목표를 XY 평면에서 바라본 방향각으로 정함.
    # 목표가 z축 바로 위(x=y=0)면 방향이 정의되지 않으므로 0 으로 둠.
    if np.allclose(x, 0.0) and np.allclose(y, 0.0):
        theta1 = 0.0
    else:
        theta1 = np.arctan2(y, x)

    # yaw 로 방향을 분리하고 나면, XY 투영거리 r 과 shoulder 기준 높이 z' 가 이루는
    # 세로 평면(r, z')상의 2링크 문제로 단순해짐.
    r = np.sqrt(x**2 + y**2)
    z_prime = z - L0

    # (r, z') 평면에서 코사인 제2법칙으로 elbow 각도의 cos 값을 구함.
    cos_theta3 = (r**2 + z_prime**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # |cos| > 1 이면 삼각형이 닫히지 않는다 = 목표점이 작업 영역 밖 (해 없음).
    if np.abs(cos_theta3) > 1.0:
        return None, None

    # sin 의 두 부호(+/-)가 elbow 의 꺾이는 방향을 결정 → Elbow-down / Elbow-up 두 해.
    sin_theta3_sol1 = np.sqrt(1 - cos_theta3**2)
    sin_theta3_sol2 = -np.sqrt(1 - cos_theta3**2)

    theta3_sol1 = np.arctan2(sin_theta3_sol1, cos_theta3)
    theta3_sol2 = np.arctan2(sin_theta3_sol2, cos_theta3)

    # theta2 = (목표 방향각) - (elbow 때문에 꺾인 보정각). theta3 부호에 따라 두 해로 갈림.
    theta2_sol1 = np.arctan2(z_prime, r) - np.arctan2(L2 * sin_theta3_sol1, L1 + L2 * cos_theta3)
    theta2_sol2 = np.arctan2(z_prime, r) - np.arctan2(L2 * sin_theta3_sol2, L1 + L2 * cos_theta3)

    return (theta1, theta2_sol1, theta3_sol1), (theta1, theta2_sol2, theta3_sol2)

def main():
    L0, L1, L2 = 0.5, 1.0, 1.0
    init = {'x': 1.0, 'y': 0.6, 'z': 1.2}

    # 두 해를 좌/우 3D subplot 에 나란히 보여줌.
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axes = [ax1, ax2]
    plt.subplots_adjust(bottom=0.20)
    add_axis_color_note(fig)
    titles = ["Solution 1 (Elbow-Down)", "Solution 2 (Elbow-Up)"]

    def draw_one(ax, sol, title, target):
        """한 subplot 을 비우고 해당 해를 다시 그림."""
        ax.cla()
        tx, ty, tz = target

        # 해가 존재할 때만 팔을 그림 (None 이면 도달 불가).
        if sol is not None:
            # 역운동학으로 구한 각도를 정운동학(FK)으로 다시 풀어 각 관절 좌표를 복원함.
            t1, t2, t3 = sol
            p0 = (0.0, 0.0, 0.0)
            p1 = (0.0, 0.0, L0)
            r1 = L1 * np.cos(t2)
            p2 = (r1 * np.cos(t1), r1 * np.sin(t1), L0 + L1 * np.sin(t2))
            r2 = L1 * np.cos(t2) + L2 * np.cos(t2 + t3)
            p3 = (r2 * np.cos(t1), r2 * np.sin(t1), L0 + L1 * np.sin(t2) + L2 * np.sin(t2 + t3))

            x_coords = [p0[0], p1[0], p2[0], p3[0]]
            y_coords = [p0[1], p1[1], p2[1], p3[1]]
            z_coords = [p0[2], p1[2], p2[2], p3[2]]
            ax.plot(x_coords, y_coords, z_coords, '-o', linewidth=3, markersize=8, label='Robot Arm')

            # 고정 base(world) 좌표계
            draw_base_3d(ax, p0, size=0.35)

            # 각 관절의 로컬 좌표축 + 회전(+) 방향 표시
            R01, R02, R03, axes_dirs = joint_frames(t1, t2, t3)
            draw_frame_3d(ax, p0, R01, size=0.3)
            draw_frame_3d(ax, p1, R02, size=0.3)
            draw_frame_3d(ax, p2, R03, size=0.3)
            draw_frame_3d(ax, p3, R03, size=0.25)
            draw_rot_arc_3d(ax, p0, axes_dirs['yaw'],   radius=0.28)
            draw_rot_arc_3d(ax, p1, axes_dirs['pitch'], radius=0.28)
            draw_rot_arc_3d(ax, p2, axes_dirs['pitch'], radius=0.28)
            ax.text(*p1, '  Shoulder', fontsize=8, color='black')
            ax.text(*p2, '  Elbow', fontsize=8, color='black')

            ax.set_title(f"{title}\nTh1: {np.degrees(t1):.1f}°, Th2: {np.degrees(t2):.1f}°, Th3: {np.degrees(t3):.1f}°")
        else:
            # 도달 불가한 목표점이면 팔 대신 안내 문구만 표시.
            ax.set_title(f"{title}\n[Out of reach]")

        # 목표점은 항상 빨간 x 마커로 표시 (해 유무와 무관).
        ax.plot([tx], [ty], [tz], 'rx', markersize=12, label='Target Point')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-0.15, 2.5)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.view_init(elev=20, azim=45)
        ax.grid(True)
        ax.legend(loc='upper left')

    # 슬라이더 생성 (target x, y, z)
    ax_x = plt.axes([0.25, 0.11, 0.55, 0.025])
    ax_y = plt.axes([0.25, 0.07, 0.55, 0.025])
    ax_z = plt.axes([0.25, 0.03, 0.55, 0.025])
    slider_x = Slider(ax_x, 'target x', -2.0, 2.0, valinit=init['x'])
    slider_y = Slider(ax_y, 'target y', -2.0, 2.0, valinit=init['y'])
    slider_z = Slider(ax_z, 'target z', 0.0, 2.5, valinit=init['z'])

    # 목표점이 바뀌면 두 해를 다시 계산해 좌/우 subplot 에 각각 그림.
    def update(val=None):
        tx, ty, tz = slider_x.val, slider_y.val, slider_z.val
        sol1, sol2 = inverse_kinematics(tx, ty, tz, L0, L1, L2)
        draw_one(ax1, sol1, titles[0], (tx, ty, tz))
        draw_one(ax2, sol2, titles[1], (tx, ty, tz))
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    update()  # 초기 화면 그리기

    plt.suptitle("3D Inverse Kinematics Solutions (drag sliders to move the target)")
    plt.show()

if __name__ == "__main__":
    main()
