import numpy as np                       # 수치 계산: 삼각함수·배열·행렬 연산
import matplotlib.pyplot as plt          # 그래프 창 생성 및 3D 플롯
from matplotlib.widgets import Slider    # 관절 각도를 실시간으로 조절하는 슬라이더 UI
from mpl_toolkits.mplot3d import Axes3D  # 3D 축(projection='3d') 지원 활성화
from viz_utils import draw_frame_3d, draw_rot_arc_3d, draw_base_3d, add_axis_color_note  # 좌표축·회전호·베이스 프레임 등 3D 시각화 헬퍼

# z축 기준 회전행렬 (yaw). 반시계 방향이 +.
def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# y축 기준 회전행렬 (pitch).
def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def joint_frames(theta1, theta2, theta3):
    """
    각 관절의 로컬 좌표축(회전행렬)과 회전축을 계산함.
    R01: base yaw 후, R02: shoulder pitch 후, R03: elbow pitch 후의 좌표계.
    """
    # 직전 관절의 좌표계에 회전을 누적해서 각 link의 좌표축을 만듦.
    # pitch에 -theta를 쓰는 이유: 팔을 위로 들 때(+ 각도) z가 올라가도록 부호를 맞추기 위함.
    R01 = _rz(theta1)                 # base yaw (회전축: world z)
    R02 = R01 @ _ry(-theta2)          # shoulder pitch (회전축: R01의 y축)
    R03 = R02 @ _ry(-theta3)          # elbow pitch (회전축: 동일)
    # base yaw로 한 번 돌아간 좌표계의 y축이 두 pitch 관절의 공통 회전축이 됨.
    tangential = R01[:, 1]            # shoulder/elbow pitch 회전축
    axes_dirs = {'yaw': np.array([0, 0, 1.0]), 'pitch': tangential}
    return R01, R02, R03, axes_dirs

def forward_kinematics(theta1, theta2, theta3, L0=0.5, L1=1.0, L2=1.0):
    """
    3자유도 공간(3D) 로봇 팔의 정기구학 (Base Yaw + Shoulder/Elbow Pitch).
    """
    # 원점 (Base Bottom)
    x0, y0, z0 = 0.0, 0.0, 0.0

    # Joint 2 위치 (Shoulder) - 수직 Base 기둥 끝
    x1, y1, z1 = 0.0, 0.0, L0

    # Joint 3 위치 (Elbow) - Upper Arm 끝
    # 먼저 수직 평면에서 팔의 수평 도달거리 r1을 구하고, yaw(theta1)로 XY 평면에 배치함.
    r1 = L1 * np.cos(theta2)
    x2 = r1 * np.cos(theta1)
    y2 = r1 * np.sin(theta1)
    z2 = L0 + L1 * np.sin(theta2)

    # End-Effector 위치 - Forearm 끝
    # 두 link의 수평 성분을 더해 수평거리 r2를 만들고, 높이는 각 link의 수직 성분을 누적함.
    r2 = L1 * np.cos(theta2) + L2 * np.cos(theta2 + theta3)
    x3 = r2 * np.cos(theta1)
    y3 = r2 * np.sin(theta1)
    z3 = L0 + L1 * np.sin(theta2) + L2 * np.sin(theta2 + theta3)

    return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)

def main():
    # 링크 길이 설정 (Base 기둥, Upper Arm, Forearm)
    L0, L1, L2 = 0.5, 1.0, 1.0

    # 초기 관절 각도 (도)
    init = {'t1': 30.0, 't2': 45.0, 't3': 30.0}

    # 그림 셋업
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.22)  # 슬라이더 공간 확보
    add_axis_color_note(fig)

    def draw(theta1, theta2, theta3):
        """축을 비우고 현재 각도로 로봇 팔 + 관절 좌표축/회전축을 다시 그림."""
        ax.cla()

        # 정운동학으로 네 점(원점~EE)의 좌표를 얻음.
        p0, p1, p2, p3 = forward_kinematics(theta1, theta2, theta3, L0, L1, L2)
        x_coords = [p0[0], p1[0], p2[0], p3[0]]
        y_coords = [p0[1], p1[1], p2[1], p3[1]]
        z_coords = [p0[2], p1[2], p2[2], p3[2]]

        # 팔 전체를 선+마커로, EE는 빨간 점으로 강조해 그림.
        ax.plot(x_coords, y_coords, z_coords, '-o', linewidth=4, markersize=10,
                color='blue', label='Robot Arm')
        ax.plot([p3[0]], [p3[1]], [p3[2]], 'ro', markersize=12, label='End-Effector')

        # 고정된 base(world) 좌표계 (base bottom)
        draw_base_3d(ax, p0, size=0.4)

        # 각 관절의 로컬 좌표축 + +회전 방향
        R01, R02, R03, axes_dirs = joint_frames(theta1, theta2, theta3)
        draw_frame_3d(ax, p0, R01, size=0.35)   # Joint 1 (base yaw 후)
        draw_frame_3d(ax, p1, R02, size=0.35)   # Joint 2 (shoulder pitch 후)
        draw_frame_3d(ax, p2, R03, size=0.35)   # Joint 3 (elbow pitch 후)
        draw_frame_3d(ax, p3, R03, size=0.3)    # End-Effector
        draw_rot_arc_3d(ax, p0, axes_dirs['yaw'],   radius=0.3)   # base yaw 회전축 (z)
        draw_rot_arc_3d(ax, p1, axes_dirs['pitch'], radius=0.3)   # shoulder pitch 회전축
        draw_rot_arc_3d(ax, p2, axes_dirs['pitch'], radius=0.3)   # elbow pitch 회전축

        # 관절 이름 라벨
        ax.text(*p1, '  Shoulder', fontsize=8, color='black')
        ax.text(*p2, '  Elbow', fontsize=8, color='black')
        ax.text(*p3, '  EE', fontsize=8, color='black')

        # 보기 범위/축 라벨/시점 등 3D 화면 설정
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_zlim(-0.15, 2.5)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.view_init(elev=25, azim=45)
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_title(f"3D Forward Kinematics\n"
                     f"(th1={np.degrees(theta1):.1f}°, th2={np.degrees(theta2):.1f}°, "
                     f"th3={np.degrees(theta3):.1f}°)  EE=({p3[0]:.2f}, {p3[1]:.2f}, {p3[2]:.2f})")

    # 슬라이더 생성 (theta1, theta2, theta3)
    ax_t1 = plt.axes([0.20, 0.13, 0.62, 0.025])
    ax_t2 = plt.axes([0.20, 0.08, 0.62, 0.025])
    ax_t3 = plt.axes([0.20, 0.03, 0.62, 0.025])
    slider_t1 = Slider(ax_t1, 'theta1 (deg)', -180, 180, valinit=init['t1'])
    slider_t2 = Slider(ax_t2, 'theta2 (deg)', -180, 180, valinit=init['t2'])
    slider_t3 = Slider(ax_t3, 'theta3 (deg)', -180, 180, valinit=init['t3'])

    # 슬라이더 값(도)을 라디안으로 바꿔 다시 그림.
    def update(val=None):
        draw(np.radians(slider_t1.val),
             np.radians(slider_t2.val),
             np.radians(slider_t3.val))
        fig.canvas.draw_idle()

    slider_t1.on_changed(update)
    slider_t2.on_changed(update)
    slider_t3.on_changed(update)
    update()  # 초기 화면 그리기

    plt.show()

if __name__ == "__main__":
    main()
