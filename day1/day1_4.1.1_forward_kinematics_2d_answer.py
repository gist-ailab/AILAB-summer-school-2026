import numpy as np                     # 수치 계산: 삼각함수·배열·행렬 연산
import matplotlib.pyplot as plt        # 그래프 창 생성 및 2D 플롯
from matplotlib.widgets import Slider  # 관절 각도를 실시간으로 조절하는 슬라이더 UI
from viz_utils import draw_frame_2d, draw_rot_arc_2d, draw_base_2d, add_axis_color_note  # 좌표축·회전호·베이스 프레임 등 2D 시각화 헬퍼(직접 만든 모듈)

def forward_kinematics(theta1, theta2, L1=1.0, L2=1.0):
    """
    2링크 평면 로봇 팔의 정기구학(Forward Kinematics).

    관절 각도(theta1, theta2)로부터 각 관절과 End-Effector(EE)의 위치를 구함.
    핵심: theta2는 link1을 기준으로 한 상대 각도이므로,
    EE를 계산할 때 link2의 방향은 두 각을 더한 누적 각도(theta1+theta2)가 됨.
    """
    # 원점 (Joint 1 위치)
    x0, y0 = 0.0, 0.0

    # Joint 2 위치: link1을 theta1 방향으로 뻗은 끝점
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    # End-Effector 위치: link1 끝점에서 link2를 누적각(theta1+theta2) 방향으로 더함
    x2 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y2 = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

    return (x0, y0), (x1, y1), (x2, y2)

def main():
    # 링크 길이 설정
    L1, L2 = 1.0, 1.0

    # 초기 관절 각도 (도)
    init_theta1_deg = 45.0
    init_theta2_deg = 45.0

    # 그림 및 슬라이더 영역 셋업
    fig, ax = plt.subplots(figsize=(7, 7.5))
    plt.subplots_adjust(bottom=0.25)  # 슬라이더 공간 확보
    add_axis_color_note(fig)

    def draw(theta1, theta2):
        """축을 비우고 현재 각도로 로봇 팔 + 관절 좌표축/회전방향을 다시 그림."""
        ax.cla()

        # 현재 각도로 세 점(원점, Joint2, EE) 위치 계산
        p0, p1, p2 = forward_kinematics(theta1, theta2, L1, L2)

        # 로봇 팔 링크
        ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]],
                '-o', linewidth=3, markersize=10, color='blue', label='Robot Arm')
        ax.plot([p2[0]], [p2[1]], 'ro', markersize=12, label='End-Effector')

        # 고정된 base(world) 좌표계: 원점에 있지만 관절과 달리 회전하지 않는 기준 프레임
        draw_base_2d(ax, p0, size=0.5)

        # 각 관절의 로컬 좌표축 (그 관절까지의 누적 회전각으로 방향 결정)
        draw_frame_2d(ax, p0, theta1, size=0.4)            # Joint 1 frame
        draw_frame_2d(ax, p1, theta1 + theta2, size=0.4)   # Joint 2 frame
        draw_frame_2d(ax, p2, theta1 + theta2, size=0.3)   # End-Effector frame

        # 각 관절의 +회전 방향: 출력 링크 방향에서 시작하는 반시계(CCW) 호
        draw_rot_arc_2d(ax, p0, base_angle=theta1, radius=0.3)
        draw_rot_arc_2d(ax, p1, base_angle=theta1 + theta2, radius=0.3)

        # 격자/범위/비율을 고정하고, 제목에 현재 각도와 EE 좌표를 표시
        ax.grid(True)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend(loc='upper right')
        ax.set_title(f"Forward Kinematics (theta1={np.degrees(theta1):.1f}°, "
                     f"theta2={np.degrees(theta2):.1f}°)   EE=({p2[0]:.2f}, {p2[1]:.2f})")

    # 슬라이더 생성 (theta1, theta2)
    ax_t1 = plt.axes([0.20, 0.12, 0.62, 0.03])
    ax_t2 = plt.axes([0.20, 0.06, 0.62, 0.03])
    slider_t1 = Slider(ax_t1, 'theta1 (deg)', -180, 180, valinit=init_theta1_deg)
    slider_t2 = Slider(ax_t2, 'theta2 (deg)', -180, 180, valinit=init_theta2_deg)

    # 슬라이더 값(도)을 라디안으로 바꿔 다시 그리는 콜백
    def update(val=None):
        draw(np.radians(slider_t1.val), np.radians(slider_t2.val))
        fig.canvas.draw_idle()

    slider_t1.on_changed(update)
    slider_t2.on_changed(update)
    update()  # 초기 화면 그리기

    plt.show()

if __name__ == "__main__":
    main()
