"""
2D 역기구학(Inverse Kinematics) 실습.

2링크 평면 로봇에서 목표점 (x, y)에 EE를 보내는 관절 각도를 코사인 법칙으로 구함.
같은 목표점에 대해 보통 Elbow-down / Elbow-up 두 가지 해가 나온다는 점을 확인하고,
슬라이더로 목표점을 옮기며 해가 어떻게 변하는지(또는 작업 영역을 벗어나 사라지는지) 관찰함.
"""
import numpy as np                     # 수치 계산: 삼각함수·배열·행렬 연산
import matplotlib.pyplot as plt        # 그래프 창 생성 및 2D 플롯
from matplotlib.widgets import Slider  # 목표 위치를 실시간으로 조절하는 슬라이더 UI
from viz_utils import draw_frame_2d, draw_rot_arc_2d, draw_base_2d, add_axis_color_note  # 좌표축·회전호·베이스 프레임 등 2D 시각화 헬퍼(직접 만든 모듈)

def inverse_kinematics(x, y, L1=1.0, L2=1.0):
    """
    2링크 평면 로봇의 기하학적 역기구학.

    목표점 (x, y)를 주면 EE(end-effector)를 그 점에 보내는 관절 각도를 구함.
    보통 해가 두 개(Elbow-down / Elbow-up) 존재함.
    """
    # 코사인 제2법칙으로 두 번째 관절 각도의 cos 값을 구함.
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # |cos| > 1 이면 삼각형이 닫히지 않는다 = 목표점이 팔이 닿는 작업 영역 밖 (해 없음).
    if np.abs(cos_theta2) > 1.0:
        return None, None

    # 하나의 cos 값에서 sin 은 +/- 두 부호가 나옴. 이 부호가 곧 두 해(Elbow-down / Elbow-up)를 만듦.
    sin_theta2_sol1 = np.sqrt(1 - cos_theta2**2)
    sin_theta2_sol2 = -np.sqrt(1 - cos_theta2**2)

    # atan2(sin, cos)는 부호까지 살려서 각도를 복원함.
    theta2_sol1 = np.arctan2(sin_theta2_sol1, cos_theta2)
    theta2_sol2 = np.arctan2(sin_theta2_sol2, cos_theta2)

    # theta1 = (목표점 방향각) - (두 번째 링크가 만드는 보정각).
    # theta2 부호가 다르므로 theta1 도 두 해로 갈림.
    theta1_sol1 = np.arctan2(y, x) - np.arctan2(L2 * sin_theta2_sol1, L1 + L2 * cos_theta2)
    theta1_sol2 = np.arctan2(y, x) - np.arctan2(L2 * sin_theta2_sol2, L1 + L2 * cos_theta2)

    return (theta1_sol1, theta2_sol1), (theta1_sol2, theta2_sol2)

def main():
    L1, L2 = 1.0, 1.0
    init_x, init_y = 1.2, 0.5

    # 두 해를 좌/우 subplot 에 나란히 보여줌.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    plt.subplots_adjust(bottom=0.22)
    add_axis_color_note(fig)
    titles = ["Solution 1 (Elbow-Down)", "Solution 2 (Elbow-Up)"]

    def draw_one(ax, sol, title, target):
        """한 subplot 을 비우고 해당 해(팔 + 관절 좌표축/회전방향)를 그림."""
        ax.cla()
        tx, ty = target

        # 해가 있으면 관절 각도로부터 정운동학(FK)을 풀어 각 관절 위치를 얻고 팔을 그림.
        if sol is not None:
            t1, t2 = sol
            p0 = (0.0, 0.0)
            p1 = (L1 * np.cos(t1), L1 * np.sin(t1))
            p2 = (p1[0] + L2 * np.cos(t1 + t2), p1[1] + L2 * np.sin(t1 + t2))

            ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]],
                    '-o', linewidth=3, markersize=8, color='blue', label='Robot Arm')

            # 각 관절의 로컬 좌표축 + 회전(+) 방향 표시
            draw_frame_2d(ax, p0, t1, size=0.4)
            draw_frame_2d(ax, p1, t1 + t2, size=0.4)
            draw_frame_2d(ax, p2, t1 + t2, size=0.3)
            draw_rot_arc_2d(ax, p0, base_angle=t1, radius=0.3)
            draw_rot_arc_2d(ax, p1, base_angle=t1 + t2, radius=0.3)

            ax.set_title(f"{title}\nTh1: {np.degrees(t1):.1f}°, Th2: {np.degrees(t2):.1f}°")
        # 해가 없으면(도달 불가) 안내 문구만 표시.
        else:
            ax.set_title(f"{title}\n[Out of reach]")

        # 고정 base(world) 좌표계
        draw_base_2d(ax, (0.0, 0.0), size=0.5)

        # 목표점(빨간 x)과 격자/범위/비율 설정.
        ax.plot([tx], [ty], 'rx', markersize=12, label='Target Point')
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend(loc='upper right')

    # 슬라이더 생성 (target x, y)
    ax_x = plt.axes([0.20, 0.10, 0.60, 0.03])
    ax_y = plt.axes([0.20, 0.05, 0.60, 0.03])
    slider_x = Slider(ax_x, 'target x', -2.0, 2.0, valinit=init_x)
    slider_y = Slider(ax_y, 'target y', -2.0, 2.0, valinit=init_y)

    # 목표점이 바뀔 때마다 두 해를 다시 계산해 좌/우 subplot에 그림
    def update(val=None):
        tx, ty = slider_x.val, slider_y.val
        sol1, sol2 = inverse_kinematics(tx, ty, L1, L2)
        draw_one(axes[0], sol1, titles[0], (tx, ty))
        draw_one(axes[1], sol2, titles[1], (tx, ty))
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    update()  # 초기 화면 그리기

    plt.suptitle("2D Inverse Kinematics (drag sliders to move the target)")
    plt.show()

if __name__ == "__main__":
    main()
