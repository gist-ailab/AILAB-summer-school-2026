"""
공용 시각화 헬퍼 (FK / IK 실습용)

FK/IK 실습(1.x, 2.x)에서 로봇의 관절·좌표계를 matplotlib 으로 그리는 도구 모음.
- 각 joint 에 로컬 좌표축(triad: x=red, y=green, z=blue) 표시
- 각 joint 의 회전 방향(positive rotation)을 곡선 화살표로 표시
- 고정된 base(world) 좌표계를 받침대/바닥판과 점선 축으로 표시

핵심 개념:
- 회전행렬 R 의 각 열이 곧 그 좌표계의 x/y/z 축 방향 벡터임.
- 회전 호는 회전축에 수직인 평면 위에 그려야 하므로, 축에 수직인 정규직교 기저를 만듦.
- 고정 world 축은 점선, 움직이는 joint 축은 실선으로 구분함.

좌표축 색 규칙:  X = Red,  Y = Green,  Z = Blue
"""
import numpy as np


# ============================================================
#  2D 헬퍼 (회전축은 화면 밖 +Z 방향)
# ============================================================
def draw_frame_2d(ax, origin, angle, size=0.3, alpha=1.0):
    """origin 에 angle(rad)만큼 회전된 로컬 x(red)/y(green) 좌표축을 그림.

    size 는 축 화살표 길이. 2D 회전은 (cos, sin) 로 x축 방향을 정하고,
    y축은 그에 수직인 방향으로 자동 결정됨.
    """
    ox, oy = origin
    # 회전각의 cos/sin 이 곧 회전된 x축의 성분이 됨.
    c, s = np.cos(angle), np.sin(angle)
    # x축(red): 회전된 x축 방향 (c, s)
    ax.quiver(ox, oy, c * size, s * size, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, alpha=alpha, zorder=6)
    # y축(green): x축에 수직인 방향 (-s, c)
    ax.quiver(ox, oy, -s * size, c * size, angles='xy', scale_units='xy', scale=1,
              color='green', width=0.01, alpha=alpha, zorder=6)


def draw_rot_arc_2d(ax, center, base_angle=0.0, radius=0.32, color='purple', sweep_deg=120):
    """center 주위로 +방향(CCW, 회전축 +Z) 회전 화살표 호를 그림.

    관절이 어느 방향으로 도는지를 눈으로 보여주기 위한 호.
    radius 는 호의 반지름, sweep_deg 는 호가 차지하는 각도.
    """
    # base_angle 부터 sweep_deg 만큼의 각도를 샘플링해 호 위의 점들을 만듦.
    sweep = np.radians(sweep_deg)
    th = np.linspace(base_angle, base_angle + sweep, 40)
    xs = center[0] + radius * np.cos(th)
    ys = center[1] + radius * np.sin(th)
    ax.plot(xs, ys, color=color, lw=1.6, alpha=0.8, zorder=5)
    # 호 끝점에 화살촉(접선 방향)을 붙여 회전 방향을 표시
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6), zorder=5)


# ============================================================
#  3D 헬퍼
# ============================================================
def draw_frame_3d(ax, origin, R, size=0.15, alpha=1.0, lw=2):
    """origin 에 회전행렬 R 로 정의된 로컬 좌표축(x=red, y=green, z=blue)을 그림.

    size 는 축 화살표 길이. R 의 각 열이 곧 x/y/z 축 방향이라는 점이 핵심임.
    """
    ox, oy, oz = origin
    colors = ['red', 'green', 'blue']
    # R 의 각 열(0,1,2) = 로컬 x/y/z 축 방향 벡터. size 만큼 늘려 화살표로 그림.
    for col, c in zip(range(3), colors):
        v = R[:, col] * size
        ax.quiver(ox, oy, oz, v[0], v[1], v[2],
                  color=c, linewidth=lw, alpha=alpha)


def draw_rot_arc_3d(ax, center, axis, radius=0.12, color='purple', ref=None, sweep_deg=120):
    """center 에서 axis(회전축) 주위로 +방향 회전 화살표 호를 그림.

    호는 회전축에 수직인 평면 위에 놓여야 함. 그래서 축에 수직인
    정규직교 기저 (u, w) 를 만들고, 그 평면에서 cos/sin 으로 점을 찍음.
    radius 는 호 반지름, sweep_deg 는 호 각도, ref 는 기준 벡터(생략 시 자동 선택).
    """
    # 회전축을 단위벡터로 정규화함. 길이가 0이면 방향을 정할 수 없으므로 종료.
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return
    axis = axis / n

    # 회전축에 수직인 정규직교 기저 (u, w) 를 만듦.
    # 이 두 벡터가 회전 평면(축에 수직인 평면)의 x/y 역할을 함.
    if ref is None:
        # 기준 벡터가 축과 거의 평행하면 수직 성분이 사라지므로 다른 축을 고름.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, axis)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
    # u: ref 에서 축 방향 성분을 뺀 뒤(그람-슈미트) 정규화 -> 축에 수직
    u = ref - np.dot(ref, axis) * axis
    u = u / np.linalg.norm(u)
    # w: 축과 u 에 모두 수직 -> (u, w) 가 회전 평면의 정규직교 기저 완성
    w = np.cross(axis, u)

    # 회전 평면 위에서 cos/sin 으로 호 위의 점들을 만듦.
    center = np.asarray(center, dtype=float)
    th = np.linspace(0.0, np.radians(sweep_deg), 40)
    pts = center[:, None] + radius * (np.cos(th)[None, :] * u[:, None]
                                      + np.sin(th)[None, :] * w[:, None])
    ax.plot(pts[0], pts[1], pts[2], color=color, lw=1.6, alpha=0.85)

    # 호 끝에 화살촉을 붙임. 방향은 끝점에서의 접선(CCW) 방향.
    end = pts[:, -1]
    tang = -np.sin(th[-1]) * u + np.cos(th[-1]) * w
    ax.quiver(end[0], end[1], end[2], tang[0], tang[1], tang[2],
              color=color, length=radius * 0.7, normalize=True, linewidth=1.6)


def draw_base_2d(ax, origin=(0.0, 0.0), size=0.5):
    """고정된 base(world) 표시: 채워진 사각형(받침대) + 점선 world 좌표축.

    점선 축은 움직이는 joint 의 실선 축과 구분하기 위한 것임.
    size 는 world 좌표축 길이 기준.
    """
    from matplotlib.patches import Rectangle
    ox, oy = origin
    w = size * 0.5
    # 받침대 사각형: origin 을 중심으로 한 변 w
    ax.add_patch(Rectangle((ox - w / 2, oy - w / 2), w, w,
                           facecolor='dimgray', edgecolor='black', alpha=0.9, zorder=3))
    ax.plot([], [], 's', color='dimgray', markeredgecolor='black', label='Base')  # 범례용 프록시
    # 고정 world 좌표축: 점선으로 그려 joint 의 실선 축과 구분
    ax.plot([ox, ox + size], [oy, oy], 'r--', lw=1.5, alpha=0.7, zorder=4)
    ax.plot([ox, ox], [oy, oy + size], 'g--', lw=1.5, alpha=0.7, zorder=4)
    ax.text(ox + size * 1.05, oy, r'$X_0$', color='red', fontsize=8)
    ax.text(ox, oy + size * 1.05, r'$Y_0$', color='green', fontsize=8)


def draw_base_3d(ax, origin=(0.0, 0.0, 0.0), size=0.2, label=True):
    """고정된 base 표시: 얇은 직육면체(바닥판, 윗면이 origin) + 점선 world 좌표축.

    size : 바닥판 반폭(footprint) 겸 좌표축 길이 기준.
    바닥판은 반투명(얇음)이라 base 좌표축/회전 화살표를 가리지 않음.
    점선 축은 고정 world 좌표축으로, joint 의 실선 축과 구분됨.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ox, oy, oz = origin

    # matplotlib 3D 는 아티스트별 평균 깊이(depth)로만 앞뒤 순서를 정함.
    # 그대로 두면 바닥판(면)이 앞쪽 선을 덮어 흐리게 만들 수 있음.
    # computed_zorder 를 끄고 zorder 로 순서를 직접 지정해,
    # 바닥판은 맨 뒤, 로봇 팔/좌표축은 그 위에 온전히 보이게 함.
    ax.computed_zorder = False

    half = size                 # 바닥판 반폭
    thick = max(size * 0.12, 0.01)   # 얇은 두께
    x0, x1 = ox - half, ox + half
    y0, y1 = oy - half, oy + half
    z1, z0 = oz, oz - thick     # 윗면이 base origin 높이

    # 직육면체의 8개 꼭짓점 좌표 (아랫면 4개 + 윗면 4개)
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 아랫면
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 윗면
    ])
    # 6개 면을 꼭짓점 인덱스로 정의 -> 실제 좌표 리스트로 변환
    face_idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    faces = [[corners[i] for i in f] for f in face_idx]
    # zorder=-10 : 바닥판을 가장 뒤로 보내 다른 요소를 가리지 않게 함.
    plate = Poly3DCollection(faces, facecolors='lightgray', edgecolors='gray',
                             linewidths=0.6, alpha=0.6, zorder=-10)
    ax.add_collection3d(plate)
    if label:
        ax.plot([], [], [], 's', color='lightgray', markeredgecolor='gray', label='Base')  # 범례용 프록시

    # 고정 world 좌표축(점선). origin 에서 뻗어나가며 바닥판 밖으로 살짝 나옴.
    R = np.eye(3)
    names = [r'$X_0$', r'$Y_0$', r'$Z_0$']
    alen = size * 1.2
    # R(=단위행렬)의 각 열이 world x/y/z 축 방향 -> 점선으로 그리고 라벨을 붙임.
    for col, c, name in zip(range(3), ['red', 'green', 'blue'], names):
        v = R[:, col] * alen
        ax.plot([ox, ox + v[0]], [oy, oy + v[1]], [oz, oz + v[2]],
                linestyle='--', color=c, lw=1.6, alpha=0.9, zorder=-5)
        if label:
            ax.text(ox + v[0], oy + v[1], oz + v[2], name, color=c, fontsize=8)


def add_axis_color_note(fig, x=0.01, y=0.01):
    """좌표축 색 규칙 범례 텍스트를 figure 하단에 추가."""
    fig.text(x, y, "Frame axes:  X = Red   Y = Green   Z = Blue    |    "
                   "arc = joint rotation (+) direction    |    "
                   "dashed axes (■ Base) = fixed world frame",
             fontsize=9, color='dimgray')
