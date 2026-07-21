import numpy as np
import random

def depth2pc(depth, K, rgb=None):
    """ 뎁스 이미지를 포인트 클라우드로 변환하는 함수 """

    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32)-K[0,2])
    normalized_y = (y.astype(np.float32)-K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y, x]

    pc = np.vstack([world_x, world_y, world_z]).T
    return (pc, rgb)

def get_random_color():
    """ 시각화를 위해 랜덤 RGB 색상을 생성합니다. """
    return [random.randint(50, 255) for _ in range(3)] # 너무 어둡지 않은 색상
