#!/usr/bin/env python3
"""
3.1 Single-View RGB-D → Point Cloud
================================================================
RGB-D 이미지 '한 장'을 카메라 내부 파라미터(intrinsic)로 역투영하여,
카메라(광학) 좌표계의 3D 포인트 클라우드로 복원함.

배우는 것: 핀홀 카메라 모델의 역투영
    z = depth
    x = (u - cx) * z / fx     # u: 픽셀 가로, cx/fx: 주점/초점거리
    y = (v - cy) * z / fy     # v: 픽셀 세로, cy/fy: 주점/초점거리

단일 뷰만 다루므로 world 정합(TF)은 하지 않음.
여러 뷰를 공통 좌표계로 합치는 것은 3.2 에서 다룸.

"""
import os              # 파일 경로 조합 및 존재 여부 확인
import sys             # 인터프리터 제어(예: 오류 시 프로그램 종료)
import glob            # 패턴으로 여러 이미지/데이터 파일 경로 일괄 검색
import yaml            # 카메라 내부 파라미터 등 YAML 설정 파일 읽기
import numpy as np     # 수치 계산: 배열·행렬 연산
import cv2             # 이미지 로드·처리(OpenCV)
import open3d as o3d   # 포인트클라우드 생성·처리·시각화


# ──────────────────────────────────────────────
# 기본 데이터 경로: 프로젝트 루트의 data/handeye_data (day1 폴더 기준 한 단계 위, 실행 위치와 무관)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "handeye_data")
DEPTH_SCALE = 1000.0     # depth png 단위: mm → m
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 1.5

# 사용할 뷰 인덱스 (data/handeye_data 안의 view_000, view_001, ... 중 하나)
VIEW_IDX = 0


def load_view(view_dir):
    """한 뷰의 RGB, depth(m), 카메라 내부행렬 K 로드."""
    with open(os.path.join(view_dir, "camera_info.yaml")) as f:
        cam_info = yaml.safe_load(f)
    K = np.array(cam_info["K"]).reshape(3, 3)
    rgb = cv2.cvtColor(cv2.imread(os.path.join(view_dir, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(view_dir, "depth.png"), cv2.IMREAD_ANYDEPTH)
    # depth png 는 mm 정수 → m 단위 실수로 변환
    depth_m = depth.astype(np.float32) / DEPTH_SCALE
    return rgb, depth_m, K


def backproject(rgb, depth_m, K, max_depth=MAX_DEPTH_M):
    """RGB-D → 카메라(광학) 좌표계 3D 포인트 + 색상."""
    # ──────────────────────────────────────────────────────────
    # [실습] 핀홀 카메라 모델로 depth 이미지를 3D 점으로 역투영하세요.
    #   1) intrinsic 꺼내기: fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2]
    #   2) 유효 픽셀 마스크: depth 가 0(측정 실패)이거나 너무 멀면 제외
    #        valid = (depth_m > MIN_DEPTH_M) & (depth_m < max_depth)
    #        vs, us = np.where(valid)      # v: 세로(행), u: 가로(열)
    #        z = depth_m[vs, us]
    #   3) 핀홀 역투영 (픽셀 + 깊이 → 카메라 좌표):
    #        x = (us - cx) * z / fx
    #        y = (vs - cy) * z / fy
    #   4) points = np.stack([x, y, z], axis=1)          # (N, 3)
    #      colors = rgb[vs, us] / 255.0                  # 0~1 로 정규화
    #   반환: points, colors
    # ──────────────────────────────────────────────────────────
    raise NotImplementedError("TODO: backproject 를 구현하세요")


def main():
    session_dir = DEFAULT_DATA_DIR
    if not os.path.isdir(session_dir):
        print(f"ERROR: 경로 없음: {session_dir}")
        sys.exit(1)

    view_dirs = sorted(glob.glob(os.path.join(session_dir, "view_*")))
    if not view_dirs:
        print(f"ERROR: 뷰 없음: {session_dir}")
        sys.exit(1)
    if VIEW_IDX >= len(view_dirs):
        print(f"ERROR: view {VIEW_IDX} 없음 (총 {len(view_dirs)}뷰)")
        sys.exit(1)

    view_dir = view_dirs[VIEW_IDX]
    print(f"데이터: {session_dir}")
    print(f"뷰:   {os.path.basename(view_dir)}")

    # RGB-D 로드 및 역투영
    rgb, depth_m, K = load_view(view_dir)
    print(f"이미지 크기: {rgb.shape[1]}x{rgb.shape[0]}, "
          f"intrinsic fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    points, colors = backproject(rgb, depth_m, K, max_depth=MAX_DEPTH_M)
    print(f"복원된 3D 포인트: {len(points):,}개")
    print(f"  X range: [{points[:,0].min():.3f}, {points[:,0].max():.3f}] m")
    print(f"  Y range: [{points[:,1].min():.3f}, {points[:,1].max():.3f}] m")
    print(f"  Z range: [{points[:,2].min():.3f}, {points[:,2].max():.3f}] m (카메라 전방 거리)")

    # Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 좌표축 = 카메라 광학 좌표계 원점 (Z=바라보는 방향, X=우, Y=하)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    print("\n[시각화] 좌표축 = 카메라(광학) 좌표계  (R=X우, G=Y하, B=Z=전방)")
    o3d.visualization.draw_geometries(
        [pcd, cam_frame],
        window_name=f"3.1 Single-View PCD ({os.path.basename(view_dir)})",
        width=1024, height=768,
    )


if __name__ == "__main__":
    main()
