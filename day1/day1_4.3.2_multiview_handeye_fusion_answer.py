#!/usr/bin/env python3
"""
3.2 Multi-View Hand-Eye Fusion
================================================================
여러 시점(view)에서 찍은 RGB-D 이미지를, hand-eye 캘리브레이션과 로봇 FK 를 이용해
공통 world 좌표계(panda_link0)로 정합·융합함.
(3.1 = 단일 뷰 역투영,  3.2 = 다중 뷰 정합/융합)

배우는 것
--------------
1) RGB-D → 카메라(광학) 좌표계 3D 포인트 역투영
2) 광학 좌표계 점을 OPTICAL_TO_CAMLINK 로 camera_link 기준으로 통일
3) hand-eye 캘리브(camera_link 점 → franka_ee 점)와 로봇 FK 를 곱해 world 로 정합

최종 변환식:
    p_world = base_to_ee @ T_ee_camlink @ OPTICAL_TO_CAMLINK @ p_optical

- base_to_ee    : 로봇 FK(관절각 → EE 위치). 캘리브와 무관하며 뷰마다 다름.
- T_ee_camlink  : hand-eye 캘리브. franka_ee 프레임 기준 camera_link 의 포즈이며,
                  점 변환으로는 camera_link 점 → franka_ee 점. 카메라 장착 방식을
                  나타내는 고정 변환.

"""

import os            # 파일 경로 조합 및 존재 여부 확인
import sys           # 인터프리터 제어(예: 오류 시 프로그램 종료)
import glob          # 패턴으로 여러 이미지/데이터 파일 경로 일괄 검색
import yaml          # 카메라 파라미터·핸드아이 설정 등 YAML 설정 파일 읽기
import numpy as np   # 수치 계산: 배열·행렬 연산
import cv2           # 이미지 로드·처리(OpenCV)
import open3d as o3d  # 포인트클라우드 생성·처리·시각화
from scipy.spatial.transform import Rotation as R_scipy  # 회전 표현 변환(행렬↔쿼터니언↔오일러)

# ──────────────────────────────────────────────
# 융합 파라미터
# 기본 데이터 경로: 프로젝트 루트의 data/handeye_data (day1 폴더 기준 한 단계 위)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "handeye_data")
DEPTH_SCALE = 1000.0     # depth png 단위: mm → m
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 1.5
VOXEL_SIZE  = 0.003      # 3mm 복셀 다운샘플
OUTLIER_NB  = 30
OUTLIER_STD = 2.0


def _tf_mat(t, q_xyzw):
    """(translation, quaternion[xyzw]) → 4x4 동차변환행렬."""
    M = np.eye(4)
    M[:3, :3] = R_scipy.from_quat(q_xyzw).as_matrix()
    M[:3,  3] = t
    return M


# ──────────────────────────────────────────────
# [프레임 변환] 광학 좌표계 점 → camera_link 점
#   두 좌표계는 축 방향 규칙만 다르므로(광학 x-우/y-하/z-전 ↔ camera_link x-전/y-좌/z-상),
#   순수 회전 하나로 광학 점을 camera_link 로 바로 옮긴다.
#   (표준 ROS optical 규칙 [-0.5,0.5,-0.5,0.5] 는 camera_link→optical 이고, 그 역이 아래 값.)
OPTICAL_TO_CAMLINK = np.eye(4)
OPTICAL_TO_CAMLINK[:3, :3] = R_scipy.from_quat([0.5, -0.5, 0.5, 0.5]).as_matrix()  # 광학 → camera_link


# ──────────────────────────────────────────────
# [Hand-Eye 캘리브] franka_ee 기준 camera_link 의 포즈 (점 변환: camera_link 점 → franka_ee 점)
#   값 형식: (x y z  qx qy qz qw)
T_EE_CAMLINK = _tf_mat([0.067978, -0.058042, -0.088245],
                       [0.703290,  0.055780,  0.104820, -0.700917])


def get_base_to_ee(tf_data):
    """
    로봇 FK 부분(base → franka_ee)을 tf.yaml 의 base_to_ee 에서 읽음.
    """
    if "base_to_ee" not in tf_data:
        raise ValueError("tf.yaml 에 base_to_ee 가 없습니다. ")
    return np.array(tf_data["base_to_ee"])


def load_view(view_dir):
    """한 뷰의 데이터 로드 (RGB, depth, K, tf_data)."""
    with open(os.path.join(view_dir, "camera_info.yaml")) as f:
        cam_info = yaml.safe_load(f)
    with open(os.path.join(view_dir, "tf.yaml")) as f:
        tf_data = yaml.safe_load(f)

    K = np.array(cam_info["K"]).reshape(3, 3)
    rgb = cv2.cvtColor(cv2.imread(os.path.join(view_dir, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(view_dir, "depth.png"), cv2.IMREAD_ANYDEPTH)
    depth_m = depth.astype(np.float32) / DEPTH_SCALE
    return rgb, depth_m, K, tf_data


def backproject(depth_m, K):
    """depth → 3D 포인트(광학 좌표계, 동차좌표 Nx4)와 유효 픽셀 인덱스."""
    valid = (depth_m > MIN_DEPTH_M) & (depth_m < MAX_DEPTH_M)
    vs, us = np.where(valid)
    z = depth_m[vs, us]
    # 핀홀 역투영 (3.1 과 동일)
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts_optical = np.stack([x, y, z, np.ones_like(z)], axis=1)   # (N, 4)
    return pts_optical, vs, us


def make_pcd(rgb, depth_m, K, tf_data, T_ee_camlink):
    """
    RGB-D → world 좌표계 PointCloud.
    p_world = base_to_ee @ T_ee_camlink @ OPTICAL_TO_CAMLINK @ p_optical
    """
    pts_optical, vs, us = backproject(depth_m, K)
    base_to_ee = get_base_to_ee(tf_data)

    T_base_camlink = base_to_ee @ T_ee_camlink                 # base → camera_link
    pts_world = (T_base_camlink @ OPTICAL_TO_CAMLINK @ pts_optical.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(rgb[vs, us].astype(np.float64) / 255.0)
    return pcd


def main():
    # 기본 데이터 경로와 hand-eye 캘리브(T_EE_CAMLINK)로 실행
    session_dir = DEFAULT_DATA_DIR
    T_ee_camlink = T_EE_CAMLINK

    if not os.path.isdir(session_dir):
        print(f"ERROR: 경로 없음: {session_dir}")
        sys.exit(1)

    view_dirs = sorted(glob.glob(os.path.join(session_dir, "view_*")))
    if not view_dirs:
        print(f"ERROR: 뷰 없음: {session_dir}")
        sys.exit(1)

    print("=" * 55)
    print(f"  Hand-Eye Fusion ({len(view_dirs)}뷰) → camera_link 기준 정합")
    print(f"  세션: {session_dir}")
    print("=" * 55)

    pcds_rgb, cam_poses = [], []
    for vd in view_dirs:
        rgb, depth_m, K, tf_data = load_view(vd)
        cam_frame = tf_data.get("camera_frame", "?")

        # 이 뷰를 world 로 정합한 뒤, 복셀 다운샘플로 점 개수를 줄여 누적
        pcd_r = make_pcd(rgb, depth_m, K, tf_data, T_ee_camlink)
        pcd_r = pcd_r.voxel_down_sample(VOXEL_SIZE)
        pcds_rgb.append(pcd_r)

        # 이 뷰의 카메라 optical 프레임 월드 포즈 (좌표축 시각화용).
        # camera_link 는 시선축이 Z 가 아니어서 덜 직관적이라, 시선=+Z 인 optical 로 표시.
        T_base_camlink = get_base_to_ee(tf_data) @ T_ee_camlink
        cam_poses.append(T_base_camlink @ OPTICAL_TO_CAMLINK)

        pts = np.asarray(pcd_r.points)
        print(f"  {os.path.basename(vd)} [{cam_frame}]: {len(pts):,} pts  "
              f"centroid=[{pts[:,0].mean():.3f}, {pts[:,1].mean():.3f}, {pts[:,2].mean():.3f}]")

    # point cloud 합성
    merged = o3d.geometry.PointCloud()
    for p in pcds_rgb:
        merged += p
    merged_ds = merged.voxel_down_sample(VOXEL_SIZE)
    merged_clean, _ = merged_ds.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)

    out_ply = os.path.join(session_dir, "merged_pointcloud.ply")
    o3d.io.write_point_cloud(out_ply, merged_clean)
    print(f"\n[Fusion] 합산 {len(merged.points):,} → 다운샘플 {len(merged_ds.points):,} "
          f"→ 클린 {len(merged_clean.points):,}")
    print(f"[Fusion] ✓ {out_ply}")

    # 큰 좌표축 = base(panda_link0), 작은 좌표축 = 각 뷰 카메라 optical 프레임
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    cam_frames = []
    for T_bc in cam_poses:
        cam_ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        cam_ax.transform(T_bc)
        cam_frames.append(cam_ax)

    print("\n[시각화] 큰 좌표축=base(panda_link0), 작은 좌표축=각 카메라 optical 프레임")
    print("         (R=X우, G=Y하, B=Z=카메라가 바라보는 방향)")

    center = np.concatenate([np.asarray(p.points) for p in pcds_rgb], axis=0).mean(axis=0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3.2 Multi-View Hand-Eye Fusion (RGB)", width=1280, height=720)
    for g in pcds_rgb + [base_frame] + cam_frames:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_front([0.0, -0.5, -0.8]); ctr.set_lookat(center.tolist())
    ctr.set_up([0.0, 0.0, 1.0]); ctr.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
