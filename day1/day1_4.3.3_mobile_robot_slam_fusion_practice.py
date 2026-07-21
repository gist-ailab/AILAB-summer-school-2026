#!/usr/bin/env python3
"""
3.3 Mobile-Robot SLAM PCD Fusion
================================================================
모바일 로봇 SLAM 시퀀스(여러 RGB-D 프레임 + 프레임별 카메라 pose)를 공통 world
좌표계에 정합·누적하여 하나의 3D 지도(point cloud)를 만듦.

- 각 프레임: RGB-D 역투영 → 카메라 pose(T_WC)로 world 정합 → voxel 다운샘플로 누적
- pose(T_WC) 는 metadata.pkl 에 미리 전처리돼 있어야 함.

(3.1/3.2 = franka 팔 카메라, 3.3 = 모바일 로봇 SLAM 데이터)
"""
import os              # 파일 경로 조합 및 존재 여부 확인
import pickle          # 전처리된 데이터(파이썬 객체) 직렬화 저장/로드
import numpy as np     # 수치 계산: 배열·행렬 연산
import open3d as o3d   # 포인트클라우드 생성·처리·시각화
from PIL import Image  # 이미지 파일 로드(Pillow)

def pcd_from_rgbd_vectorized(rgb_arr, depth_arr, intrinsics, max_depth=3.5):
    """
    NumPy 벡터 연산으로 2D RGB-D 이미지에서
    카메라 좌표계의 3D Point Cloud 를 한 번에 역투영함.
    """
    # ──────────────────────────────────────────────────────────
    # [실습] 핀홀 역투영을 NumPy 벡터 연산으로 '한 번에' 구현하세요.
    #   (4.3.1 의 for/where 대신 meshgrid + boolean mask 방식)
    #   1) intrinsic: fx=intrinsics[0,0], fy=[1,1], cx=[0,2], cy=[1,2]
    #   2) 픽셀 격자: u, v = np.meshgrid(np.arange(width), np.arange(height))
    #   3) 유효 depth 마스크 (여기 depth 단위는 mm):
    #        mask = (depth_arr > 0) & (depth_arr < max_depth * 1000)
    #        z_c = depth_arr[mask] / 1000.0      # mm → m
    #        u_c = u[mask],  v_c = v[mask]
    #   4) 핀홀 역투영:
    #        x_c = (u_c - cx) * z_c / fx
    #        y_c = (v_c - cy) * z_c / fy
    #   5) points_c = np.stack((x_c, y_c, z_c), axis=-1)
    #      colors_c = rgb_arr[mask] / 255.0
    #   반환: points_c, colors_c
    # ──────────────────────────────────────────────────────────
    raise NotImplementedError("TODO: pcd_from_rgbd_vectorized 를 구현하세요")

def main():
    # ── 사용자 설정 ────────────────────────────────────────
    # 데이터 폴더: 프로젝트 루트의 data (day1 폴더 기준 한 단계 위). 실행 위치와 무관하게 동작
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'slam_map_data')

    # 융합에 사용할 프레임 번호 (원하는 대로 편집)
    FRAMES = [i for i in range(0, 1800, 10)]

    max_depth_threshold = 3.5   # 노이즈 차단 최대 깊이 (m)
    voxel_size = 0.01           # 1cm 복셀 다운샘플
    # ───────────────────────────────────────────────────────

    # 카메라 축 보정 행렬: 카메라에서 사용하는 좌표축과 시각화 시 사용하는 좌표축이 다른 경우 있음
    CAM_AXIS_FIX = np.eye(4)

    # 1. SLAM 메타데이터 로드
    metadata_path = os.path.join(DATA_DIR, 'metadata.pkl')
    if not os.path.exists(metadata_path):
        print(f"[Error] '{metadata_path}'를 찾을 수 없습니다.")
        return

    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # 프레임별 카메라 포즈(T_WC), RGB/Depth 경로, 카메라 내부 파라미터
    #   portable pkl 은 배열이 list 로 저장돼 있어 np.asarray 로 복원 (원본이면 그대로 통과)
    poses = np.asarray(metadata['poses'])
    rgb_paths = metadata['rgb_paths']
    depth_paths = metadata['depth_paths']
    intrinsics = np.asarray(metadata['intrinsics'])

    total_frames = len(rgb_paths)
    print(f"Loaded metadata. Total frames: {total_frames}")

    all_points = []
    all_colors = []
    cam_poses = []   # 각 프레임의 카메라 pose (좌표축 시각화용)

    print(f"\nStarting 3D PCD Fusion over {len(FRAMES)} selected frames: {FRAMES}")

    # 2. 지정한 프레임들을 각각 RGB-D 역투영 → world 좌표계로 정합하며 누적
    processed_count = 0
    for idx in FRAMES:
        # 지정한 프레임 번호가 범위를 벗어나면 건너뜀
        if idx < 0 or idx >= total_frames:
            print(f"  [skip] frame {idx} 범위 밖 (0~{total_frames - 1})")
            continue

        # 현재 프레임의 RGB/Depth 경로와 카메라->월드 변환행렬(T_WC)
        rgb_rel_path = rgb_paths[idx]
        depth_rel_path = depth_paths[idx]
        T_WC = poses[idx]

        # 이미지 전체 파일 경로 매핑
        rgb_file = os.path.join(DATA_DIR, rgb_rel_path)
        depth_file = os.path.join(DATA_DIR, depth_rel_path)

        if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
            continue

        # 이미지 로드
        rgb_img = np.array(Image.open(rgb_file))
        depth_img = np.array(Image.open(depth_file))

        # 단일 RGB-D 이미지에서 카메라 좌표계 3D 포인트 추출
        points_c, colors_c = pcd_from_rgbd_vectorized(
            rgb_img, depth_img, intrinsics, max_depth=max_depth_threshold
        )

        # Open3D 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_c)
        pcd.colors = o3d.utility.Vector3dVector(colors_c)

        # 카메라 pose(T_WC)로 공통 world 좌표계에 정합.
        # 프레임마다 카메라 위치가 다르므로, 모두 같은 world 좌표계로 옮겨야 겹쳐짐.
        # 적용 순서: p_world = T_WC @ CAM_AXIS_FIX @ p_camera
        T_world_cam = T_WC @ CAM_AXIS_FIX
        pcd.transform(T_world_cam)

        # 월드 좌표계 포인트 누적 + 카메라 pose 저장(시각화용)
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))
        cam_poses.append(T_world_cam)

        processed_count += 1
        print(f"  [{processed_count}] Fused Frame {idx:04d} -> Extracted {len(points_c)} points")

    if len(all_points) == 0:
        print("[Error] Fused point cloud is empty!")
        return

    # 3. 누적된 3D 점들을 하나의 PCD 객체로 결합
    print("\nMerging all individual point clouds into a single map...")
    fused_pcd = o3d.geometry.PointCloud()
    fused_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    fused_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    print(f"Merged raw point cloud size: {len(fused_pcd.points)} points")

    # 4. 복셀 다운샘플링: 여러 프레임이 겹쳐 같은 위치에 몰린 점들을
    #    일정 크기 격자마다 대표 1점으로 줄여 중복 제거 및 용량 최적화
    print(f"Applying Voxel Downsampling (voxel size = {voxel_size} m)...")
    optimized_pcd = fused_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Optimized point cloud size: {len(optimized_pcd.points)} points")

    # 5. 최종 퓨전 PCD 맵 저장 (데이터 폴더 안에)
    output_pcd_path = os.path.join(DATA_DIR, 'fused_map.pcd')
    print(f"Saving fused point cloud to {output_pcd_path}...")
    o3d.io.write_point_cloud(output_pcd_path, optimized_pcd)

    # 6. 3D 지도 시각화 (맵 + world 좌표축 + 각 프레임 카메라 pose 좌표축)
    #    큰 좌표축 = world 원점, 작은 좌표축 = 각 카메라 (R=X, G=Y, B=Z=바라보는 방향)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cam_frames = []
    for T in cam_poses:
        cam_ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        cam_ax.transform(T)
        cam_frames.append(cam_ax)

    print("Launching Open3D visualizer. Close the window to complete the script.")
    print("  큰 좌표축=world 원점, 작은 좌표축=각 카메라 pose (파란 Z=바라보는 방향)")
    o3d.visualization.draw_geometries(
        [optimized_pcd, world_frame] + cam_frames,
        window_name="3.3 Mobile-Robot SLAM PCD Fusion Map",
        width=1024,
        height=768
    )

if __name__ == "__main__":
    main()
