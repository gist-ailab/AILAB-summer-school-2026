import torch
import cgnet.utils.utils as utils
import numpy as np
from cgnet.utils.collision_detector import ModelFreeCollisionDetector
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def inference_cgnet(pc, model, device, hand_pose_w, env, object_mask=None, obj_dist_thresh=0.02, pc_colors=None, vis=False):
    """
    Args:
        pc: (N, 3) 전체 장면 point cloud (camera frame). 바닥/주변 물체 포함.
        object_mask: (N,) boolean. pc 중 타겟 물체에 속하는 점을 표시.
            지정되면, 파지 anchor 점이 타겟 물체 표면에 있는 grasp만 남김
            (바닥면/옆 물체에 생기는 파지 제거). 충돌검사는 전체 pc로 수행.
        obj_dist_thresh: anchor 점이 타겟 물체 점과 이 거리(m) 이내이면 타겟 파지로 간주.
        pc_colors: (N, 3) 각 점의 RGB 색상 (0~255 또는 0~1). 지정되면 시각화에 활용.
    """
    # RGB 컬러를 open3d 형식(0~1 float, (N,3))으로 정규화
    pc_colors_norm = None
    if pc_colors is not None:
        pc_colors_norm = np.asarray(pc_colors, dtype=np.float32)[:, :3]
        if pc_colors_norm.max() > 1.0:
            pc_colors_norm = pc_colors_norm / 255.0
        pc_colors_norm = np.clip(pc_colors_norm, 0.0, 1.0)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
 
    pc_torch = torch.from_numpy(pc).float().to(device)
    pred = model(pc_torch.unsqueeze(0))
    pred_grasps = pred['pred_grasps'].detach().cpu() #(B, N, 4, 4)
    pred_scores = pred['pred_scores'] # (B, N)
    pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
    pred_graps_width_bin = pred['pred_width'] # (B, N, 1)

    pred_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
    pred_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)

    sorted_pred_score, sorted_idx = torch.topk(pred_scores.squeeze(), k=2048, largest=True)
    sorted_idx = sorted_idx.detach().cpu()
    sorted_pred_rot = pred_rot[:, sorted_idx, :, :]
    sorted_pred_trans = pred_trans[:, sorted_idx, :]
    sorted_pred_width_bin = pred_graps_width_bin[:, sorted_idx, :]
    sorted_pred_points = pred_points[:, sorted_idx, :]

    sorted_pred_rot = sorted_pred_rot.detach().cpu().numpy()[0] #(2048, 3, 3)
    sorted_pred_trans = sorted_pred_trans.detach().cpu().numpy()[0] #(2048, 3)
    sorted_pred_score = sorted_pred_score.detach().cpu().numpy()
    sorted_pred_width_bin = sorted_pred_width_bin.detach().cpu().numpy()[0, :, 0]
    sorted_pred_points = sorted_pred_points.detach().cpu().numpy()[0] #(2048, 3)

    # fix for numpy.float issue in graspnetAPI
    np.float = float
    np.float_ = np.float64

    # from graspnetAPI.graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
    # from graspnetAPI.graspnetAPI import GraspGroup
    from graspnetAPI import GraspGroup

    def _build_grasp_group(rots, transs, scores, widths):
        """sorted 파지 배열들로부터 GraspGroup 생성 (시각화/후처리용)"""
        arr = []
        for s, w, rt, tr in zip(scores, widths, rots, transs):
            arr.append([s, w, 0.02, 0.02, *rt.reshape(-1), *tr.reshape(-1), -1])
        return GraspGroup(np.array(arr))

    # ===== 타겟 물체 마스크로 파지 필터링 =====
    # 파지 anchor 점(pred_points)이 타겟 물체 표면에 속하는 grasp만 남겨,
    # 바닥면이나 옆에 붙어있는 다른 물체에 생긴 파지를 제거한다.
    if object_mask is not None:
        from scipy.spatial import cKDTree
        object_mask = np.asarray(object_mask).reshape(-1).astype(bool)
        obj_pc = pc[object_mask]

        # 필터링 전 파지 백업 (시각화용)
        before_rot, before_trans = sorted_pred_rot.copy(), sorted_pred_trans.copy()
        before_score, before_width = sorted_pred_score.copy(), sorted_pred_width_bin.copy()

        if len(obj_pc) > 0:
            tree = cKDTree(obj_pc)
            nn_dist, _ = tree.query(sorted_pred_points)  # 각 anchor의 타겟 물체까지 최단거리
            keep = nn_dist < obj_dist_thresh
            if keep.sum() > 0:
                sorted_pred_rot = sorted_pred_rot[keep]
                sorted_pred_trans = sorted_pred_trans[keep]
                sorted_pred_score = sorted_pred_score[keep]
                sorted_pred_width_bin = sorted_pred_width_bin[keep]
                sorted_pred_points = sorted_pred_points[keep]
                print(f"[INFO] Object-mask grasp filtering: {int(keep.sum())}/{len(keep)} grasps on target object")
            else:
                print("[WARN] No grasp anchored on target object. Using unfiltered grasps.")
        else:
            print("[WARN] Empty object mask. Using unfiltered grasps.")

        # ----- 필터링 전(빨강) vs 후(초록) 파지 시각화 -----
        TOPK_VIS = 100
        vis_pc = o3d.geometry.PointCloud()
        vis_pc.points = o3d.utility.Vector3dVector(pc)
        if pc_colors_norm is not None:
            # RGB 이미지 기반 컬러 point cloud, 타겟 물체는 파란색을 살짝 섞어 강조
            colors = pc_colors_norm.copy()
            colors[object_mask] = 0.5 * colors[object_mask] + 0.5 * np.array([0.0, 0.3, 1.0])
        else:
            colors = np.tile(np.array([0.6, 0.6, 0.6]), (len(pc), 1))  # 전체 점: 회색
            colors[object_mask] = np.array([0.0, 0.3, 1.0])            # 타겟 물체 점: 파랑
        vis_pc.colors = o3d.utility.Vector3dVector(colors)

        before_geoms = [vis_pc]
        gg_before = _build_grasp_group(before_rot[:TOPK_VIS], before_trans[:TOPK_VIS], before_score[:TOPK_VIS], before_width[:TOPK_VIS])
        for g in gg_before.to_open3d_geometry_list():
            g.paint_uniform_color([1.0, 0.0, 0.0])  # 빨강: 필터링 전
            before_geoms.append(g)
        print("[VIS] Showing grasps BEFORE object-mask filtering (red). Close window to continue...")
        if vis:
            # 카메라 광학 프레임(+z 전방, +y 아래)에 맞춘 초기 뷰 → RGB 이미지처럼 똑바로 보이게
            o3d.visualization.draw_geometries(
                before_geoms, window_name="Grasps BEFORE filtering (red)",
                front=[0.0, 0.0, -1.0], up=[0.0, -1.0, 0.0],
                lookat=vis_pc.get_center(), zoom=0.7,
            )

        after_geoms = [vis_pc]
        gg_after = _build_grasp_group(sorted_pred_rot[:TOPK_VIS], sorted_pred_trans[:TOPK_VIS], sorted_pred_score[:TOPK_VIS], sorted_pred_width_bin[:TOPK_VIS])
        for g in gg_after.to_open3d_geometry_list():
            g.paint_uniform_color([0.0, 1.0, 0.0])  # 초록: 필터링 후
            after_geoms.append(g)
        print("[VIS] Showing grasps AFTER object-mask filtering (green). Close window to continue...")
        if vis:
            # 카메라 광학 프레임에 맞춘 초기 뷰
            o3d.visualization.draw_geometries(
                after_geoms, window_name="Grasps AFTER filtering (green)",
                front=[0.0, 0.0, -1.0], up=[0.0, -1.0, 0.0],
                lookat=vis_pc.get_center(), zoom=0.7,
            )

    g_array = []
    for i in range(len(sorted_pred_score)):
        score = sorted_pred_score[i]
        width = sorted_pred_width_bin[i]
        rot = sorted_pred_rot[i] # (3, 3)
        trans = sorted_pred_trans[i].reshape(-1) # (3,)
        
        trans = trans.reshape(1, 3)
        trans = trans.reshape(-1)
        
        rot = rot.reshape(-1)
        g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
    
    g_array = np.array(g_array)
    gg = GraspGroup(g_array)
    
    # check collisoin
    mfcdetector = ModelFreeCollisionDetector(pc, voxel_size=0.005)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    gg = gg.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 10:
        gg = gg[:10]
    gg_vis = gg[:1]
    gg_vis_trans = gg[:1]
    gg_vis_trans.translations += (gg_vis_trans.rotation_matrices[0] @ np.array([[0.1, 0, 0]]).reshape(3, 1)).reshape(1,3)
    grippers = gg_vis.to_open3d_geometry_list()
    grippers_trans = gg_vis_trans.to_open3d_geometry_list()
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    
    rot_top1 = gg.rotation_matrices[0]           # (3, 3)
    trans_top1 = gg.translations[0]              # (3,)
    width_top1 = gg.widths[0]                    # scalar

    T_g2c = np.eye(4)
    T_g2c[:3, :3] = rot_top1
    T_g2c[:3, 3] = trans_top1

    # ======================== 1) Origin & Base Pose ========================
    origin = env.scene.env_origins.cpu().numpy()

    rs = env.scene["robot"].data.root_state_w[0]
    base_pos_w = rs[:3].cpu().numpy() - origin
    base_quat_wxyz = rs[3:7].cpu().numpy()  # (w, x, y, z)


    base_rot = R.from_quat(base_quat_wxyz, scalar_first=True)
    T_b2w = np.eye(4)
    T_b2w[:3, :3] = base_rot.as_matrix()
    T_b2w[:3, 3] = base_pos_w

    # ======================== 2) Handeye Camera Pose ========================
    # 손으로 offset/convention(z 180도 등)을 재구성하지 않고, Camera 센서가 제공하는
    # 실제 월드 pose를 그대로 사용한다. (depth2pc로 만든 점은 ROS optical 프레임 기준)

    hand_pos_w = hand_pose_w[0, 0:3].cpu().numpy()
    hand_quat_w = hand_pose_w[0, 3:7].cpu().numpy()  # (w, x, y, z)

    # rotate hand z축으로 180도 (아래 robot_hand_frame 시각화용)
    hand_rot = R.from_quat(hand_quat_w, scalar_first=True)
    T_ee2w = np.eye(4)
    T_ee2w[:3, :3] = hand_rot.as_matrix()
    T_ee2w[:3, 3] = hand_pos_w

    # World to EE frame 시각화
    ee_frame_sensor = env.scene["ee_frame"]
    ee_pos_w = ee_frame_sensor.data.target_pos_w[0, :].cpu().numpy()
    ee_quat_w = ee_frame_sensor.data.target_quat_w[0, :].cpu().numpy()  # (x, y, z, w) or (w, x, y, z) 확인 필요

    # 쿼터니언 → 회전행렬 변환 (scipy는 (x, y, z, w) 순서)
    ee_rot_w = R.from_quat(ee_quat_w, scalar_first=True)
    ee_rot_w = ee_rot_w # * R.from_euler('z', 180, degrees=True)  # z축 180도 회전 추가
    ee_rot_w_mat = ee_rot_w.as_matrix()

    T_w_ee_isaac = np.eye(4)
    T_w_ee_isaac[:3, :3] = ee_rot_w_mat
    T_w_ee_isaac[:3, 3] = ee_pos_w

    # 카메라 pose 계산: Camera 센서의 실제 월드 pose를 직접 사용 (env-origin 기준)
    # pos=(0.1, 0.035, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678)
    offset_pos = np.array([0.1, 0.035, 0.0])
    offset_quat_wxyz = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # (w, x, y, z)
    offset_rot = R.from_quat([
        offset_quat_wxyz[1], offset_quat_wxyz[2], offset_quat_wxyz[3], offset_quat_wxyz[0]
    ])  # (x, y, z, w)

    # offset_rot = R.from_quat(offset_quat_wxyz)
    T_c2ee = np.eye(4)
    T_c2ee[:3, :3] = offset_rot.as_matrix()
    T_c2ee[:3, 3] = offset_pos 
    
    T_c2w = T_ee2w @ T_c2ee

    # ======================== 3) PointCloud & Grippers → world ========================
    pc_world = (T_c2w[:3, :3] @ pc.T).T + T_c2w[:3, 3]
    pc_o3d.points = o3d.utility.Vector3dVector(pc_world)
    # RGB 이미지 기반 컬러 적용 (color point cloud)
    if pc_colors_norm is not None:
        pc_o3d.colors = o3d.utility.Vector3dVector(pc_colors_norm)

    # Grippers → world
    for g in grippers:
        g.transform(T_c2w)
    for g in grippers_trans:
        g.transform(T_c2w)

    # ======================== 4) Grasp Pose Axes ========================
    # Grasp pose (world)
    T_p2g = np.eye(4)
    T_p2g[:3, :3] = R.from_euler('xyz', [0, -np.pi/2, np.pi]).as_matrix()   # y+90, z+180
    T_grasp2w = T_c2w @ T_g2c @ T_p2g

    # 시각화용 좌표축 생성
    grasp_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    grasp_axis.transform(T_grasp2w)

    # ======================== 4-1) EE 좌표축으로 변환 ========================
    # EE 프레임 기준 grasp pose
    # T_grasp_ee = np.eye(4)
    # T_grasp_ee[:3, :3] = rot_ee
    # T_grasp_ee[:3, 3] = trans_ee


    # EE 프레임 기준 grasp pose 시각화
    grasp_ee_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    grasp_ee_axis.transform(T_c2w @ T_g2c)

    # ======================== 5) Coordinate Frames (world 기준 시각화) ========================
    # World frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # Camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame.transform(T_c2w)
    
    # Robot base frame
    robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    robot_base_frame.transform(T_b2w)
    # robot_base_frame.transform(np.eye(4))
    
    # Robot hand frame (hand_pose_w 기준)
    robot_hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    robot_hand_frame.transform(T_ee2w)

    # IsaacSim EE frame (ee_frame_sensor 기준)
    ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    ee_frame.transform(T_w_ee_isaac)

    diff = T_w_ee_isaac[:3, 3] - T_ee2w[:3, 3]
    diff_norm = np.linalg.norm(diff)

    # # ======================== 6) 최종 시각화 ========================
    if vis:
        # 월드 프레임(+z up)에 맞춘 초기 뷰 → 씬을 위/앞에서 내려다보게
        o3d.visualization.draw_geometries([
            pc_o3d,
            grasp_axis,  # GraspNetAPI 축
            grasp_ee_axis,  # EE 프레임 기준 grasp pose
            *grippers,

        #     # 좌표 프레임들
            # world_frame,
            camera_frame,          # 빨간색: 카메라 frame
            robot_base_frame,      # 초록색: 로봇 base frame
            robot_hand_frame,      # 파란색: 로봇 hand frame (hand_pose_w)
            ee_frame,        # 주황색: IsaacSim EE frame (ee_frame_sensor)
        ], window_name="Grasp result (world)",
           front=[1.0, -0.5, 0.6], up=[0.0, 0.0, 1.0],
           lookat=pc_o3d.get_center(), zoom=0.7
        )

    # T_grasp_base = np.linalg.inv(T_grasp2w) @ T_b2w
    # T_grasp_base = np.linalg.inv(T_grasp2w) @ T_b2w
    T_grasp_base = np.linalg.inv(T_b2w) @ T_grasp2w
    rot_grasp_base = T_grasp_base[:3, :3]
    trans_grasp_base = T_grasp_base[:3, 3] #+ np.array([0.0, 0.0, 0.5])  # z축 방향 offset 추가
        
    return rot_grasp_base, trans_grasp_base, width_top1

