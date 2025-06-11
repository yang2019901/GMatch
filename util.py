"""Provides core data structure `MatchData` for GMatch, and utility functions useful in our pose estimation pipeline,
e.g., CAD-related and point cloud processing, visualization.
"""

import numpy as np
import open3d as o3d
import pickle
import os.path
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import patches, collections
from scipy.spatial.transform import Rotation


@dataclass
class MatchData:
    """Data structure for matching keypoints and estimating pose."""

    ## input data for GMatch
    imgs_src: list
    clds_src: list
    masks_src: list
    poses_src: list
    img_dst: np.ndarray
    cld_dst: np.ndarray
    mask_dst: np.ndarray

    ## GMatch result
    matches_list = None  # list of matches, see gmatch.Match
    cost_list = None  # list of distance matrix cost, ranging 0-1, see gmatch.Match
    uvs_src = None  # keypoints extracted from each source image
    uv_dst = None  # keypoints extracted from the destination image
    idx_best = None  # index of the best matches (longest)
    mat_m2c = None  # model to camera transformation matrix, 4x4


def Solve(match_data: MatchData):
    """Solve correspondence with the best match in match_data."""

    ii = match_data.idx_best
    matches = match_data.matches_list[ii]

    ## to solve pose from 3d-3d correspondence, we need at least 3 matches
    if len(matches) < 3:
        print("Warning: Pose estimation failed, not enough matches")
        match_data.mat_m2c = np.eye(4)
        return

    ## alias for src and dst data
    clds_src, masks_src, poses_src = match_data.clds_src, match_data.masks_src, match_data.poses_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

    ## get 3d keypoints from correspondence
    uv1_m = match_data.uvs_src[ii][matches[:, 0]]
    uv2_m = match_data.uv_dst[matches[:, 1]]
    pts1_m = clds_src[ii][uv1_m[:, 1], uv1_m[:, 0], :]
    pts2_m = cld_dst[uv2_m[:, 1], uv2_m[:, 0], :]
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1_m)
    pcd2.points = o3d.utility.Vector3dVector(pts2_m)
    corres = o3d.utility.Vector2iVector([[i, i] for i in range(len(uv1_m))])

    ## solve correspondence with Kabsch algorithm
    estim = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    mat_v2c = estim.compute_transformation(pcd1, pcd2, corres)

    ## get SE(3) matrix from view to model.
    mat_v2m = pose2mat(poses_src[ii])

    ## get SE(3) matrix from model to camera, aka the estimated pose of the object w.r.t. scene camera coordinate system.
    match_data.mat_m2c = mat_v2c @ np.linalg.inv(mat_v2m)


def Refine(match_data: MatchData, voxel_size=0.01):
    """Refine the pose estimation with ICP. match_data.mat_m2c is used as the initial pose and will be written in place.

    - voxel_size: the voxel size for downsampling point cloud. unit: meter. Smaller value makes the refinement more accurate but slower.

    Note: Tune `voxel_size` for your model and data. unit: meter.
    """

    ## alias for src and dst data
    clds_src, cld_dst = match_data.clds_src, match_data.cld_dst
    imgs_src, img_dst = match_data.imgs_src, match_data.img_dst
    masks_src, mask_dst = match_data.masks_src, match_data.mask_dst
    poses_src = match_data.poses_src
    mat_m2c = match_data.mat_m2c

    ## create point cloud
    pcd_src, pcd_dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    for i in range(len(clds_src)):
        pts = transform(clds_src[i][masks_src[i] != 0], poses_src[i])
        pcd_src.points.extend(o3d.utility.Vector3dVector(pts.reshape(-1, 3)))
    pcd_dst.points = o3d.utility.Vector3dVector(cld_dst[mask_dst != 0].reshape(-1, 3))

    ## downsample point cloud and compute normals
    pcd_src_down = pcd_src.voxel_down_sample(voxel_size)
    pcd_dst_down = pcd_dst.voxel_down_sample(voxel_size)

    ## refine with colored icp
    rlt = o3d.pipelines.registration.registration_icp(pcd_src_down, pcd_dst_down, 5 * voxel_size, mat_m2c)
    mat_m2c = rlt.transformation

    ## write result in place
    match_data.mat_m2c = mat_m2c


def pose2mat(pose):
    """pose: [[x, y, z], [qx, qy, qz, qw]], M: 4x4"""
    M = np.eye(4)
    M[:3, 3] = np.array(pose[0]).flatten()
    M[:3, :3] = Rotation.from_quat(pose[1]).as_matrix()
    return M


def mat2pose(M):
    """pose: [[x, y, z], [qx, qy, qz, qw]], M: 4x4"""
    return [np.copy(M[:3, 3]), Rotation.from_matrix(np.copy(M[:3, :3])).as_quat()]


def transform(pts, pose):
    """pts: (..., 3), pose: [[x, y, z], [qx, qy, qz, qw]], return: (..., 3)"""
    _pts = np.array(pts)
    M = pose2mat(pose)
    return _pts @ M[:3, :3].T + M[:3, 3]


def depth2cld(depth, intrisic):
    """Convert depth to point cloud."""
    intrin = np.array(intrisic, dtype=np.float32).reshape(3, 3)
    z = np.asarray(depth, np.float32)
    u, v = np.meshgrid(range(z.shape[1]), range(z.shape[0]))
    uv = np.stack((u, v, np.ones_like(u)), axis=-1, dtype=np.float32)
    pts = np.linalg.inv(intrin) @ uv.reshape(-1, 3).T * z.reshape(-1)
    return pts.T.reshape(z.shape[0], z.shape[1], 3)


def vis_cld(clds, colors=None, poses=None):
    """Transform and visualize point clouds."""
    _clds = [transform(cld, pose).reshape(-1, 3) for cld, pose in zip(clds, poses)] if poses is not None else clds

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(_clds, axis=0))
    if colors is not None:
        _colors = [np.array(color).reshape(-1, 3) / 255.0 for color in colors]
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(_colors, axis=0))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis], lookat=[0, 0, 1], front=[0, 0, -1], up=[0, -1, 0], zoom=0.2)
    return


def get_snapshots(mesh):
    """Render the mesh from six views and return snapshots.

    - mesh: o3d.geometry.TriangleMesh
    - snapshots: [(rgb, cld, mask, M_ex), ...]
    - rgb: (H, W, 3), 0~1
    - cld: (H, W, 3), meters
    - mask: (H, W), bool
    - M_ex: (4, 4)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)

    ## 6-axis sampling
    camera_params = [
        {"front": [1, 0, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # x+
        {"front": [-1, 0, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # x-
        {"front": [0, 1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # y+
        {"front": [0, -1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # y-
        {"front": [0, 0, 1], "lookat": [0, 0, 0], "up": [0, 1, 0]},  # z+
        {"front": [0, 0, -1], "lookat": [0, 0, 0], "up": [0, 1, 0]},  # z-
    ]

    snapshots = []

    for params in camera_params:
        ctr = vis.get_view_control()
        ctr.set_lookat(params["lookat"])
        ctr.set_front(params["front"])
        ctr.set_up(params["up"])
        ctr.set_zoom(1.2)
        vis.poll_events()
        vis.update_renderer()

        # Capture depth and color images
        _depth = vis.capture_depth_float_buffer(True)
        _rgb = vis.capture_screen_float_buffer(True)

        # Convert to Open3D RGBD image
        depth = np.asarray(_depth)  # (H, W)
        rgb = np.asarray(_rgb)  # (H, W, 3)
        cam_info = ctr.convert_to_pinhole_camera_parameters()
        M_ex, M_in = cam_info.extrinsic, cam_info.intrinsic.intrinsic_matrix
        # Convert depth to point cloud
        H, W = depth.shape
        depth = depth.reshape(-1)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u, v = u.reshape(-1), v.reshape(-1)
        z = depth
        x = (u - M_in[0, 2]) * z / M_in[0, 0]
        y = (v - M_in[1, 2]) * z / M_in[1, 1]
        cld = np.stack([x, y, z], axis=1).reshape(H, W, 3)
        snapshots.append((rgb, cld, depth.reshape(H, W) != 0, np.linalg.inv(M_ex)))

    vis.destroy_window()
    return snapshots


def save_snapshots(snapshots, path):
    """Save snapshots to a file.

    - imgs: (N, H, W, 3), 0~255, uint8
    - clds: (N, H, W, 3), meters, float32
    - masks: (N, H, W), bool
    - poses: [(pos, quat), ...]
    """
    rgbs, clds, masks, M_ex_list = zip(*snapshots)
    imgs = np.asarray(np.stack(rgbs) * 255, dtype=np.uint8)
    masks = np.asarray(np.stack(masks), dtype=bool)
    clds = np.asarray(np.stack(clds), dtype=np.float32)
    poses = [mat2pose(M_ex) for M_ex in M_ex_list]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((imgs, clds, masks, poses), f)


def vis_snapshots(snapshots):
    """Visualize snapshots.

    - snapshots: [(rgb, cld, mask, M_ex), ...]
    - rgb: (H, W, 3), 0~1 or 0~255
    - cld: (H, W, 3), meters
    - mask: (H, W), all non-zero pixel will be displayed
    - M_ex: (4, 4), camera extrinsic matrix
    """

    clds = []
    for rgb, cld, mask, M_pose in snapshots:
        pcd = o3d.geometry.PointCloud()
        if rgb.dtype == np.uint8:
            rgb = rgb / 255.0
        if mask is not None:
            cld = cld[mask != 0]
            rgb = rgb[mask != 0]
        pcd.points = o3d.utility.Vector3dVector(cld.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        pcd.transform(M_pose)
        axis_mesh.transform(M_pose)
        clds.append(pcd)
        clds.append(axis_mesh)
    o3d.visualization.draw_geometries(clds)


def plot_matches(img1, img2, uv1, uv2):
    """Plot matches between two images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax2.imshow(img2)
    for pt1, pt2 in zip(uv1, uv2):
        cir1 = patches.Circle(pt1, 3, color="red", fill=False)
        cir2 = patches.Circle(pt2, 3, color="red", fill=False)
        ax1.add_patch(cir1)
        ax2.add_patch(cir2)
        l = patches.ConnectionPatch(
            xyA=pt1,
            xyB=pt2,
            axesA=ax1,
            axesB=ax2,
            coordsA="data",
            coordsB="data",
            color="green",
        )
        fig.add_artist(l)
    fig.suptitle(f"matches: {len(uv1)}")
    fig.tight_layout()
    plt.show()
    return


def plot_keypoints(img1, img2, uv1, uv2, Mf12, thresh_feat):
    """Plot keypoints between two images, and mark locally matched points interactively.
    Useful for try out the appropriate thresh_feat for Mf12.
    """
    idx1 = -1  # index of selected keypoint in img1
    alts = []  # indices of locally matched keypoints in img2
    idx2 = -1  # index of selected keypoint in img2

    R, G, B = [1, 0, 0, 1], [0, 0.5, 0, 1], [0, 0, 1, 1]

    def update_color():
        nonlocal idx1, alts, idx2, clr1, clr2, scat1, scat2
        clr1[:] = G
        clr2[:] = G
        if idx1 != -1:
            clr1[idx1] = R
            clr2[alts] = R
        if idx2 != -1:
            clr2[idx2] = B
        scat1.set_facecolor(clr1)
        scat2.set_facecolor(clr2)

    def on_click_src(event):
        nonlocal idx1, alts, idx2
        if event.inaxes != ax1:
            return
        x, y = int(event.xdata), int(event.ydata)
        distances = np.linalg.norm(uv1 - np.array([x, y]), axis=1)
        ## Note: due to np.int in Match, there may be multiple idx2
        ## so we take the last one, which is drawed last s.t. we can see the color
        idx1 = np.argwhere(distances == distances.min())[-1].item()
        ax1.set_title(f"keypoints: {len(uv1)}, selected: {idx1}")
        dists = Mf12[idx1]
        alts = np.where(dists < thresh_feat)[0]
        if idx2 != -1:
            fig.suptitle(f"feature distance: {Mf12[idx1, idx2]:.3f}")
        update_color()
        fig.canvas.draw()

    def on_click_dst(event):
        nonlocal idx1, idx2, alts
        if event.inaxes != ax2:
            return
        ## set idx2
        x, y = int(event.xdata), int(event.ydata)
        distances = np.linalg.norm(uv2 - np.array([x, y]), axis=1)
        ## Note: due to np.int in Match, there may be multiple idx2
        ## so we take the last one, which is drawed last s.t. we can see the color
        idx2 = np.argwhere(distances == distances.min())[-1].item()
        ax2.set_title(f"keypoints: {len(uv2)}, selected: {idx2}")
        if idx1 != -1:
            fig.suptitle(f"feature distance: {Mf12[idx1, idx2]:.3f}")
        update_color()
        fig.canvas.draw()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax2.imshow(img2)
    clr1 = np.repeat([G], len(uv1), axis=0)
    clr2 = np.repeat([G], len(uv2), axis=0)
    scat1: collections.PathCollection = ax1.scatter(uv1[:, 0], uv1[:, 1], c=clr1, s=3)
    scat2: collections.PathCollection = ax2.scatter(uv2[:, 0], uv2[:, 1], c=clr2, s=3)
    ax1.set_title(f"keypoints: {len(uv1)}")
    ax2.set_title(f"keypoints: {len(uv2)}")
    ax1.axis("off")
    ax2.axis("off")
    fig.suptitle(f"feature distance: ")
    fig.tight_layout()
    fig.canvas.mpl_connect("button_press_event", on_click_src)
    fig.canvas.mpl_connect("button_press_event", on_click_dst)
    plt.show()
