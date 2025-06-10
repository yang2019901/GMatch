"""Demonstrate how to use GMatch-SIFT to estimate the object pose from a RGBD image with its CAD model."""

import cv2
import numpy as np
import open3d as o3d

import gmatch, util


""" Load source images """
## get o3d mesh from CAD model
mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("sugar_box/mesh.obj", enable_post_processing=True)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

## render from six views and save as 'snapshot' (RGB, point_cloud, mask and camera_pose)
snapshots = util.get_snapshots(mesh)

## visualize the snapshots
# util.vis_snapshots(snapshots)

## parse snapshots and convert imgs/clds/masks/poses to numpy arrays
imgs_src, clds_src, masks_src, M_ex_list = zip(*snapshots)
imgs_src = np.asarray(np.stack(imgs_src) * 255, dtype=np.uint8)
masks_src = np.stack(masks_src).astype(np.uint8) * 255
clds_src = np.asarray(np.stack(clds_src), dtype=np.float32)
poses_src = [util.mat2pose(M_ex) for M_ex in M_ex_list]


""" Load the target image """
## load dst RGB, depth and mask. also, check RGB-D camera info for intrinsic
img_dst = cv2.imread("sugar_box/rgb.png", cv2.IMREAD_COLOR_RGB)
depth_dst = cv2.imread("sugar_box/depth.png", cv2.IMREAD_UNCHANGED)
mask_dst = cv2.imread("sugar_box/mask.png", cv2.IMREAD_UNCHANGED)
cam_intrin = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])

## convert depth to point cloud
cld_dst = util.depth2cld(depth_dst * 0.0001, cam_intrin)

## crop dst RGB and point cloud to the bounding box of the mask
ind = np.argwhere(mask_dst != 0)
r1, c1 = ind.min(axis=0)
r2, c2 = ind.max(axis=0)
mask_dst[r1 : r2 + 1, c1 : c2 + 1] = 255
img_dst = img_dst[r1 : r2 + 1, c1 : c2 + 1]
mask_dst = mask_dst[r1 : r2 + 1, c1 : c2 + 1]
cld_dst = cld_dst[r1 : r2 + 1, c1 : c2 + 1]


""" Preprocessing: denoise RGB images of YCB-Video """
imgs_src = [cv2.GaussianBlur(img, (5, 5), 0) for img in imgs_src]
img_dst = cv2.GaussianBlur(img_dst, (5, 5), 0)


""" GMatch-SIFT """
## put data into util.MatchData
match_data = util.MatchData(
    imgs_src=imgs_src,
    clds_src=clds_src,
    masks_src=masks_src,
    poses_src=poses_src,
    img_dst=img_dst,
    cld_dst=cld_dst,
    mask_dst=mask_dst,
)

## GMatch matching under SIFT description
gmatch.Match(match_data, cache_id="sugar_box")

## get the best matches and convert them to matched 3d points
i = match_data.idx_best
matches = match_data.matches_list[i]
uv1 = match_data.uvs_src[i]
uv2 = match_data.uv_dst
uv1_m = uv1[matches[:, 0]]
uv2_m = uv2[matches[:, 1]]
pts1_m = match_data.clds_src[i][uv1_m[:, 1], uv1_m[:, 0]]
pts2_m = match_data.cld_dst[uv2_m[:, 1], uv2_m[:, 0]]

## visualize the matches
# util.plot_matches(match_data.imgs_src[i], match_data.img_dst, uv1_m, uv2_m)


""" Solve the pose """
util.Solve(match_data)

## ICP refinement (optional)
util.Refine(match_data)

## get SE(3) matrix from model to camera, aka the estimated pose of the object w.r.t. scene camera coordinate system.
mat_m2c = match_data.mat_m2c
print("Estimated pose (model to camera):\n", mat_m2c)


""" Visualization of Accuracy """
pcd_model, pcd_scene = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

## transform the source point clouds to the model coordinate system and concatenate them.
for i in range(len(clds_src)):
    pts = util.transform(clds_src[i][masks_src[i] != 0], poses_src[i])
    pcd_model.points.extend(o3d.utility.Vector3dVector(pts.reshape(-1, 3)))

## downsample the model point cloudl, paint it red, and transform it to the camera coordinate system.
pcd_model = pcd_model.voxel_down_sample(voxel_size=0.002)
pcd_model.paint_uniform_color([1, 0, 0])
pcd_model.transform(mat_m2c)

pcd_scene.points = o3d.utility.Vector3dVector(cld_dst[mask_dst != 0].reshape(-1, 3))
pcd_scene.colors = o3d.utility.Vector3dVector(img_dst[mask_dst != 0].reshape(-1, 3) / 255)

## visualize the model and scene point clouds, the more accurate the estimated pose is, the better the model fits the scene.
o3d.visualization.draw_geometries([pcd_model, pcd_scene], lookat=[0, 0, 1], front=[0, 0, -1], up=[0, -1, 0], zoom=1)
