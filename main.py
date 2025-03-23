import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle, cv2
import json, os, time
import cProfile

import FeatMatch
import util


def render(meta_data):
    """render model to snapshots and save to pt_path"""
    mesh = util.load_ply(meta_data.model_path)
    """ <Check Unit> calc diameter of the model to compare with 'models/models_info.json' """
    pts = np.asarray(mesh.vertices)
    bbox = (np.max(pts, axis=0) - np.min(pts, axis=0)) * 1000
    print(f"saved to {meta_data.pt_path}, bbox: {bbox} mm")
    # o3d.visualization.draw_geometries([mesh])
    snapshots = util.get_snapshots(mesh)
    util.vis_snapshots(snapshots)
    util.save_snapshots(snapshots, meta_data.pt_path)


def load(meta_data, match_data):
    """load by meta_data and store to match_data"""
    if not os.path.exists(meta_data.pt_path):
        render(meta_data)
    """load model images"""
    data = pickle.load(open(meta_data.pt_path, "rb"))
    if len(data) == 3:
        imgs_src, clds_src, poses_src = data
        masks_src = None
    else:
        imgs_src, clds_src, masks_src, poses_src = data
        masks_src = masks_src.astype(np.uint8) * 255

    """ load scene image """
    img_dst = cv2.imread(meta_data.img_path, cv2.IMREAD_COLOR_RGB)
    depth_dst = cv2.imread(meta_data.depth_path, cv2.IMREAD_UNCHANGED)
    mask_dst = cv2.imread(meta_data.mask_path, cv2.IMREAD_UNCHANGED)

    cld_dst = util.depth2cld(depth_dst * 0.001 * meta_data.depth_scale, meta_data.cam_intrin)
    # util.vis_cld(cld_dst, img_dst)
    """ store data to match_data """
    match_data.imgs_src = imgs_src
    match_data.clds_src = clds_src
    match_data.masks_src = masks_src
    match_data.poses_src = poses_src
    match_data.img_dst = img_dst
    match_data.cld_dst = cld_dst
    match_data.mask_dst = mask_dst


def solve(match_data):
    """solve correspondence with the best match in match_data"""
    idx = match_data.idx_best
    matches = match_data.matches_list[idx]
    uv_src = match_data.uvs_src[idx][matches[:, 0]]
    uv_dst = match_data.uv_dst[matches[:, 1]]
    clds_src, masks_src, poses_src = match_data.clds_src, match_data.masks_src, match_data.poses_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

    if len(uv_src) < 3:
        match_data.mat_m2c = np.eye(4)

    """ solve correspondence (Note: dont use cv2.PnPRansac, it sucks) """
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(clds_src[idx, uv_src[:, 1], uv_src[:, 0], :])
    pcd2.points = o3d.utility.Vector3dVector(cld_dst[uv_dst[:, 1], uv_dst[:, 0], :])
    corres = np.array([[i, i] for i in range(len(uv_src))])
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd1, pcd2, o3d.utility.Vector2iVector(corres), 0.01
    )
    mat_v2c = result.transformation
    mat_v2m = util.pose2mat(poses_src[idx])
    mat_m2c = mat_v2c @ np.linalg.inv(mat_v2m)

    """ create point cloud """
    # pcd_src, pcd_dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    # for i in range(len(clds_src)):
    #     pts = util.transform(clds_src[i][masks_src[i] != 0], poses_src[i])
    #     pcd_src.points.extend(o3d.utility.Vector3dVector(pts.reshape(-1, 3)))
    # pcd_dst.points = o3d.utility.Vector3dVector(cld_dst[mask_dst != 0].reshape(-1, 3))

    """ refine with icp """
    # pcd_src = pcd_src.voxel_down_sample(voxel_size=0.002)
    # rlt = o3d.pipelines.registration.registration_icp(pcd_src, pcd_dst, 0.01, mat_m2c)
    # mat_m2c = rlt.transformation

    """ visualization """
    # pcd_src.paint_uniform_color([1, 0, 0])
    # pcd_src.transform(mat_m2c)
    # pcd_dst.colors = o3d.utility.Vector3dVector(img_dst[mask_dst != 0].reshape(-1, 3) / 255)
    # o3d.visualization.draw_geometries([pcd_src, pcd_dst], lookat=[0, 0, 1], front=[0, 0, -1], up=[0, -1, 0], zoom=1)

    """ store result """
    match_data.mat_m2c = mat_m2c
    print(f"model in camera, pos: \n{mat_m2c}")


if __name__ == "__main__":
    meta_data = util.MetaData(proj_path="/home/yang2019901/GMatch-ORB", dataset="hope")
    match_data = util.MatchData()

    """ FeatMatch parameters override """
    FeatMatch.N1 = 500
    FeatMatch.N2 = 500
    FeatMatch.N_good = 25
    FeatMatch.D = 24
    FeatMatch.thresh_des = 0.1
    FeatMatch.thresh_loss = 0.08
    FeatMatch.thresh_flip = 0.05

    # meta_data.init(pt_id="3", scene_id="1", img_id="0", mask_id="14")
    # load(meta_data, match_data)
    # cProfile.run("FeatMatch.match_features(match_data)", "a.prof", sort="cumulative")
    # solve(match_data)
    # exit()

    """ bop19 test set """
    with open(f"{meta_data.proj_path}/bop_data/{meta_data.dataset}/test_targets_bop19.json", "r") as f:
        targets = json.load(f)
        obj_list = []
        num_dup = 0
        i = 0
        img_id = None
        for target in targets:
            if img_id is not None and target["im_id"] != img_id:
                print(f"im_id switching: {img_id} -> {target['im_id']}")
                """im_id switching, time to deal with id_list"""
                for j in range(num_dup):
                    mask_id = i + j
                    mds = []  # match_data list
                    print(f"scene: {meta_data.scene_id}, img: {img_id}, mask: {mask_id}")
                    for obj_id in obj_list:
                        meta_data.init(pt_id=obj_id, img_id=img_id, mask_id=mask_id)
                        print(f"\tobj: {meta_data.pt_id}")
                        load(meta_data, match_data)
                        FeatMatch.match_features(match_data)
                        mds.append(match_data)
                    match_data = max(mds, key=lambda x: len(x.matches_list))
                    solve(match_data)
                i, num_dup = 0, 0
                obj_list.clear()

            if target["inst_count"] > 1:
                obj_list.append(target["obj_id"])
                num_dup += target["inst_count"] - 1
            meta_data.init(pt_id=target["obj_id"], scene_id=target["scene_id"], img_id=target["im_id"], mask_id=i)
            print(
                f"obj: {meta_data.pt_id}, scene: {meta_data.scene_id}, img: {meta_data.img_id}, mask: {meta_data.mask_id}"
            )
            load(meta_data, match_data)
            FeatMatch.match_features(match_data)
            solve(match_data)
            img_id = target["im_id"]
            i += 1
