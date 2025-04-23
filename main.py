import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle, cv2
import json, os.path, time
import cProfile
import copy

import gmatch
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


cache = {}


def load(meta_data: util.MetaData, match_data: util.MatchData):
    """load by meta_data and store to match_data"""
    if not os.path.exists(meta_data.pt_path):
        render(meta_data)
    """load model images"""
    if meta_data.pt_id not in cache:
        """load from disk"""
        data = pickle.load(open(meta_data.pt_path, "rb"))
        if len(data) == 3:
            imgs_src, clds_src, poses_src = data
            masks_src = None
        else:
            imgs_src, clds_src, masks_src, poses_src = data
            masks_src = masks_src.astype(np.uint8) * 255
        cache[meta_data.pt_id] = (imgs_src, clds_src, masks_src, poses_src)
    else:
        imgs_src, clds_src, masks_src, poses_src = cache[meta_data.pt_id]

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

    if len(matches) < 3:
        print("Warning: Pose estimation failed, not enough matches")
        match_data.mat_m2c = np.eye(4)
        return

    uv_src = match_data.uvs_src[idx][matches[:, 0]]
    uv_dst = match_data.uv_dst[matches[:, 1]]
    clds_src, masks_src, poses_src = match_data.clds_src, match_data.masks_src, match_data.poses_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

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
    pcd_src, pcd_dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    for i in range(len(clds_src)):
        pts = util.transform(clds_src[i][masks_src[i] != 0], poses_src[i])
        pcd_src.points.extend(o3d.utility.Vector3dVector(pts.reshape(-1, 3)))
    pcd_dst.points = o3d.utility.Vector3dVector(cld_dst[mask_dst != 0].reshape(-1, 3))

    """ refine with icp """
    pcd_src = pcd_src.voxel_down_sample(voxel_size=0.002)
    rlt = o3d.pipelines.registration.registration_icp(pcd_src, pcd_dst, 0.01, mat_m2c)
    mat_m2c = rlt.transformation

    """ visualization """
    # pcd_src.paint_uniform_color([1, 0, 0])
    # pcd_src.transform(mat_m2c)
    # pcd_dst.colors = o3d.utility.Vector3dVector(img_dst[mask_dst != 0].reshape(-1, 3) / 255)
    # o3d.visualization.draw_geometries([pcd_src, pcd_dst], lookat=[0, 0, 1], front=[0, 0, -1], up=[0, -1, 0], zoom=1)

    """ store result """
    # print(f"model in camera, pos: \n{mat_m2c}")
    match_data.mat_m2c = mat_m2c
    return


def result2record(meta_data: util.MetaData, match_data: util.MatchData):
    """record is formatted as bop19 result except that timespan is missing"""
    scene_id, im_id, obj_id = meta_data.scene_id, meta_data.img_id, meta_data.pt_id
    score = len(match_data.matches_list[match_data.idx_best])
    ## Note: convert `t` to mm, leave `R` as it is for it has no unit
    R, t = match_data.mat_m2c[:3, :3], match_data.mat_m2c[:3, 3] * 1000
    R = " ".join(map(lambda x: f"{x:.6f}", R.flatten().tolist()))
    t = " ".join(map(lambda x: f"{x:.6f}", t.flatten().tolist()))
    return [str(scene_id), str(im_id), str(obj_id), str(score), R, t]


def process_img(meta_data: util.MetaData, match_data: util.MatchData, targets):
    """targets: a list of `target` where `target` is (mask_id, scene_id, img_id, objs_id), dtype=(int, int, int, List[int])
    meta_data, match_data: cache assigned to the function
    """
    t0 = time.time()
    record_list = []
    for target in targets:
        mask_id, scene_id, img_id, obj_ids = target
        print(f"scene: {scene_id}, img: {img_id}, mask: {mask_id}")
        match_data_list = []
        ## for each possible obj_id, match it with the scene
        for obj_id in obj_ids:
            meta_data.init(pt_id=obj_id, scene_id=scene_id, img_id=img_id, mask_id=mask_id)
            load(meta_data, match_data)
            gmatch.Match(match_data, meta_data.pt_id)
            print(f"\tobj: {meta_data.pt_id}, len: {len(match_data.matches_list[match_data.idx_best])}")
            match_data_list.append(copy.copy(match_data))
        ## take the object with the most matches
        k = max(enumerate(match_data_list), key=lambda x: len(x[1].matches_list[x[1].idx_best]))[0]
        match_data = match_data_list[k]
        meta_data.init(pt_id=obj_ids[k], scene_id=scene_id, img_id=img_id, mask_id=mask_id)
        solve(match_data)
        record_list.append(result2record(meta_data, match_data))
    timespan = time.time() - t0
    return [f'{", ".join(rec)}, {timespan:.2f}\n' for rec in record_list]


if __name__ == "__main__":
    meta_data = util.MetaData(proj_path=os.path.dirname(os.path.abspath(__file__)), dataset="hope")
    match_data = util.MatchData()

    meta_data.init(pt_id=15, scene_id=1, img_id=2, mask_id=1)
    load(meta_data, match_data)
    t0 = time.time()
    gmatch.Match(match_data)
    print(f"match time: {time.time() - t0:.3f}")
    print(f"obj: {meta_data.pt_id}, len: {len(match_data.matches_list[match_data.idx_best])}")
    solve(match_data)
    exit()

    """ bop19 test set """
    with open("targets_manual_label.json", "r") as f:
        content = json.load(f)

    img_id_last, scene_id_last = None, None
    num_dup = 0
    objs_id = []
    targets = []
    targets_list = []
    ## Obs1: mask_id starts from 0
    ## Obs2: in test_targets_bop19.json, the order of obj_id is just the same as mask file suffix order (aka, mask_id, here)
    for _, line in enumerate(content):
        if img_id_last is None:
            img_id_last = line["im_id"]
        if line["im_id"] != img_id_last:
            n = len(targets)
            targets += [(mask_id, scene_id_last, img_id_last, objs_id) for mask_id in range(n, n + num_dup)]
            targets_list.append(targets)
            num_dup = 0
            objs_id = []
            targets = []
        ## instance count > 1, add it to candidates `objs_id`
        if line["inst_count"] > 1:
            num_dup += line["inst_count"] - 1
            objs_id.append(line["obj_id"])
        targets.append((len(targets), line["scene_id"], line["im_id"], [line["obj_id"]]))

        img_id_last = line["im_id"]
        scene_id_last = line["scene_id"]

    print("all images: ", len(targets_list))

    with open("result.csv", "w") as f:
        for targets in targets_list:
            results = process_img(meta_data, match_data, targets)
            f.writelines(results)
            f.flush()
