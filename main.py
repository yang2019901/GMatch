import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle, cv2
import json, os.path, time
import cProfile
import copy, sys, logging

import gmatch
import util


cache = {}

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[%(levelname)5s] [%(funcName)s] %(message)s"))
logger.addHandler(h)
logger.propagate = False


def render(meta_data):
    """render model to snapshots and save to pt_path"""
    mesh = util.load_ply(meta_data.model_path)
    """ <Check Unit> calc diameter of the model to compare with 'models/models_info.json' """
    pts = np.asarray(mesh.vertices)
    bbox = (np.max(pts, axis=0) - np.min(pts, axis=0)) * 1000
    # axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([mesh, axis_mesh])
    snapshots = util.get_snapshots(mesh)
    util.vis_snapshots(snapshots)
    util.save_snapshots(snapshots, meta_data.pt_path)
    logger.info(f"saved to {meta_data.pt_path}, bbox: {bbox} mm")


def load(meta_data: util.MetaData, match_data: util.MatchData):
    """load by meta_data and store to match_data"""
    if not os.path.exists(meta_data.pt_path):
        render(meta_data)

    blur = meta_data.dataset == "ycbv"

    """load model images"""
    if meta_data.pt_id not in cache:
        """load from disk"""
        with open(meta_data.pt_path, "rb") as f:
            data = pickle.load(f)
        if len(data) == 3:
            imgs_src, clds_src, poses_src = data
            masks_src = None
        else:
            imgs_src, clds_src, masks_src, poses_src = data
            masks_src = masks_src.astype(np.uint8) * 255

        if blur:
            imgs_src = [cv2.GaussianBlur(img, (5, 5), 0) for img in imgs_src]

        cache[meta_data.pt_id] = (imgs_src, clds_src, masks_src, poses_src)
    else:
        imgs_src, clds_src, masks_src, poses_src = cache[meta_data.pt_id]

    """ load scene image """
    img_dst = cv2.imread(meta_data.img_path, cv2.IMREAD_COLOR_RGB)
    depth_dst = cv2.imread(meta_data.depth_path, cv2.IMREAD_UNCHANGED)
    mask_dst = cv2.imread(meta_data.mask_path, cv2.IMREAD_UNCHANGED)
    cld_dst = util.depth2cld(depth_dst * (meta_data.depth_scale * 0.001), meta_data.cam_intrin)

    """ get bbox from mask_dst (orb/sift can work well with bbox, no need for segmentation) """
    ind = np.argwhere(mask_dst != 0)
    r1, c1 = ind.min(axis=0)
    r2, c2 = ind.max(axis=0)
    mask_dst[r1 : r2 + 1, c1 : c2 + 1] = 255
    """ crop img_dst (and cld_dst) """
    img_dst = img_dst[r1 : r2 + 1, c1 : c2 + 1]
    mask_dst = mask_dst[r1 : r2 + 1, c1 : c2 + 1]
    cld_dst = cld_dst[r1 : r2 + 1, c1 : c2 + 1]

    if blur:
        img_dst = cv2.GaussianBlur(img_dst, (5, 5), 0)

    # util.vis_cld(cld_dst, img_dst)
    """ store data to match_data """
    match_data.imgs_src = imgs_src
    match_data.clds_src = clds_src
    match_data.masks_src = masks_src
    match_data.poses_src = poses_src
    match_data.img_dst = img_dst
    match_data.cld_dst = cld_dst
    match_data.mask_dst = mask_dst


def solve(match_data: util.MatchData, icp_refine=False):
    """solve correspondence with the best match in match_data"""

    matches = match_data.matches
    pt_src = match_data.pt_src
    pt_dst = match_data.pt_dst

    clds_src, masks_src, poses_src = match_data.clds_src, match_data.masks_src, match_data.poses_src
    cld_dst, mask_dst = match_data.cld_dst, match_data.mask_dst

    if len(matches) < 3:
        logging.warning(f"Too few ({len(matches)}) matches found, switch to point cloud method.")
        # create point cloud
        voxel_sz = 0.005
        pcd_src, pcd_dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        for i in range(len(clds_src)):
            pt = util.transform(clds_src[i][masks_src[i] != 0], poses_src[i])
            pcd_src.points.extend(o3d.utility.Vector3dVector(pt.reshape(-1, 3)))
        pcd_src = pcd_src.voxel_down_sample(voxel_size=voxel_sz)
        pcd_dst.points = o3d.utility.Vector3dVector(cld_dst[mask_dst != 0].reshape(-1, 3))
        pcd_dst = pcd_dst.voxel_down_sample(voxel_size=voxel_sz)

        pcd_dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel_sz, max_nn=30))
        pcd_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel_sz, max_nn=30))

        fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_src, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_sz, max_nn=100)
        )

        fpfh_dst = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_dst, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_sz, max_nn=100)
        )

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_src, pcd_dst, fpfh_src, fpfh_dst, True, max_correspondence_distance=5 * voxel_sz
        )
        match_data.mat_m2c = result.transformation
        return

    # solve correspondence
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt_src[matches[:, 0], :])
    pcd2.points = o3d.utility.Vector3dVector(pt_dst[matches[:, 1], :])
    corres = o3d.utility.Vector2iVector([[i, i] for i in range(len(matches))])
    # Kabsch
    estim = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    mat_m2c = estim.compute_transformation(pcd1, pcd2, corres)

    if icp_refine:
        # create point cloud
        pcd_src, pcd_dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        for i in range(len(clds_src)):
            pt = util.transform(clds_src[i][masks_src[i] != 0], poses_src[i])
            pcd_src.points.extend(o3d.utility.Vector3dVector(pt.reshape(-1, 3)))
        pcd_src = pcd_src.voxel_down_sample(voxel_size=0.002)
        pcd_dst.points = o3d.utility.Vector3dVector(cld_dst[mask_dst != 0].reshape(-1, 3))
        # refine with icp
        rlt = o3d.pipelines.registration.registration_icp(pcd_src, pcd_dst, 0.01, mat_m2c)
        mat_m2c = rlt.transformation

    # store result
    logger.debug(f"model in camera, pos: \n{mat_m2c}")
    match_data.mat_m2c = mat_m2c


def result2record(meta_data: util.MetaData, match_data: util.MatchData):
    """record is formatted as bop19 result except that timespan is missing"""
    scene_id, im_id, obj_id = meta_data.scene_id, meta_data.img_id, meta_data.pt_id
    score = len(match_data.matches)
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
        mask_id, scene_id, img_id, obj_id = target
        logger.debug(f"scene: {scene_id}, img: {img_id}, mask: {mask_id}")
        meta_data.init(pt_id=obj_id, scene_id=scene_id, img_id=img_id, mask_id=mask_id)
        load(meta_data, match_data)
        gmatch.Match(match_data, meta_data.pt_id, debug=-1)
        logger.debug(f"\tobj: {meta_data.pt_id}, len: {len(match_data.matches)}")
        solve(match_data, icp_refine=True)
        record_list.append(result2record(meta_data, match_data))
    timespan = time.time() - t0
    return [f'{", ".join(rec)}, {timespan:.2f}\n' for rec in record_list]


def run_per_dataset(dataset_name, targets_path, result_path):
    meta_data = util.MetaData(proj_path=os.path.dirname(os.path.abspath(__file__)), dataset=dataset_name)
    match_data = util.MatchData()

    """ bop19 test set """
    with open(targets_path, "r") as f:
        content = json.load(f)

    img_id_last, scene_id_last = None, None
    num_dup = 0
    objs_id = []
    targets = []
    targets_list = []

    for _, line in enumerate(content):
        if img_id_last is None:
            img_id_last = line["im_id"]
        elif line["im_id"] != img_id_last or line["scene_id"] != scene_id_last:
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

    logger.info("all images: ", len(targets_list))

    with open(result_path, "w") as f:
        for targets in targets_list:
            results = process_img(meta_data, match_data, targets)
            f.writelines(results)
            f.flush()


def run_ycbv_targets(dataset_name, scenes, debug, icp_refine):
    """test perception stability (precision, run-time, etc) on video
    :param dataset_name: e.g. 'ycbv'
    :param scenes: a list of (scene_id, pt_id, mask_id)
    """
    meta_data = util.MetaData(proj_path=os.path.dirname(os.path.abspath(__file__)), dataset=dataset_name)
    match_data = util.MatchData()

    for scene_id, pt_id, mask_id in scenes:
        logger.info(f"Processing: scene={scene_id}, obj={pt_id}, mask={mask_id}")
        img_folder = os.path.join(meta_data.proj_path, f"bop_data/{dataset_name}/test/{str(scene_id).zfill(6)}/rgb")
        with open(f"bop_data/{dataset_name}/test/{str(scene_id).zfill(6)}/scene_gt.json", "r") as f:
            content = json.load(f)
        files = os.listdir(img_folder)
        imgs_id = [int(f.split(".")[0]) for f in files]
        imgs_id.sort()
        result = []
        for img_id in imgs_id:
            meta_data.init(pt_id=pt_id, scene_id=scene_id, img_id=img_id, mask_id=mask_id)
            load(meta_data, match_data)

            t0 = time.time()
            gmatch.Match(match_data, cache_id=meta_data.pt_id, debug=debug)
            solve(match_data, icp_refine=icp_refine)
            dt = time.time() - t0

            M_pred = match_data.mat_m2c

            M = np.eye(4)
            gt = next((x for x in content[str(img_id)] if x["obj_id"] == pt_id))
            M[:3, :3] = np.array(gt["cam_R_m2c"]).reshape(3, 3)
            M[:3, 3] = np.array(gt["cam_t_m2c"]) * 0.001

            M_err = np.linalg.inv(M) @ M_pred

            dist_err = np.linalg.norm(M_err[:3, 3])
            ang_err = np.arccos((np.trace(M_err[:3, :3]) - 1) / 2)
            logger.info(
                f"img_id: {meta_data.img_id:>3}, len: {len(match_data.matches):>3}, dist_err: {dist_err*1000:>5.1f} mm, ang_err: {np.rad2deg(ang_err):>5.1f} deg, dt: {dt*1000:.0f} ms"
            )
            result.append(f"{meta_data.img_id}, {dist_err*1000:.1f}, {np.rad2deg(ang_err):.1f}, {dt*1000:.1f}\n")

        # with open(f"result_ycbv_{scene_id}_{pt_id}.csv", "w") as f:
        #     f.writelines(result)


def run_per_object(dataset_name, scene_id, img_id, obj_id, mask_id, debug):
    meta_data = util.MetaData(proj_path=os.path.dirname(os.path.abspath(__file__)), dataset=dataset_name)
    match_data = util.MatchData()
    meta_data.init(scene_id=scene_id, img_id=img_id, pt_id=obj_id, mask_id=mask_id)
    load(meta_data, match_data)
    t0 = time.time()
    gmatch.Match(match_data, debug=debug)
    logger.info(f"match time: {time.time() - t0:.3f}")
    logger.info(f"best loss: {match_data.cost}")
    logger.info(f"obj: {meta_data.pt_id}, len: {len(match_data.matches)}")
    solve(match_data)


if __name__ == "__main__":
    # run_per_object("ycbv", 54, 22, 2, 0, debug=-1)
    run_ycbv_targets("ycbv", [(54, 2, 0)], debug=1, icp_refine=False)
    # run_per_dataset("hope", "./targets_manual_label.json", "result_hope-test.csv")
    # run_per_dataset("ycbv", "./bop_data/ycbv/test_targets_bop19.json", "result_ycbv-test.csv")
