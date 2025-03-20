import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle, cv2
import json, os, time

import FeatMatch
import util


def render():
    # 加载模型
    mesh = util.load_ply(model_path)
    pts = np.asarray(mesh.vertices)
    # calc diameter of the model
    bbox = (np.max(pts, axis=0) - np.min(pts, axis=0)) * 1000
    print(f"saved to {pt_path}, bbox: {bbox} mm")

    # # 显示模型
    # o3d.visualization.draw_geometries([mesh])

    # 拍摄 RGBD 图像
    snapshots = util.get_snapshots(mesh)
    util.vis_snapshots(snapshots)
    util.save_snapshots(snapshots, pt_path)


def main():
    if not os.path.exists(pt_path):
        render()
    """load model images"""
    data = pickle.load(open(pt_path, "rb"))
    if len(data) == 3:
        imgs_src, clds_src, poses_src = data
        masks_src = None
    else:
        imgs_src, clds_src, masks_src, poses_src = data
        masks_src = masks_src.astype(np.uint8) * 255

    """ load scene image """
    img_dst = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    depth_dst = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask_dst = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    cld_dst = util.depth2cld(depth_dst * 0.001 * depth_scale, cam_intrin)
    # util.vis_cld(cld_dst, img_dst)

    """ match features """
    idx, uv_src, uv_dst = FeatMatch.match_features(imgs_src, img_dst, clds_src, cld_dst, masks_src, mask_dst)
    print(f"length of matches: {len(uv_src)}")

    """ solve PnP """
    pts_obj = clds_src[idx, uv_src[:, 1], uv_src[:, 0], :]
    ret, rvec, tvec, inliners = cv2.solvePnPRansac(
        pts_obj, uv_dst.astype(np.float32), np.array(cam_intrin).reshape(3, 3), None
    )
    R, _ = cv2.Rodrigues(rvec)
    rot = util.Rotation.from_matrix(R).as_quat()
    pos = tvec.flatten()
    mat_v2c = util.pose2mat([pos, rot])  # view to camera
    mat_v2m = util.pose2mat(poses_src[idx])
    mat_m2c = mat_v2c @ np.linalg.inv(mat_v2m)

    """ create point cloud """
    pcd_src, pcd_dst = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    for i in range(len(clds_src)):
        pts = util.transform(clds_src[i][masks_src[i] != 0], poses_src[i])
        pcd_src.points.extend(o3d.utility.Vector3dVector(pts.reshape(-1, 3)))
    pcd_src = pcd_src.voxel_down_sample(voxel_size=0.002)
    pcd_dst.points = o3d.utility.Vector3dVector(cld_dst.reshape(-1, 3))
    pcd_src.paint_uniform_color([1, 0, 0])

    """ refine with icp """
    rlt = o3d.pipelines.registration.registration_icp(pcd_src, pcd_dst, 0.01, mat_m2c)
    mat_m2c = rlt.transformation
    print(f"icp result: {rlt}")

    pcd_src.transform(mat_m2c)
    o3d.visualization.draw_geometries([pcd_src, pcd_dst], lookat=[0, 0, 1], front=[0, 0, -1], up=[0, -1, 0], zoom=0.2)
    print(f"model in camera, pos: \n{mat_m2c}")


def set_meta():
    global proj_path, dataset, pt_name, scene_id, img_id, mask_id
    model_name = f"obj_{pt_name.zfill(6)}"
    scene_id = scene_id.zfill(6)
    img_id = img_id.zfill(6)
    mask_id = mask_id.zfill(6)
    model_path = f"{proj_path}/bop_data/{dataset}/models/{model_name}.ply"
    pt_path = f"{proj_path}/{dataset}/{pt_name}.pt"
    img_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/rgb/{img_id}.png"
    depth_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/depth/{img_id}.png"
    mask_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/mask/{img_id}_{mask_id}.png"
    json_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/scene_camera.json"
    with open(json_path, "r") as f:
        s = img_id.lstrip("0")
        img_camera = json.load(f)["0" if s == "" else s]
        cam_intrin = img_camera["cam_K"]
        depth_scale = img_camera["depth_scale"]
    globals().update(locals())


if __name__ == "__main__":
    ## no suffix here
    proj_path = "/home/yang2019901/GMatch-ORB"
    dataset = "hope"
    pt_name = "16"
    scene_id = "1"
    img_id = "1"
    mask_id = "0"

    ## FeatMatch parameters override
    FeatMatch.N1 = 500
    FeatMatch.N2 = 500
    FeatMatch.N_good = 30
    FeatMatch.D = 24
    FeatMatch.thresh_des = 0.1
    FeatMatch.thresh_loss = 0.08
    FeatMatch.thresh_flip = 0.05

    set_meta()
    if not os.path.exists(pt_path):
        render()
    main()
    exit()

    """ bop19 test set """
    with open(f"{proj_path}/bop_data/{dataset}/test_targets_bop19.json", "r") as f:
        targets = json.load(f)
        id_list = []
        num_dup = 0
        i = 0
        img_id = None
        for target in targets:
            if img_id is not None and str(target["im_id"]).zfill(6) != img_id:
                print(f"im_id switching: {img_id} -> {target['im_id']}")
                """im_id switching, time to deal with id_list"""
                for j in range(num_dup):
                    mask_id = str(i + j)
                    for obj_id in id_list:
                        pt_name = str(obj_id)
                        print(f"obj: {pt_name}, scene: {scene_id}, img: {img_id}, mask: {mask_id}")
                        set_meta()
                        main()
                i, num_dup = 0, 0
                id_list.clear()

            if target["inst_count"] > 1:
                id_list.append(target["obj_id"])
                num_dup += target["inst_count"] - 1
            pt_name = str(target["obj_id"])
            img_id = str(target["im_id"])
            scene_id = str(target["scene_id"])
            mask_id = str(i)
            print(f"obj: {pt_name}, scene: {scene_id}, img: {img_id}, mask: {mask_id}")
            set_meta()
            main()
            i += 1
