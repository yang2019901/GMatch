import open3d as o3d
import numpy as np
import pickle
import cv2
import json
import os

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
    FeatMatch.match_features(imgs_src, img_dst, clds_src, cld_dst, masks_src, mask_dst)

    # imgs_src, clds_src, poses_src = pickle.load(open("cabinet.pt", "rb"))
    # imgs_dst, clds_dst, poses_dst = pickle.load(open("cabinet_eval2.pt", "rb"))

    # img_dst = imgs_dst[3]
    # cld_dst = clds_dst[3]
    # mask = np.zeros(img_dst.shape[:2], dtype=np.uint8)
    # mask[50:300, 200:400] = 255
    # match_features(imgs_src, clds_src, img_dst, cld_dst, mask)


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
    pt_name = "15"
    scene_id = "1"
    img_id = "0"
    mask_id = "1"

    ## FeatMatch parameters override
    FeatMatch.N1 = 500
    FeatMatch.N2 = 600
    FeatMatch.N_good = 30
    FeatMatch.D = 24
    FeatMatch.thresh_ham = 100
    FeatMatch.thresh_loss = 0.08
    FeatMatch.thresh_flip = 0.05

    set_meta()
    if not os.path.exists(pt_path):
        render()
    main()

    ## bop19 test set
    with open(f"{proj_path}/bop_data/{dataset}/test_targets_bop19.json", "r") as f:
        targets = json.load(f)
        id_list = []
        for i, target in enumerate(targets):
            if target["im_id"] != 0 or target["scene_id"] != 1:
                break
            if target["inst_count"] > 1:
                id_list.append(target["obj_id"])
            pt_name = str(target["obj_id"])
            img_id = str(target["im_id"])
            scene_id = str(target["scene_id"])
            mask_id = str(i)
            set_meta()
            main()
