import open3d as o3d
import numpy as np
import util
import pickle
import cv2
import FeatMatch


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
    depth_scale = 1
    cld_dst = util.depth2cld(depth_dst * 0.001 * depth_scale, [531.15, 0.0, 320.0, 0.0, 531.15, 240.0, 0.0, 0.0, 1.0])

    util.vis_cld(cld_dst, img_dst)

    """ match features """
    li = [0, 1, 2, 3, 4, 5]
    FeatMatch.match_features(imgs_src[li], img_dst, clds_src[li], cld_dst, masks_src[li], mask_dst)

    # imgs_src, clds_src, poses_src = pickle.load(open("cabinet.pt", "rb"))
    # imgs_dst, clds_dst, poses_dst = pickle.load(open("cabinet_eval2.pt", "rb"))

    # img_dst = imgs_dst[3]
    # cld_dst = clds_dst[3]
    # mask = np.zeros(img_dst.shape[:2], dtype=np.uint8)
    # mask[50:300, 200:400] = 255
    # match_features(imgs_src, clds_src, img_dst, cld_dst, mask)


if __name__ == "__main__":
    dataset = "ruapc"
    model_name = "obj_000001"
    pt_name = "1"
    scene_id = "000001"
    img_id = "000000"
    mask_id = "000000"

    proj_path = "/home/yang2019901/GMatch-ORB"
    model_path = f"{proj_path}/bop_data/{dataset}/models/{model_name}.ply"
    pt_path = f"{proj_path}/{dataset}/{pt_name}.pt"
    img_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/rgb/{img_id}.png"
    depth_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/depth/{img_id}.png"
    mask_path = f"{proj_path}/bop_data/{dataset}/test/{scene_id}/mask/{img_id}_{mask_id}.png"

    render()
    main()
