import cv2, pickle, time
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import os.path as osp
import os, sys
import matplotlib.pyplot as plt

import gmatch, util


H, W = 480, 640


def record(output_file):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    records = []
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # get rgb, np.ndarray, (H, W, 3)
            rgb = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)

            # get point cloud, np.ndarray, (H, W, 3)
            pc = rs.pointcloud()
            points = pc.calculate(aligned_depth_frame)
            v, _ = points.get_vertices(), points.get_texture_coordinates()
            xyz = np.asanyarray(v).view(np.float32).reshape(H, W, 3)

            cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth = np.asanyarray(aligned_depth_frame.get_data())
            cv2.imshow("depth", depth / 1000.0)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                # save as a key frame
                print("record(): a key frame saved.")
                records.append((rgb, xyz))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(records) == 0:
        print("No records captured.")
        return

    with open(output_file, "wb") as f:
        pickle.dump(records, f)


def calibrate(records):
    if len(records) < 1:
        return

    ## use the first record as the base coordinate system
    imgs_src = [records[0][0]]
    clds_src = [records[0][1]]
    masks_src = [None]
    poses_src = [util.mat2pose(np.eye(4))]

    ## calibrate poses one by one
    for img_dst, cld_dst in records[1:]:
        mask_dst = None
        match_data = util.MatchData(
            imgs_src=imgs_src,
            clds_src=clds_src,
            masks_src=masks_src,
            poses_src=poses_src,
            img_dst=img_dst,
            cld_dst=cld_dst,
            mask_dst=mask_dst,
        )
        t0 = time.time()
        global obj_name
        gmatch.Match(match_data, cache_id=obj_name)
        t1 = time.time()
        idx = match_data.idx_best
        cost_list = match_data.cost_list
        print(f"Match with frame No.{idx}: {t1 - t0:.3f} seconds. Best cost {cost_list[idx]:.3f}")
        if cost_list[idx] >= 1:
            continue
        util.Solve(match_data)
        ## update the source data
        imgs_src.append(img_dst)
        clds_src.append(cld_dst)
        masks_src.append(mask_dst)
        poses_src.append(util.mat2pose(np.linalg.inv(match_data.mat_m2c)))
    return imgs_src, clds_src, masks_src, poses_src


def visualize_point_clouds_with_toggle(point_clouds):
    visible = [True] * len(point_clouds)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    ctr: o3d.visualization.ViewControl = vis.get_view_control()

    for pc, v in zip(point_clouds, visible):
        if v:
            vis.add_geometry(pc)

    def toggle(idx):
        def callback(vis_obj: o3d.visualization.VisualizerWithKeyCallback):
            ## retrieve the current camera parameters
            params = ctr.convert_to_pinhole_camera_parameters()
            ## clear and add geometries with `visible`
            visible[idx] = not visible[idx]
            vis_obj.clear_geometries()
            for pc, v in zip(point_clouds, visible):
                if v:
                    vis.add_geometry(pc)
            print(f"visible: {[i for i, v in enumerate(visible) if v]}")
            ## restore the camera parameters
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            return False

        return callback

    binds = "1234567890QWERTYUIOPASDFGHJKL"
    for i in range(min(len(binds), len(point_clouds))):
        print(f"Press '{binds[i]}' to toggle point cloud {i}.")
        vis.register_key_callback(ord(binds[i]), toggle(i))

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    cache_folder = './cache'
    os.makedirs(cache_folder, exist_ok=True)
    obj_name = sys.argv[1] if len(sys.argv) > 1 else 'default_object'
    raw_file = osp.join(cache_folder, f"{obj_name}_raw.pkl")
    model_file = osp.join(cache_folder, f"{obj_name}.pt")

    if not osp.exists(raw_file):
        record(raw_file)

    records = pickle.load(open(raw_file, "rb"))

    rgbs, clds, masks, poses = calibrate(records)
    M_list = [util.pose2mat(pose) for pose in poses]

    D_near, D_far = 0.1, 2.0
    masks = [np.where((cld[..., 2] > D_near) & (cld[..., 2] < D_far), 255, 0).astype(np.uint8) for cld in clds]

    snapshots = list(zip(rgbs, clds, masks, M_list))
    with open(model_file, "wb") as f:
        pickle.dump(snapshots, f)

    snapshots = pickle.load(open(model_file, "rb"))

    pcds = []
    for rgb, cld, mask, M in snapshots:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cld[mask != 0])
        pcd.colors = o3d.utility.Vector3dVector(rgb[mask != 0] / 255.0)
        pcd.transform(M)
        pcds.append(pcd)
    visualize_point_clouds_with_toggle(pcds)
