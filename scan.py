import cv2, pickle, time
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import os.path as osp
import os, sys
import matplotlib.pyplot as plt
import argparse

import gmatch


H, W = 480, 640

import cv2
import numpy as np


class BBoxAnnotator:
    """
    A helper class to annotate a bounding box on an image using mouse events.
    - Left-click and drag to draw a rectangle.
    - Press 'r' to reset the current drawing.
    - Press 'q' or 'Esc' to quit and return current box (if any).
    - All other keys are ignored (no action).
    """

    def __init__(self, img):
        """
        Initialize with an input image (BGR format, as displayed by OpenCV).
        
        Args:
            img (np.ndarray): Input image (H x W x 3), uint8, BGR.
        """
        self.img = img.copy()          # Current display image (may show temporary rectangle)
        self.img_bk = img.copy()        # Original image (for reset)
        self.drawing = False           # Flag to indicate if user is drawing
        self.start_point = None        # Starting point (x, y) of rectangle
        self.end_point = None          # Current end point (x, y) during drag
        self.bbox = None               # Final bbox as (r1, c1, r2, c2) = (y1, x1, y2, x2)

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for OpenCV window.
        Handles drawing logic on mouse events.
        """
        x = np.clip(x, 0, W)
        y = np.clip(y, 0, H)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update rectangle while dragging
            if self.drawing:
                self.end_point = (x, y)
                # Restore original image and redraw current rectangle
                self.img = self.img_bk.copy()
                cv2.rectangle(self.img, self.start_point, self.end_point, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            self.end_point = (x, y)
            
            # Normalize to (top-left, bottom-right)
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            c1, c2 = min(x1, x2), max(x1, x2)  # column indices (x)
            r1, r2 = min(y1, y2), max(y1, y2)  # row indices (y)
            self.bbox = (r1, c1, r2, c2)

            # Draw final rectangle
            self.img = self.img_bk.copy()
            cv2.rectangle(self.img, (c1, r1), (c2, r2), (0, 255, 0), 2)

    def annotate(self, window_name):
        """
        Launch an interactive window for bounding box annotation.

        Args:
            window_name (str): Name of the OpenCV window.

        Returns:
            tuple or None: Bounding box as (r1, c1, r2, c2) if confirmed,
                           or None if user pressed 'q' or 'Esc'.
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, self.img)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("Draw a bounding box by dragging with the left mouse button.")
        print("Press 'r' to reset the box, 'q' or 'Esc' to skip.")

        while True:
            cv2.imshow(window_name, self.img)
            key = cv2.waitKey(100) & 0xFF  # Wait 100ms to keep GUI responsive

            if key == ord('r') or key == ord('R'):
                # Reset the drawing
                self.img = self.img_bk.copy()
                self.bbox = None
                print("Bounding box reset.")

            elif key == ord('q') or key == ord('Q') or key == 27:  # 27 = Esc
                break

        cv2.destroyAllWindows()
        return self.bbox


def record(path_save):
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

    with open(path_save, "wb") as f:
        pickle.dump(records, f)


def calibrate(records):
    if len(records) < 1:
        return

    ## use the first record as the base coordinate system
    imgs_src = []
    clds_src = []
    masks_src = []
    poses_src = []

    ## calibrate poses one by one
    for i, (img_dst, cld_dst) in enumerate(records):
        window_name = f"No.{i+1}/{len(records)} RGB"
        img_bgr = cv2.cvtColor(img_dst, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    
        annotator = BBoxAnnotator(img_bgr)
        bbox = annotator.annotate(window_name)  # Returns (r1, c1, r2, c2) or None

        if bbox is not None:
            r1, c1, r2, c2 = bbox
            mask_dst = np.zeros((H, W), dtype=np.uint8)
            mask_dst[r1:r2, c1:c2] = 255
        
            # Optional: Show masked image
            masked_rgb = img_dst * (mask_dst[:, :, None] > 0)
            cv2.imshow(window_name, cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
            print(f"Annotated bbox: r1={r1}, c1={c1}, r2={r2}, c2={c2}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No bounding box selected. Using depth-based mask.")
            mask_dst = np.where((cld_dst[:, :, 2] > 1e-2), 255, 0).astype(np.uint8)

        if i == 0:
            imgs_src.append(img_dst)
            clds_src.append(cld_dst)
            masks_src.append(mask_dst)
            poses_src.append(gmatch.util.mat2pose(np.eye(4)))
            continue

        match_data = gmatch.util.MatchData(
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
        gmatch.Match(match_data, cache_id=obj_name, debug=2)
        t1 = time.time()
        idx = match_data.idx_best
        cost_list = match_data.cost_list
        print(f"Match with frame No.{idx}: {t1 - t0:.3f} seconds. Best cost {cost_list[idx]:.3f}")
        if cost_list[idx] >= 1:
            continue
        gmatch.util.Solve(match_data)
        ## update the source data
        imgs_src.append(img_dst)
        clds_src.append(cld_dst)
        masks_src.append(mask_dst)
        poses_src.append(gmatch.util.mat2pose(np.linalg.inv(match_data.mat_m2c)))
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
    parser = argparse.ArgumentParser(description="The scanner wizard to get snapshots of target object.")
    parser.add_argument("--object_label", type=str, required=True, help="The store label of target object.")
    parser.add_argument("--resume", action="store_true", help="resume from raw.pkl")
    args = parser.parse_args()

    cache_folder = "./cache"
    os.makedirs(cache_folder, exist_ok=True)

    obj_name = args.object_label
    path_raw = osp.join(cache_folder, "raw.pkl")
    path_model = osp.join(cache_folder, f"{obj_name}.pt")

    print(f"Env: path_raw_file={path_raw}; path_model_file={path_model}")

    if not args.resume or not osp.exists(path_raw):
        record(path_raw)

    records = pickle.load(open(path_raw, "rb"))

    rgbs, clds, masks, poses = calibrate(records)
    M_list = [gmatch.util.pose2mat(pose) for pose in poses]

    tmp = input("redraw box [y/N]?")
    if tmp == 'y' or tmp == 'Y':
        for i in range(len(records)):
            window_name = f"No.{i+1}/{len(records)} RGB"
            bgr = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            annotator = BBoxAnnotator(bgr)
            bbox = annotator.annotate(window_name)  # Returns (r1, c1, r2, c2) or None

            if bbox is not None:
                r1, c1, r2, c2 = bbox
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[r1:r2, c1:c2] = 255

                masked_bgr = bgr * (mask[:, :, None] > 0)
                cv2.imshow(window_name, masked_bgr)
                print(f"Annotated bbox: r1={r1}, c1={c1}, r2={r2}, c2={c2}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No bounding box selected. Using depth-based mask.")
                mask = np.where((clds[i][:, :, 2] > 1e-2), 255, 0).astype(np.uint8)
            masks[i] = mask

    snapshots = list(zip(rgbs, clds, masks, M_list))
    with open(path_model, "wb") as f:
        pickle.dump(snapshots, f)

    snapshots = pickle.load(open(path_model, "rb"))

    pcds = []
    for rgb, cld, mask, M in snapshots:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cld[mask != 0])
        pcd.colors = o3d.utility.Vector3dVector(rgb[mask != 0] / 255.0)
        pcd.transform(M)
        pcds.append(pcd)
    visualize_point_clouds_with_toggle(pcds)
