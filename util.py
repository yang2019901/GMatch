import numpy as np
import open3d as o3d
import pickle
import struct
import os
import json

import matplotlib.pyplot as plt
from matplotlib import patches, collections
from scipy.spatial.transform import Rotation


class MetaData:
    """
    xxx_id: int
    xxx_name: 0-padded string
    """

    def __init__(self, proj_path, dataset):
        self.proj_path = proj_path
        self.dataset = dataset

    def init(self, pt_id: int = None, scene_id: int = None, img_id: int = None, mask_id: int = None):
        self.pt_id = pt_id if pt_id is not None else self.pt_id
        self.scene_id = scene_id if scene_id is not None else self.scene_id
        self.img_id = img_id if img_id is not None else self.img_id
        self.mask_id = mask_id if mask_id is not None else self.mask_id

        model_name = f"obj_{str(self.pt_id).zfill(6)}"
        scene_name = str(self.scene_id).zfill(6)
        img_name = str(self.img_id).zfill(6)
        mask_name = str(self.mask_id).zfill(6)

        self.model_path = os.path.join(self.proj_path, f"bop_data/{self.dataset}/models/{model_name}.ply")
        self.pt_path = os.path.join(self.proj_path, f"{self.dataset}/{self.pt_id}.pt")
        self.img_path = os.path.join(self.proj_path, f"bop_data/{self.dataset}/test/{scene_name}/rgb/{img_name}.png")
        self.depth_path = os.path.join(
            self.proj_path, f"bop_data/{self.dataset}/test/{scene_name}/depth/{img_name}.png"
        )
        self.mask_path = os.path.join(
            self.proj_path, f"bop_data/{self.dataset}/test/{scene_name}/mask/{img_name}_{mask_name}.png"
        )

        json_path = os.path.join(self.proj_path, f"bop_data/{self.dataset}/test/{scene_name}/scene_camera.json")
        with open(json_path, "r") as f:
            img_camera = json.load(f)[str(self.img_id)]
            self.cam_intrin = img_camera["cam_K"]
            self.depth_scale = img_camera["depth_scale"]


class MatchData:
    """data required for match and result"""

    def __init__(self):
        """data required for match"""
        self.imgs_src = []
        self.clds_src = []
        self.masks_src = []
        self.poses_src = []
        self.img_dst = None
        self.cld_dst = None
        self.mask_dst = None
        """ match result """
        self.matches_list = []  # list of matches, see gmatch.match_features
        self.cost_list = []  # list of cost, ranging 0-1, see gmatch.match_features
        self.uvs_src = []  # keypoints extracted from each source image
        self.uv_dst = None  # keypoints extracted from the destination image
        self.idx_best = None  # index of the best matches (longest)
        self.mat_m2c = None  # model to camera transformation matrix, 4x4


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
    intrin = np.array(intrisic, dtype=np.float32).reshape(3, 3)
    z = np.asarray(depth, np.float32)
    u, v = np.meshgrid(range(z.shape[1]), range(z.shape[0]))
    uv = np.stack((u, v, np.ones_like(u)), axis=-1, dtype=np.float32)
    pts = np.linalg.inv(intrin) @ uv.reshape(-1, 3).T * z.reshape(-1)
    return pts.T.reshape(z.shape[0], z.shape[1], 3)


def vis_cld(clds, colors=None, poses=None):
    _clds = [transform(cld, pose).reshape(-1, 3) for cld, pose in zip(clds, poses)] if poses is not None else clds

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(_clds, axis=0))
    if colors is not None:
        _colors = [np.array(color).reshape(-1, 3) / 255.0 for color in colors]
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(_colors, axis=0))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis], lookat=[0, 0, 1], front=[0, 0, -1], up=[0, -1, 0], zoom=0.2)


def load_ply(path) -> o3d.geometry.TriangleMesh:
    """Loads a 3D mesh model from a PLY file and returns an Open3d mesh.

    :param path: Path to a PLY file.
    :return: Open3d mesh

    key var: The loaded model given by a dictionary with items:
     - 'pts' (nx3 ndarray)
     - 'normals' (nx3 ndarray), optional
     - 'colors' (nx3 ndarray), optional
     - 'faces' (mx3 ndarray), optional
     - 'texture_uv' (nx2 ndarray), optional
     - 'texture_uv_face' (mx6 ndarray), optional
     - 'texture_file' (string), optional
    """
    f = open(path, "rb")

    # Only triangular faces are supported.
    face_n_corners = 3

    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:
        # Strip the newline character(s).
        line = f.readline().decode("utf8").rstrip("\n").rstrip("\r")

        if line.startswith("comment TextureFile"):
            texture_file = line.split()[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith("element"):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith("property") and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith("property list") and header_face_section:
            elems = line.split()
            if elems[-1] == "vertex_indices" or elems[-1] == "vertex_index":
                # (name of the property, data type)
                face_props.append(("n_corners", elems[2]))
                for i in range(face_n_corners):
                    face_props.append(("ind_" + str(i), elems[3]))
            elif elems[-1] == "texcoord":
                # (name of the property, data type)
                face_props.append(("texcoord", elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(("texcoord_ind_" + str(i), elems[3]))
            else:
                print("Warning: Not supported face property: " + elems[-1])
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model["texture_file"] = texture_file
    model["pts"] = np.zeros((n_pts, 3), np.float64)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.int32)

    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]

    is_normal = False
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        is_normal = True
        model["normals"] = np.zeros((n_pts, 3), np.float64)

    is_color = False
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        is_color = True
        model["colors"] = np.zeros((n_pts, 3), np.float64)

    is_texture_pt = False
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model["texture_uv"] = np.zeros((n_pts, 2), np.float64)

    is_texture_face = False
    if {"texcoord"}.issubset(set(face_props_names)):
        is_texture_face = True
        model["texture_uv_face"] = np.zeros((n_faces, 6), np.float64)

    # Formats for the binary case.
    formats = {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uint": ("I", 4),
        "uchar": ("B", 1),
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "texture_u",
            "texture_v",
        ]
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode("utf8").rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model["pts"][pt_id, 0] = float(prop_vals["x"])
        model["pts"][pt_id, 1] = float(prop_vals["y"])
        model["pts"][pt_id, 2] = float(prop_vals["z"])

        if is_normal:
            model["normals"][pt_id, 0] = float(prop_vals["nx"])
            model["normals"][pt_id, 1] = float(prop_vals["ny"])
            model["normals"][pt_id, 2] = float(prop_vals["nz"])

        if is_color:
            model["colors"][pt_id, 0] = float(prop_vals["red"])
            model["colors"][pt_id, 1] = float(prop_vals["green"])
            model["colors"][pt_id, 2] = float(prop_vals["blue"])

        if is_texture_pt:
            model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])
            model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == "n_corners":
                    if val != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if val != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode("utf8").rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == "n_corners":
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model["faces"][face_id, 0] = int(prop_vals["ind_0"])
        model["faces"][face_id, 1] = int(prop_vals["ind_1"])
        model["faces"][face_id, 2] = int(prop_vals["ind_2"])

        if is_texture_face:
            for i in range(6):
                model["texture_uv_face"][face_id, i] = float(prop_vals["texcoord_ind_{}".format(i)])

    f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(model["pts"] * 0.001)
    mesh.triangles = o3d.utility.Vector3iVector(model["faces"])
    if "texture_file" in model:
        model_texture_path = os.path.join(os.path.dirname(path), model["texture_file"])
        model_texture = o3d.io.read_image(model_texture_path)
        mesh.textures = [o3d.geometry.Image(model_texture)]
        faces = np.asarray(model["faces"]).flatten()
        uvs = model["texture_uv"][faces]
        uvs[:, 1] = 1 - uvs[:, 1]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
    elif "colors" in model:
        mesh.vertex_colors = o3d.utility.Vector3dVector(model["colors"] / 255)
    return mesh


def get_snapshots(mesh):
    """
    snapshots: [(rgb, cld, M_ex), ...]
    rgb: (H, W, 3), 0~1
    cld: (H, W, 3), meters
    M_ex: (4, 4)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)

    # 设置相机参数
    camera_params = [
        ## 6-axis sampling
        {"front": [1, 0, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # x+
        {"front": [-1, 0, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # x-
        {"front": [0, 1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # y+
        {"front": [0, -1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # y-
        {"front": [0, 0, 1], "lookat": [0, 0, 0], "up": [0, 1, 0]},  # z+
        {"front": [0, 0, -1], "lookat": [0, 0, 0], "up": [0, 1, 0]},  # z-
        ## rotated 6-axis sampling
        # {"front": [1, 1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},
        # {"front": [-1, 1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},
        # {"front": [-1, -1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},
        # {"front": [1, -1, 0], "lookat": [0, 0, 0], "up": [0, 0, 1]},
        # {"front": [0, 0, 1], "lookat": [0, 0, 0], "up": [0, 1, 0]},
        # {"front": [0, 0, -1], "lookat": [0, 0, 0], "up": [0, 1, 0]},
        ## 8-axis sampling
        # {"front": [1, 1, 1], "lookat": [0, 0, 0], "up": [0, -1, 1]},
        # {"front": [1, 1, -1], "lookat": [0, 0, 0], "up": [0, 1, 1]},
        # {"front": [1, -1, -1], "lookat": [0, 0, 0], "up": [0, -1, 1]},
        # {"front": [1, -1, 1], "lookat": [0, 0, 0], "up": [0, 1, 1]},
        # {"front": [-1, 1, 1], "lookat": [0, 0, 0], "up": [0, -1, 1]},
        # {"front": [-1, 1, -1], "lookat": [0, 0, 0], "up": [0, 1, 1]},
        # {"front": [-1, -1, -1], "lookat": [0, 0, 0], "up": [0, -1, 1]},
        # {"front": [-1, -1, 1], "lookat": [0, 0, 0], "up": [0, 1, 1]},
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
    """
    imgs: (N, H, W, 3), 0~255, uint8
    clds: (N, H, W, 3), meters, float32
    masks: (N, H, W), bool
    poses: [(pos, quat), ...]
    """
    rgbs, clds, masks, M_ex_list = zip(*snapshots)
    imgs = np.asarray(np.stack(rgbs) * 255, dtype=np.uint8)
    masks = np.asarray(np.stack(masks), dtype=bool)
    clds = np.asarray(np.stack(clds), dtype=np.float32)
    poses = [mat2pose(M_ex) for M_ex in M_ex_list]
    with open(path, "wb") as f:
        pickle.dump((imgs, clds, masks, poses), f)


def vis_snapshots(snapshots):
    clds = []
    for rgb, cld, _, M_pose in snapshots:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cld.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
        pcd.transform(M_pose)
        clds.append(pcd)
    axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    clds.append(axis_mesh)
    o3d.visualization.draw_geometries(clds)


def plot_matches(img1, img2, uv1, uv2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax2.imshow(img2)
    for pt1, pt2 in zip(uv1, uv2):
        cir1 = patches.Circle(pt1, 5, color="red", fill=False)
        cir2 = patches.Circle(pt2, 2, color="red", fill=False)
        ax1.add_patch(cir1)
        ax2.add_patch(cir2)
        l = patches.ConnectionPatch(
            xyA=pt1, xyB=pt2, axesA=ax1, axesB=ax2, coordsA="data", coordsB="data", color="green"
        )
        fig.add_artist(l)
    fig.suptitle(f"matches: {len(uv1)}")
    fig.tight_layout()
    plt.show()
    return


def plot_keypoints(img1, img2, uv1, uv2, Mf12, thresh_des):
    idx1 = -1
    alts = []
    idx2 = -1

    R, G, B = [1, 0, 0, 1], [0, 0.5, 0, 1], [0, 0, 1, 1]

    def on_click_src(event):
        nonlocal idx1, alts, idx2
        if event.inaxes != ax1:
            return
        clr1[idx1] = G
        clr2[alts] = G
        x, y = int(event.xdata), int(event.ydata)
        distances = np.linalg.norm(uv1 - np.array([x, y]), axis=1)
        idx1 = np.argmin(distances)
        ax1.set_title(f"keypoints: {len(uv1)}, selected: {idx1}")
        clr1[idx1] = R
        dists = Mf12[idx1]
        alts = np.where(dists < thresh_des)[0]
        clr2[alts] = R
        if idx2 != -1:
            fig.suptitle(f"feature distance: {Mf12[idx1, idx2]}")
        scat1.set_facecolor(clr1)
        scat2.set_facecolor(clr2)
        fig.canvas.draw()

    def on_click_dst(event):
        nonlocal idx1, idx2, alts
        if event.inaxes != ax2:
            return
        if idx2 != -1:
            clr2[idx2] = G if idx2 not in alts else R
        x, y = int(event.xdata), int(event.ydata)
        distances = np.linalg.norm(uv2 - np.array([x, y]), axis=1)
        idx2 = np.argmin(distances)
        clr2[idx2] = B
        ax2.set_title(f"keypoints: {len(uv2)}, selected: {idx2}")
        if idx1 != -1:
            fig.suptitle(f"feature distance: {Mf12[idx1, idx2]}")
        scat2.set_facecolor(clr2)
        fig.canvas.draw()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax2.imshow(img2)
    clr1 = np.array([G for _ in range(len(uv1))])
    clr2 = np.array([G for _ in range(len(uv2))])
    scat1: collections.PathCollection = ax1.scatter(uv1[:, 0], uv1[:, 1], c=clr1, s=3)
    scat2: collections.PathCollection = ax2.scatter(uv2[:, 0], uv2[:, 1], c=clr2, s=3)
    ax1.set_title(f"keypoints: {len(uv1)}")
    ax2.set_title(f"keypoints: {len(uv2)}")
    ax1.axis("off")
    ax2.axis("off")
    fig.tight_layout()
    fig.canvas.mpl_connect("button_press_event", on_click_src)
    fig.canvas.mpl_connect("button_press_event", on_click_dst)
    plt.show()
