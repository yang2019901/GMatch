import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
import pickle, cv2
import json, time
import copy
import struct
import cProfile

import gmatch
import util


cache = {}


class MetaData:
    """
    xxx_id: int
    xxx_name: 0-padded string
    """

    def __init__(self, proj_path, dataset):
        self.proj_path = proj_path
        self.dataset = dataset

    def init(
        self,
        pt_id: int = None,
        scene_id: int = None,
        img_id: int = None,
        mask_id: int = None,
    ):
        self.pt_id = pt_id if pt_id is not None else self.pt_id
        self.scene_id = scene_id if scene_id is not None else self.scene_id
        self.img_id = img_id if img_id is not None else self.img_id
        self.mask_id = mask_id if mask_id is not None else self.mask_id

        model_name = f"obj_{str(self.pt_id).zfill(6)}"
        scene_name = str(self.scene_id).zfill(6)
        img_name = str(self.img_id).zfill(6)
        mask_name = str(self.mask_id).zfill(6)

        self.model_path = osp.join(self.proj_path, f"bop_data/{self.dataset}/models/{model_name}.ply")
        self.pt_path = osp.join(self.proj_path, f"{self.dataset}/{self.pt_id}.pt")
        self.img_path = osp.join(
            self.proj_path,
            f"bop_data/{self.dataset}/test/{scene_name}/rgb/{img_name}.png",
        )
        self.depth_path = osp.join(
            self.proj_path,
            f"bop_data/{self.dataset}/test/{scene_name}/depth/{img_name}.png",
        )
        self.mask_path = osp.join(
            self.proj_path,
            f"bop_data/{self.dataset}/test/{scene_name}/mask/{img_name}_{mask_name}.png",
        )

        json_path = osp.join(
            self.proj_path,
            f"bop_data/{self.dataset}/test/{scene_name}/scene_camera.json",
        )
        with open(json_path, "r") as f:
            img_camera = json.load(f)[str(self.img_id)]
            self.cam_intrin = img_camera["cam_K"]
            self.depth_scale = img_camera["depth_scale"]


def load_ply(path) -> o3d.geometry.TriangleMesh:
    """Loads a 3D mesh model from a PLY file and returns an Open3d mesh. Used to read BOP dataset models.

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
        model_texture_path = osp.join(osp.dirname(path), model["texture_file"])
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


def render(meta_data: MetaData):
    """render model to snapshots and save to pt_path"""
    mesh = load_ply(meta_data.model_path)
    """ calc diameter of the model to compare with 'models/models_info.json' """
    pts = np.asarray(mesh.vertices)
    bbox = (np.max(pts, axis=0) - np.min(pts, axis=0)) * 1000
    # axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([mesh, axis_mesh])
    snapshots = util.get_snapshots(mesh)
    util.vis_snapshots(snapshots)
    util.save_snapshots(snapshots, meta_data.pt_path)
    print(f"saved to {meta_data.pt_path}, bbox: {bbox} mm")


def load(meta_data: MetaData, match_data: util.MatchData):
    """load by meta_data and store to match_data"""
    if not osp.exists(meta_data.pt_path):
        render(meta_data)
    """load model images"""
    if meta_data.pt_id not in cache:
        """load from disk"""
        with open(meta_data.pt_path, "rb") as f:
            data = pickle.load(f)
        imgs_src, clds_src, masks_src, poses_src = data
        imgs_src = [cv2.GaussianBlur(img, (5, 5), 0) for img in imgs_src]
        masks_src = masks_src.astype(np.uint8) * 255
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


def result2record(meta_data: MetaData, match_data: util.MatchData):
    """record is formatted as bop19 result except that timespan is missing"""
    scene_id, im_id, obj_id = meta_data.scene_id, meta_data.img_id, meta_data.pt_id
    score = len(match_data.matches_list[match_data.idx_best])
    ## Note: convert `t` to mm, leave `R` as it is for it has no unit
    R, t = match_data.mat_m2c[:3, :3], match_data.mat_m2c[:3, 3] * 1000
    R = " ".join(map(lambda x: f"{x:.6f}", R.flatten().tolist()))
    t = " ".join(map(lambda x: f"{x:.6f}", t.flatten().tolist()))
    return [str(scene_id), str(im_id), str(obj_id), str(score), R, t]


def process_img(meta_data: MetaData, match_data: util.MatchData, targets):
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
        k = max(
            enumerate(match_data_list),
            key=lambda x: len(x[1].matches_list[x[1].idx_best]),
        )[0]
        match_data = match_data_list[k]
        meta_data.init(pt_id=obj_ids[k], scene_id=scene_id, img_id=img_id, mask_id=mask_id)
        util.Solve(match_data)
        record_list.append(result2record(meta_data, match_data))
    timespan = time.time() - t0
    return [f'{", ".join(rec)}, {timespan:.2f}\n' for rec in record_list]


def run_hope():
    meta_data = MetaData(proj_path=osp.dirname(osp.abspath(__file__)), dataset="hope")
    match_data = util.MatchData()

    # meta_data.init(pt_id=23, scene_id=6, img_id=0, mask_id=1)
    # meta_data.init(scene_id=1, img_id=1, pt_id=19, mask_id=12)
    # load(meta_data, match_data)
    # # t0 = time.time()
    # gmatch.Match(match_data)
    # # print(f"match time: {time.time() - t0:.3f}")
    # print(f"best loss: {match_data.cost_list[match_data.idx_best]:.3f}")
    # print(f"obj: {meta_data.pt_id}, len: {len(match_data.matches_list[match_data.idx_best])}")
    # util.Solve(match_data)
    # exit()

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


def run_ycbv():
    """test perception stability (precision, run-time, etc) on video"""
    meta_data = MetaData(proj_path=osp.dirname(osp.abspath(__file__)), dataset="ycbv")
    match_data = util.MatchData()

    result = []
    for pt_id, scene_id, mask_id in [(3, 54, 1), (12, 54, 2), (8, 58, 2), (2, 50, 0)]:
        img_folder = osp.join(meta_data.proj_path, f"bop_data/ycbv/test/{str(scene_id).zfill(6)}/rgb")
        with open(f"bop_data/ycbv/test/{str(scene_id).zfill(6)}/scene_gt.json", "r") as f:
            content = json.load(f)
        files = os.listdir(img_folder)
        imgs_id = [int(f.split(".")[0]) for f in files]
        imgs_id.sort()
        for img_id in imgs_id:
            meta_data.init(pt_id=pt_id, scene_id=scene_id, img_id=img_id, mask_id=mask_id)
            load(meta_data, match_data)
            t0 = time.time()
            gmatch.Match(match_data, cache_id=meta_data.pt_id)
            util.Solve(match_data)
            dt = time.time() - t0
            print(
                f"img_id: {meta_data.img_id}, len: {len(match_data.matches_list[match_data.idx_best])}",
                end=", ",
            )

            M_pred = match_data.mat_m2c

            M = np.eye(4)
            gt = next((x for x in content[str(img_id)] if x["obj_id"] == pt_id))
            M[:3, :3] = np.array(gt["cam_R_m2c"]).reshape(3, 3)
            M[:3, 3] = np.array(gt["cam_t_m2c"]) * 0.001

            M_err = np.linalg.inv(M) @ M_pred

            dist_err = np.linalg.norm(M_err[:3, 3]) * 1000
            ang_err = np.rad2deg(np.arccos((np.trace(M_err[:3, :3]) - 1) / 2))
            print(f"dist_err: {dist_err:.1f} mm, ang_err: {ang_err:.1f} deg, dt: {dt*1000:.1f} ms")
            result.append(f"{dist_err}\n")

    with open(f"result_D{gmatch.D}.csv", "w") as f:
        f.writelines(result)


if __name__ == "__main__":
    for D in [3, 6, 8, 12, 16, 20, 24, 32]:
        gmatch.D = D
        run_ycbv()
