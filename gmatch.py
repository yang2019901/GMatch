"""Implements GMatch to match keypoints extracted by ORB/SIFT detector."""

import numpy as np
import open3d as o3d
import cv2
import time, logging
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.dirname(__file__))

import util


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[%(levelname)5s] [%(funcName)s] %(message)s"))
logger.addHandler(h)
logger.propagate = False


HAM_TAB = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)  # for computing hamming distance, only ORB uses it now
CACHE = {}  # cache for keypoints and features of imgs_src


""" SIFT settings """
detector: cv2.SIFT = cv2.SIFT_create()
# detector.setContrastThreshold(0.03)
N_good = 32  # number of good matches candidates
D = 24  # max search depth
thresh_feat = 0.65  # threshold for feature distance, used to judge the similarity of two feature vectors
feat_mat = lambda feat1, feat2: sift_mat(feat1, feat2)  # feature distance matrix


""" ORB settings """
# detector = cv2.ORB_create(scaleFactor=1.4)
# N_good = 30  # number of good matches
# D = 24  # max search depth
# thresh_feat = 90  # threshold for feature distance, used to judge the similarity of two feature vectors
# feat_mat = lambda feat1, feat2: orb_mat(feat1, feat2)  # feature distance matrix


""" GMatch settings """
# threshold for geometric cost, applied to 3d distance error ratio when attempting to add `m` to matches.
thresh_geom_ratio = 0.1
# threshold for geometric cost, applied to 3d distance directly when attempting to add `m` to matches. (unit: meter)
thresh_geom_abs = 0.005
# threshold for flipover judgement
thresh_flip = 0.8


def orb_mat(feat1, feat2):
    """Compute feature distance matrix `Mf` for ORB, whose metric is Hamming distance.
    > Mh[i, j] == HamDist(feat1[i], feat2[j])

    - feat1: (n1, 32), uint8
    - feat2: (n2, 32), uint8
    """
    global HAM_TAB
    # broadcast feat1 and feat2
    feat1_ = feat1[:, np.newaxis, :]
    feat2_ = feat2[np.newaxis, :, :]
    # compute xor result
    xor_result = feat1_ ^ feat2_
    # compute hamming distance
    hamming_distances = HAM_TAB[xor_result]
    # sum along the last axis to get the hamming distance matrix
    Mf = np.sum(hamming_distances, axis=-1)
    return Mf


def sift_mat(feat1, feat2):
    """Compute feature distance matrix `Mf` for SIFT, whose metric is Euclidean distance.

    Note: feat1 and feat2 will be L1-normalized

    - feat1: (n1, 128), integer stored in float32
    - feat2: (n2, 128), integer stored in float32
    """
    feat1_ = feat1 / np.sum(feat1, axis=-1, keepdims=True)
    feat2_ = feat2 / np.sum(feat2, axis=-1, keepdims=True)
    feat1_ = np.sqrt(feat1_)
    feat2_ = np.sqrt(feat2_)
    Mf = np.linalg.norm(feat1_[:, np.newaxis, :] - feat2_[np.newaxis, :, :], axis=-1)
    return Mf


def cost(matches, pairs, Me11, Me22):
    """Cost function to differ two distance matrices.

    matches: (d, 2), pairs: (n, 2), Me11: (n1, n1), Me22: (n2, n2)
    """
    if len(matches) == 0:
        return 0
    m0, m1 = pairs[:, 0], pairs[:, 1]  # (n, )
    i, j = zip(*matches)  # (d, )
    dist1 = Me11[m0[:, np.newaxis], i]
    dist2 = Me22[m1[:, np.newaxis], j]
    err = np.abs(dist1 - dist2)
    ratio = (1e-5 + err) / (1e-5 + dist1)  # (n, d), error rate
    c = np.where(np.max(err, axis=-1) <= thresh_geom_abs, np.max(ratio, axis=-1), 1)  # (n, d), penalize large errors
    return c  # (n, )


def volume_equal(matches, pairs, pts1, pts2):
    """pairs: (n, 2), return (n, ) boolean array"""
    if len(matches) < 3:
        return np.ones(len(pairs), dtype=bool)
    m = matches  # alias
    S1 = np.cross(pts1[m[0][0]] - pts1[m[-1][0]], pts1[m[0][0]] - pts1[m[-2][0]])
    S2 = np.cross(pts2[m[0][1]] - pts2[m[-1][1]], pts2[m[0][1]] - pts2[m[-2][1]])
    v1 = pts1[pairs[:, 0]] - pts1[m[0][0]]  # (n, 3)
    v2 = pts2[pairs[:, 1]] - pts2[m[0][1]]  # (n, 3)
    V1 = np.sum(S1 * v1, axis=-1)  # (n, )
    V2 = np.sum(S2 * v2, axis=-1)  # (n, )
    flags = np.abs(V1 - V2) < 1e-5  # unit: m^3
    return flags


def flipover(matches, pairs, pts1, pts2):
    """flipover judgement
    pairs: (n, 2), return (n, ) boolean array"""
    global thresh_flip
    if len(matches) < 2:
        return np.zeros(len(pairs), dtype=bool)
    v1_1 = pts1[matches[-2][0]] - pts1[matches[-1][0]]
    v1_2 = pts1[pairs[:, 0]] - pts1[matches[-1][0]]
    v2_1 = pts2[matches[-2][1]] - pts2[matches[-1][1]]
    v2_2 = pts2[pairs[:, 1]] - pts2[matches[-1][1]]
    n1, n2 = np.cross(v1_1, v1_2), np.cross(v2_1, v2_2)
    n1 = np.divide(n1, np.linalg.norm(n1, axis=-1, keepdims=True), out=np.zeros_like(n1), where=n1 != 0)
    n2 = np.divide(n2, np.linalg.norm(n2, axis=-1, keepdims=True), out=np.zeros_like(n2), where=n2 != 0)
    flags = np.bitwise_and(n1[:, 2] * n2[:, 2] < 0, np.abs(n1[:, 2] - n2[:, 2]) > thresh_flip)
    return flags


def gmatch_search(pts1, pts2, Mf12):
    """search with geometric constraints (distance matrix and flip-over removal)"""
    n1, n2 = Mf12.shape
    matches = []
    rlt = []
    rlt_cost = 1
    Me11 = np.linalg.norm(pts1[:, np.newaxis, :] - pts1, axis=-1)
    Me22 = np.linalg.norm(pts2[:, np.newaxis, :] - pts2, axis=-1)

    part_indices = (
        np.argpartition(np.reshape(Mf12, -1), N_good)[:N_good] if N_good < n1 * n2 else np.arange(n1 * n2, dtype=int)
    )
    pairs_good = np.array(np.unravel_index(part_indices, Mf12.shape)).T
    # pairs_good = np.argwhere(Mf12 < 0.1)

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    logger.info(f"Found {len(pairs_simi)} similar pairs.")
    if len(pairs_simi) == 0:
        return np.array([]), 1

    for i, j in pairs_good:
        pairs = pairs_simi
        costs = np.zeros(len(pairs))  # each pair v.s. matches[:-1]
        matches = [(i, j)]
        c = 0
        # step(), search for the next match
        while True:
            if len(matches) == D:
                break

            # update with dynamic programming, Cost(m, matches) = max{ Cost(m, matches[:-1]), Cost(m, matches[-1]) }
            costs = np.maximum(costs, cost([matches[-1]], pairs, Me11, Me22))  # (n, )

            # filter with geometric cost
            ind = np.argwhere(costs < thresh_geom_ratio).flatten()
            pairs = pairs[ind]
            costs = costs[ind]

            if len(pairs) == 0:
                break

            # filter with flip-over test
            # flags = ~flipover(matches, pairs, pts1, pts2)  # (n, )

            flags = volume_equal(matches, pairs, pts1, pts2)  # (n, )
            pairs_ = pairs[flags]
            costs_ = costs[flags]

            if len(pairs_) == 0:
                break

            # get the best match
            best = np.argmin(costs_)
            c = max(c, costs_[best])
            matches.append(tuple(pairs_[best]))

        if len(matches) > len(rlt) or (len(matches) == len(rlt) and c < rlt_cost):
            rlt = matches
            rlt_cost = c
        if len(rlt) == D:
            break

    return np.asarray(rlt), rlt_cost


def ransac_search(pts1, pts2, Mf12):
    import ransac

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    logger.info(f"Found {len(pairs_simi)} similar pairs.")

    if len(pairs_simi) == 0:
        return (np.array([]), 1)

    result = ransac.registration_ransac_based_on_correspondence(pts1, pts2, pairs_simi, 0.008, max_iteration=1000)
    if result is None or result.correspondence_set is None:
        logger.warning("RANSAC failed to find enough correspondences.")
        return (np.array([]), 1)

    return np.array(result.correspondence_set), result.inlier_rmse


def teaserpp_search(pts1, pts2, Mf12):
    import teaserpp_python

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    logger.info(f"Found {len(pairs_simi)} similar pairs.")

    if len(pairs_simi) == 0:
        return (np.array([]), 1)

    src = pts1[pairs_simi[:, 0]].T
    dst = pts2[pairs_simi[:, 1]].T

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = 0.003
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(src, dst)
    solution = solver.getSolution()
    R = solution.rotation.reshape(3, 3)
    t = solution.translation.reshape(3, 1)
    mask = np.linalg.norm(R @ src + t - dst, axis=0) < 0.003
    return np.array(pairs_simi[mask]), 0


def Match(match_data: util.MatchData, cache_id=None, debug=-1):
    """match each of imgs_src with img_dst in match_data and store the result in it;
        keypoints and features for imgs_src will be cached with cache_id if provided
    imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3)
    """
    global detector, CACHE
    assert len(match_data.imgs_src) > 0, "imgs_src is empty"
    """ load from match_data """
    imgs_src, clds_src, masks_src = match_data.imgs_src, match_data.clds_src, match_data.masks_src
    poses_src = match_data.poses_src

    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

    kp_dst, feat_dst = detector.detectAndCompute(img_dst, mask_dst)  # 0.3s for 1920x1080 => 0.014s for 211x200

    if len(kp_dst) == 0:
        print("No keypoints found in img2")
        match_data.matches = np.zeros((0, 2), dtype=np.int32)
        return
    uv_dst = np.array([k.pt for k in kp_dst], dtype=np.int32)
    pt_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    """ extract the keypoints and features with descriptor """
    uvs_src = []
    pts_src = []
    feats_src = []
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        kp_src, feat_src = (
            CACHE[(cache_id, i)] if (cache_id, i) in CACHE else detector.detectAndCompute(img_src, mask_src)
        )

        if cache_id is not None:
            CACHE[(cache_id, i)] = (kp_src, feat_src)

        if len(kp_src) == 0:
            uv_src.append(np.zeros((0, 2), dtype=np.int32))
            pts_src.append(np.zeros((0, 3)))
            feats_src.append(np.zeros((0, feat_src.shape[1])))
            continue

        uv_src = np.array([k.pt for k in kp_src], dtype=np.int32)
        pts_src.append(util.transform(cld_src[uv_src[:, 1], uv_src[:, 0]], poses_src[i]))
        uvs_src.append(uv_src)
        feats_src.append(feat_src)

    if debug > 1:
        for img_src, uv_src, feat_src in zip(imgs_src, uvs_src, feats_src):
            Mf12 = feat_mat(feat_src, feat_dst)
            util.plot_keypoints(img_src, img_dst, uv_src, uv_dst, Mf12, thresh_feat, show_immidiate=False)
            plt.gcf().canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    """ concatenate all keypoints and features """
    pt_src = np.concatenate(pts_src, axis=0)
    feat_src = np.concatenate(feats_src, axis=0)
    """ Feature Distance Matrix """
    Mf12 = feat_mat(feat_src, feat_dst)

    matches, cost = gmatch_search(pt_src, pt_dst, Mf12)

    """ take max depth matches as the best """
    match_data.pt_src = pt_src
    match_data.pt_dst = pt_dst
    match_data.matches = matches
    match_data.cost = cost

    """ visualization """
    if debug > -1 and len(matches) > 0:
        matches_list = split_matches(matches, uvs_src)
        for matches, img_src, uv_src in zip(matches_list, imgs_src, uvs_src):
            util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]], show_immidiate=False)
            plt.gcf().canvas.mpl_connect("key_press_event", on_key)
        plt.show()


def on_key(event):
    if event.key == "q":
        plt.close("all")


def split_matches(matches, uvs):
    n = len(uvs)
    matches_list = [[] for _ in range(n)]

    lengths = np.array([len(uv) for uv in uvs])  # Shape: (n_sources,)
    ind = np.repeat(np.arange(n), lengths)  # Repeat i, len(uv_src[i]) times
    offset = np.concatenate([np.arange(k) for k in lengths])  # Concatenate [0,1,...,n-1] for each

    for i, j in matches:
        matches_list[ind[i]].append((offset[i], j))
    matches_list = [np.array(m) if len(m) > 0 else np.zeros((0, 2), dtype=int) for m in matches_list]
    return matches_list
