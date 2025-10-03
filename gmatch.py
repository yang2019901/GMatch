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
logger.setLevel(logging.WARN)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[%(levelname)5s] [%(funcName)s] %(message)s"))
logger.addHandler(h)
logger.propagate = False


CACHE = {}  # cache for keypoints and features of imgs_src

""" SIFT settings """
detector: cv2.SIFT = cv2.SIFT_create()
detector.setContrastThreshold(0.03)
thresh_feat = 0.65  # threshold for feature distance, used to judge the similarity of two feature vectors
feat_mat = lambda feat1, feat2: sift_mat(feat1, feat2, rootsift=True)  # feature distance matrix

""" GMatch settings """
N_good = 24  # number of good matches candidates
L = 24  # max search length
# threshold for geometric cost, applied to 3d distance error ratio when attempting to add `m` to matches.
thresh_geom_ratio = 0.1
# threshold for geometric cost, applied to 3d distance directly when attempting to add `m` to matches. (unit: meter)
thresh_geom_abs = 0.005
# threshold for flipover judgement
thresh_flip = 0.8


def sift_mat(feat1, feat2, rootsift):
    """Compute feature distance matrix `Mf` for SIFT, whose metric is Euclidean distance.

    Note: feat1 and feat2 will be L1-normalized

    - feat1: (n1, 128), integer stored in float32
    - feat2: (n2, 128), integer stored in float32
    """
    feat1_ = feat1 / np.sum(feat1, axis=-1, keepdims=True)
    feat2_ = feat2 / np.sum(feat2, axis=-1, keepdims=True)
    if rootsift:
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
    c = np.where(np.max(err, axis=-1) < thresh_geom_abs, np.max(ratio, axis=-1), 1)  # (n, d), penalize large errors
    return c  # (n, )


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
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-5
    n2 /= np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-5 
    flags = np.bitwise_and(n1[:, 2] * n2[:, 2] < 0, np.abs(n1[:, 2] - n2[:, 2]) > thresh_flip)
    return flags


def gmatch_search_greedy(pts1, pts2, Mf12):
    """Greedy search with geometric constraints (distance matrix and flip-over removal)"""
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
            if len(matches) == L:
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
            flags = ~flipover(matches, pairs, pts1, pts2)  # (n, )
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
        if len(rlt) == L:
            break

    return np.asarray(rlt), rlt_cost


def gmatch_search_bnb(pts1, pts2, Mf12):
    """Branch-and-Bound search with geometric constraints (distance matrix and flip-over removal)"""
    import gmatch_cpp
    matches, cost = gmatch_cpp.gmatch_search_bnb(pts1, pts2, Mf12, thresh_feat, L, N_good, thresh_geom_ratio, thresh_geom_abs, thresh_flip)
    # print(f"gmatch_cpp: found {len(matches)} matches, cost {cost:.3f}")
    # print(f"matches: {matches}")
    return np.array(matches), cost


def ransac_search(pts1, pts2, Mf12):
    import ransac

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    logger.info(f"Found {len(pairs_simi)} similar pairs.")

    if len(pairs_simi) == 0:
        return (np.array([]), 1)

    result = ransac.registration_ransac_based_on_correspondence(pts1, pts2, pairs_simi, thresh_geom_abs, max_iteration=1000)
    if result is None or result.correspondence_set is None:
        logger.warning("RANSAC failed to find enough correspondences.")
        return (np.array([]), 1)

    return np.array(result.correspondence_set), result.inlier_rmse


def teaserpp_search(pts1, pts2, Mf12):
    import teaserpp_python

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    max_pairs = 2000
    # NOTE: too many pairs will cause teaserpp to be very slow or even crash
    if len(pairs_simi) > max_pairs: 
        pairs_simi = pairs_simi[np.argpartition(Mf12[pairs_simi[:, 0], pairs_simi[:, 1]], max_pairs)[:max_pairs]]
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
    mask = np.linalg.norm(R @ src + t - dst, axis=0) < thresh_geom_abs
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
    t0 = time.time()
    imgs_src, clds_src, masks_src = match_data.imgs_src, match_data.clds_src, match_data.masks_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

    kp_dst, feat_dst = detector.detectAndCompute(img_dst, mask_dst)  # 0.3s for 1920x1080 => 0.014s for 211x200
    dt1 = time.time() - t0

    if len(kp_dst) == 0:
        print("No keypoints found in img2")
        match_data.matches_list = [[]]
        match_data.cost_list = [1]
        match_data.uvs_src = []
        match_data.uv_dst = None
        match_data.idx_best = 0
        return
    uv_dst = np.array([k.pt for k in kp_dst], dtype=np.int32)
    mask = cld_dst[uv_dst[:, 1], uv_dst[:, 0], 2] > 1e-3
    uv_dst = uv_dst[mask]
    feat_dst = feat_dst[mask]
    pts_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    """ extract the keypoints and features with descriptor """
    matches_list = []
    uvs_src = []
    dt2, dt3, dt4 = 0, 0, 0
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        t0 = time.time()
        if (cache_id, i) in CACHE:
            uv_src, feat_src = CACHE[(cache_id, i)]
        else:
            kp_src, feat_src = detector.detectAndCompute(img_src, mask_src)
            if len(kp_src) == 0:
                uv_src = np.zeros((0, 2), dtype=np.int32)
                feat_src = np.zeros((0, detector.descriptorSize()), dtype=bool)
            else:
                uv_src = np.array([k.pt for k in kp_src], dtype=np.int32)
                uv_src, ind = np.unique(uv_src, axis=0, return_index=True)
                feat_src = feat_src[ind]
                # filter out points with invalid depth
                mask = cld_src[uv_src[:, 1], uv_src[:, 0], 2] > 1e-3
                uv_src = uv_src[mask]
                feat_src = feat_src[mask]
            if cache_id is not None:
                CACHE[(cache_id, i)] = (uv_src, feat_src)

        if len(uv_src) == 0:
            matches_list.append(([], 1))
            continue
        pts_src = cld_src[uv_src[:, 1], uv_src[:, 0]]
        uvs_src.append(uv_src)
        dt2 += time.time() - t0

        """ Feature Distance Matrix (for visual similarity) """
        t0 = time.time()
        Mf12 = feat_mat(feat_src, feat_dst)
        dt3 += time.time() - t0

        """ <Tune>
            N1 and N2: plot to see whether keypoints are enough
            thresh_feat: find a suitable threshold for feature distance
        """
        t0 = time.time()
        if debug > 1:
            util.plot_keypoints(img_src, img_dst, uv_src, uv_dst, Mf12, thresh_feat)
        matches, cost = gmatch_search_bnb(pts_src, pts_dst, Mf12)
        matches_list.append((matches, cost))
        dt4 += time.time() - t0

        """ visualization """
        if debug > 0:
            if len(matches) < 3:
                print(f"\timgs_src[{i}]: matches NOT found.")
            else:
                print(f"\timgs_src[{i}]: matches found. depth {len(matches)}, cost {cost:.3f}")
                util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])

        if len(matches) == L:
            break

    """ take max depth matches as the best """
    match_data.matches_list, match_data.cost_list = zip(*matches_list)
    match_data.uvs_src = uvs_src
    match_data.uv_dst = uv_dst
    match_data.idx_best = np.argmax([len(matches) for matches, _ in matches_list])

    """ visualization """
    if debug > -1 and len(match_data.matches_list[match_data.idx_best]) > 0:
        img_src = imgs_src[match_data.idx_best]
        uv_src = uvs_src[match_data.idx_best]
        matches = matches_list[match_data.idx_best][0]
        util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])

    return dt1, dt2, dt3, dt4