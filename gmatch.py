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
D = 24  # max search depth
# threshold for geometric cost, applied to 3d distance error ratio when attempting to add `m` to matches.
thresh_geom_ratio = 0.08
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


def update_cov(Cov, mean, x, n):
    """update covariance matrix with new point x
    Cov: (3, 3)
    mean: (3, )
    x: (3, )
    n: int, number of points before adding x
    """
    Corr = n * (Cov + np.outer(mean, mean)) + np.outer(x, x)
    mean = (mean * n + x) / (n + 1)
    Cov = Corr / (n + 1) - np.outer(mean, mean)
    return Cov, mean


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

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    logger.info(f"Found {len(pairs_simi)} similar pairs.")
    if len(pairs_simi) == 0:
        return np.array([]), 1

    k = 8  # branch number
    t0 = time.time()
    # Branch and bound
    li = []
    for i, j in pairs_good:
        matches = [(i, j)]
        costs = cost(matches, pairs_simi, Me11, Me22)  # (n, )
        flags = costs < thresh_geom_ratio
        costs = costs[flags]
        pairs = pairs_simi[flags]
        ind = np.argpartition(costs, k)[:k] if k < len(costs) else np.arange(len(costs), dtype=int)
        for idx in ind:
            li.append(([(i, j), tuple(pairs[idx])], pairs, costs, costs[idx]))
    logger.info(f"step1 cost: {time.time()-t0:.3f}s")

    # Branch and bound again (because 3 pairs can determine a SE(3) transformation if not colinear)
    li2 = []
    t0 = time.time()
    for matches, pairs, costs, c in li:
        costs = np.maximum(costs, cost([matches[-1]], pairs, Me11, Me22))  # (n, )
        flags = costs < thresh_geom_ratio
        costs = costs[flags]
        pairs = pairs[flags]
        ind = np.argpartition(costs, k)[:k] if k < len(costs) else np.arange(len(costs), dtype=int)
        for idx in ind:
            li2.append((matches + [tuple(pairs[idx])], pairs, costs, max(c, costs[idx])))
    logger.info(f"step2 cost: {time.time()-t0:.3f}s")

    if len(li2) == 0:
        logger.info("No valid candidates found.")
        return np.array([]), 1

    li2 = sorted(li2, key=lambda x: x[3])[: N_good * k]

    # remove duplicates
    logger.info(f"Initial candidates 2: {len(li2)}")
    matches_list, _, _, costs = zip(*li2)
    tmp = np.array(matches_list).sum(axis=1)
    fingerprints = tmp[:, 0] + 1 / (tmp[:, 1] + 1) + np.array(costs)
    _, idx_unique = np.unique(fingerprints, return_index=True)
    li2 = [li2[i] for i in idx_unique]
    logger.info(f"Filtered candidates 2: {len(li2)}, max pairs length: {len(li2[0][1])}")

    rlt_cost = 1
    for i, (matches, pairs, costs, c) in enumerate(li2):
        logger.info(f"No.{i:<2}, initial matches: {matches}, candidate pairs: {len(pairs)}")
        idx1, idx2 = zip(*matches)
        mean1 = np.mean(pts1[idx1, :], axis=0)
        mean2 = np.mean(pts2[idx2, :], axis=0)
        Cov1 = np.cov(pts1[idx1, :].T, bias=True)  # 1/n xi xi^T - mu mu^T
        Cov2 = np.cov(pts2[idx2, :].T, bias=True)

        while True:
            if len(matches) >= D:
                break

            # update with dynamic programming, Cost(m, matches) = max{ Cost(m, matches[:-1]), Cost(m, matches[-1]) }
            costs = np.maximum(costs, cost([matches[-1]], pairs, Me11, Me22))  # (n, )

            # filter with geometric cost
            flags = costs < thresh_geom_ratio
            pairs = pairs[flags]
            costs = costs[flags]

            if len(pairs) == 0:
                break

            w1, v1 = np.linalg.eigh(Cov1)
            w2, v2 = np.linalg.eigh(Cov2)
            w1 = np.sqrt(np.abs(w1))  # sqrt to make it comparable to stddev
            w2 = np.sqrt(np.abs(w2))
            if np.any(np.abs(w1 - w2) >= thresh_geom_abs):
                break
            if w1[1] < thresh_geom_abs:
                logger.info(f"points in pts1 are colinear.")
                # get the best match
                best = np.argmin(costs)
                c = max(c, costs[best])
                Cov1, mean1 = update_cov(Cov1, mean1, pts1[pairs[best, 0]], len(matches))
                Cov2, mean2 = update_cov(Cov2, mean2, pts2[pairs[best, 1]], len(matches))
                matches.append(tuple(pairs[best]))
                continue

            assert w1[0] < thresh_geom_abs

            d1 = np.dot(pts1[pairs[:, 0]] - mean1, v1[:, 0])
            d2 = np.dot(pts2[pairs[:, 1]] - mean2, v2[:, 0])
            flags = (np.abs(d1 - d2) < thresh_geom_abs) & (d1 * d2 >= 0)

            pairs_ = pairs[flags]
            costs_ = costs[flags]

            if len(pairs_) == 0:
                break

            # get the best match
            best = np.argmin(costs_)
            c = max(c, costs_[best])
            logger.info(f"d1: {d1[flags][best]:.3f}, d2: {d2[flags][best]:.3f}")
            if abs(d1[flags][best]) < thresh_geom_abs:
                Cov1, mean1 = update_cov(Cov1, mean1, pts1[pairs_[best, 0]], len(matches))
                Cov2, mean2 = update_cov(Cov2, mean2, pts2[pairs_[best, 1]], len(matches))
            matches.append(tuple(pairs_[best]))

        logger.info(f"No.{i:<2}, final matches ({len(matches):<2}): {matches}, cost: {c:.3f}")

        if len(matches) > len(rlt) or (len(matches) == len(rlt) and c < rlt_cost):
            rlt = matches
            rlt_cost = c

        if len(rlt) >= D:
            break

    logger.info(f"Final matches ({len(rlt):<2}): {rlt}, cost: {rlt_cost:.3f}")
    return np.asarray(rlt), rlt_cost


def ransac_search(pts1, pts2, Mf12):
    import ransac

    pairs_simi = np.argwhere(Mf12 < thresh_feat)
    logger.info(f"Found {len(pairs_simi)} similar pairs.")

    if len(pairs_simi) == 0:
        return (np.array([]), 1)

    result = ransac.registration_ransac_based_on_correspondence(
        pts1, pts2, pairs_simi, thresh_geom_abs, max_iteration=1000
    )
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
    imgs_src, clds_src, masks_src = match_data.imgs_src, match_data.clds_src, match_data.masks_src
    poses_src = match_data.poses_src

    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

    kp_dst, feat_dst = detector.detectAndCompute(img_dst, mask_dst)  # 0.3s for 1920x1080 => 0.014s for 211x200

    if len(kp_dst) == 0:
        print("No keypoints found in img2")
        match_data.matches = np.zeros((0, 2), dtype=np.int32)
        return

    uv_dst = np.array([k.pt for k in kp_dst], dtype=np.int32)
    uv_dst, uniq_ind = np.unique(uv_dst, axis=0, return_index=True)
    feat_dst = feat_dst[uniq_ind]
    pt_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    """ extract the keypoints and features with descriptor """
    uvs_src = []
    pts_src = []
    feats_src = []
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        if cache_id in CACHE:
            uv_src, feat_src, pt_src = CACHE[(cache_id, i)]
        else:
            kp_src, feat_src = detector.detectAndCompute(img_src, mask_src)
            if len(kp_src) == 0:
                uv_src = np.zeros((0, 2), dtype=np.int32)
                pt_src = np.zeros((0, 3), dtype=np.float32)
                feat_src = np.zeros(
                    (0, detector.descriptorSize), dtype=np.bool_
                )  # use weak dtype to avoid descriptor dtype being promoted unexpectedly when concatenating
            else:
                uv_src = np.array([k.pt for k in kp_src], dtype=np.int32)
                uv_src, uniq_ind = np.unique(uv_src, axis=0, return_index=True)
                feat_src = feat_src[uniq_ind]
                pt_src = util.transform(cld_src[uv_src[:, 1], uv_src[:, 0]], poses_src[i])
            if cache_id is not None:
                CACHE[(cache_id, i)] = (uv_src, feat_src, pt_src)
        uvs_src.append(uv_src)
        pts_src.append(pt_src)
        feats_src.append(feat_src)

    if debug > 1:
        for img, uv, feat in zip(imgs_src, uvs_src, feats_src):
            Mf12 = feat_mat(feat, feat_dst)
            util.plot_keypoints(img, img_dst, uv, uv_dst, Mf12, thresh_feat, show_immidiate=False)
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
            if len(matches) == 0:
                continue
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
