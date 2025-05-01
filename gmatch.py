""" Use ORB/SIFT detector to match features between two images """

import numpy as np
import cv2
import util
import time
import matplotlib.pyplot as plt


HAM_TAB = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)  # used to compute hamming distance, only ORB uses it now
CACHE = {}  # cache for keypoints and features of imgs_src

""" SIFT settings """
detector: cv2.SIFT = cv2.SIFT_create()
detector.setContrastThreshold(0.03)
N_good = 32  # number of good matches candidates
D = 24  # max search depth
thresh_feat = 0.1  # threshold for feature distance, used to judge the similarity of two feature vectors
thresh_cost = 0.08  # if the maximum cost of adding `m` to matches is less than this, accept `m`
thresh_flip = 0.8  # threshold for flipover judgement
feat_mat = lambda des1, des2: sift_mat(des1, des2)  # feature distance matrix


""" ORB settings """
# detector = cv2.ORB_create(scaleFactor=1.4)
# N_good = 30  # number of good matches
# D = 24  # max search depth
# thresh_feat = 90  # threshold for feature distance, used to judge the similarity of two feature vectors
# thresh_cost = 0.08  # if the maximum cost of adding `m` to matches is less than this, accept `m`
# thresh_flip = 0.8  # threshold for flipover judgement
# feat_mat = lambda des1, des2: orb_mat(des1, des2)  # feature distance matrix


def orb_mat(des1, des2):
    """compute feature distance matrix `Mf` for ORB, whose metric is Hamming distance.
    > Mh[i, j] == HamDist(des1[i], des2[j])

    - Input:
        des1: (n1, 32), uint8
        des2: (n2, 32), uint8
    """
    global HAM_TAB
    ## broadcast des1 and des2
    des1_ = des1[:, np.newaxis, :]
    des2_ = des2[np.newaxis, :, :]
    ## compute xor result
    xor_result = des1_ ^ des2_
    ## compute hamming distance
    hamming_distances = HAM_TAB[xor_result]
    ## sum along the last axis to get the hamming distance matrix
    Mf = np.sum(hamming_distances, axis=-1)
    return Mf


def sift_mat(des1, des2):
    """compute feature distance matrix `Mf` for SIFT, whose metric is Euclidean distance.

    Note: des1 and des2 will be L1-normalized

    - Input:
        des1: (n1, 128), integer stored in float32
        des2: (n2, 128), integer stored in float32
    """
    des1_ = des1 / np.sum(des1, axis=-1, keepdims=True)
    des2_ = des2 / np.sum(des2, axis=-1, keepdims=True)
    Mf = np.linalg.norm(des1_[:, np.newaxis, :] - des2_[np.newaxis, :, :], axis=-1)
    return Mf


def cost(matches, pairs, Me11, Me22):
    """cost function for distance matrix.
    matches: (d, 2), pairs: (n, 2), Me11: (n1, n1), Me22: (n2, n2)
    """
    if len(matches) == 0:
        return 0
    m0, m1 = pairs[:, 0], pairs[:, 1]  # (n, )
    i, j = zip(*matches)  # (d, )
    dist1 = Me11[m0[:, np.newaxis], i]
    dist2 = Me22[m1[:, np.newaxis], j]
    err = np.divide(np.abs(dist1 - dist2), dist1, out=np.ones_like(dist1), where=dist1 != 0)  # (n, d), error rate
    res = np.max(err, axis=-1)  # (n, )
    return res


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
    # n1 = np.cross(v1_1, v1_2) / np.linalg.norm(v1_1) / np.linalg.norm(v1_2)
    # n2 = np.cross(v2_1, v2_2) / np.linalg.norm(v2_1) / np.linalg.norm(v2_2)
    flags = np.bitwise_and(n1[:, 2] * n2[:, 2] < 0, np.abs(n1[:, 2] - n2[:, 2]) > thresh_flip)
    return flags


def search(pts1, pts2, Mf12):
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
    if len(pairs_simi) == 0:
        return np.array([]), 1

    for i, j in pairs_good:
        matches.append((i, j))
        c = 0
        ## step(), search for the next match
        while True:
            if len(matches) == D:
                break
            ## filter with geometric cost
            costs = cost(matches, pairs_simi, Me11, Me22)  # (n, )
            ind = np.argwhere(costs < thresh_cost).flatten()
            pairs = pairs_simi[ind]
            costs = costs[ind]
            if len(pairs) == 0:
                break
            flags = ~flipover(matches, pairs, pts1, pts2)  # (n, )
            pairs = pairs[flags]
            costs = costs[flags]
            if len(pairs) == 0:
                break
            ## get the best match
            best = np.argmin(costs)
            c = max(c, costs[best])
            matches.append(tuple(pairs[best]))

        if len(matches) > len(rlt):
            rlt = matches
            rlt_cost = c
        matches = []
        if len(rlt) == D:
            break

    rlt = np.asarray(rlt)
    return (rlt, rlt_cost) if len(rlt) >= 3 else (np.array([]), 1)


def Match(match_data: util.MatchData, cache_id=None):
    """match each of imgs_src with img_dst in match_data and store the result in it;
        keypoints and features for imgs_src will be cached with cache_id if provided
    imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3)
    """
    global detector, CACHE
    assert len(match_data.imgs_src) > 0, "imgs_src is empty"
    """ load from match_data """
    imgs_src, clds_src, masks_src = match_data.imgs_src, match_data.clds_src, match_data.masks_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst

    kp_dst, des_dst = detector.detectAndCompute(img_dst, mask_dst)  # 0.3s for 1920x1080 => 0.014s for 211x200
    if len(kp_dst) == 0:
        print("No keypoints found in img2")
        return
    uv_dst = np.array([k.pt for k in kp_dst], dtype=np.int32)
    pts_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    """ extract the keypoints and features with descriptor """
    if masks_src is None:
        masks_src = [None] * len(imgs_src)
    matches_list = []
    uvs_src = []
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        kp_src, des_src = (
            CACHE[(cache_id, i)] if (cache_id, i) in CACHE else detector.detectAndCompute(img_src, mask_src)
        )

        if cache_id is not None:
            CACHE[(cache_id, i)] = (kp_src, des_src)

        if len(kp_src) == 0:
            matches_list.append(([], 1))
            continue
        uv_src = np.array([k.pt for k in kp_src], dtype=np.int32)
        pts_src = cld_src[uv_src[:, 1], uv_src[:, 0]]
        uvs_src.append(uv_src)

        """ Feature Distance Matrix (for visual similarity) """
        Mf12 = feat_mat(des_src, des_dst)

        """ <Tune>
            N1 and N2: plot to see whether keypoints are enough
            thresh_des: find a suitable threshold for feature distance
        """
        # util.plot_keypoints(img_src, img_dst, uv_src, uv_dst, Mf12, thresh_des)

        matches, cost = search(pts_src, pts_dst, Mf12)
        matches_list.append((matches, cost))

        """ visualization """
        # if len(matches) < 3:
        #     print(f"\timgs_src[{i}]: matches NOT found.")
        # else:
        #     print(f"\timgs_src[{i}]: matches found. depth {len(matches)}, cost {cost:.3f}")
        #     util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])

        if len(matches) == D:
            break

    """ take max depth matches as the best """
    match_data.matches_list, match_data.cost_list = zip(*matches_list)
    match_data.uvs_src = uvs_src
    match_data.uv_dst = uv_dst
    match_data.idx_best = np.argmax([len(matches) for matches, _ in matches_list])

    """ visualization """
    # if len(match_data.matches_list[match_data.idx_best]) > 0:
    #     img_src = imgs_src[match_data.idx_best]
    #     uv_src = uvs_src[match_data.idx_best]
    #     matches = matches_list[match_data.idx_best][0]
    #     util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])
    return
