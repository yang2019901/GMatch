""" use ORB/SIFT detector to match features between two images """

import numpy as np
import cv2
import util
import time
import matplotlib.pyplot as plt

from lightglue import SuperPoint, DISK
from lightglue.utils import numpy_image_to_torch


HAM_TAB = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)  # used to compute hamming distance, only ORB uses it now
CACHE = {}

""" SIFT settings """
# detector: cv2.SIFT = cv2.SIFT_create()
# detector.setContrastThreshold(0.03)
# N_good = 32  # number of good matches candidates
# D = 24  # max search depth
# thresh_des = 0.1  # threshold for descriptor distance, used to judge two descriptors' similarity
# thresh_cost = 0.08  # if the maximum cost of adding `m` to matches is less than this, accept `m`

""" ORB settings """
# detector = cv2.ORB_create()
# N1 = 500
# N2 = 500
# N_good = 50  # number of good matches
# D = 24  # max search depth
# thresh_des = 100  # threshold for descriptor distance, used to judge two descriptors' similarity
# thresh_cost = 0.08  # if the maximum cost of adding `m` to matches is less than this, accept `m`

""" SuperPoint settings """
detector: SuperPoint = SuperPoint(max_num_keypoints=512).eval()
N_good = 32  # number of good matches candidates
D = 24  # max search depth
thresh_des = 1.25  # threshold for descriptor distance, used to judge two descriptors' similarity
thresh_cost = 0.08  # if the maximum cost of adding `m` to matches is less than this, accept `m`
thresh_flip = 0.01  # threshold for flipover judgement


def GetHamMat(des1, des2):
    """compute hamming distance matrix `Mh` between two descriptors
    Mh[i, j] == HamDist(des1[i], des2[j])
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
    Mh = np.sum(hamming_distances, axis=-1)
    return Mh


def cost(matches, pairs, Me11, Me22):
    """geometric cost function
    matches: (d, 2), pairs: (n, 2), Me11: (n1, n1), Me22: (n2, n2)
    """
    if len(matches) == 0:
        return 0
    m0, m1 = pairs[:, 0], pairs[:, 1]  # (n, )
    i, j = np.array(matches).T  # (d, )
    dist1 = Me11[m0[:, np.newaxis], i]
    dist2 = Me22[m1[:, np.newaxis], j]
    err = np.divide(np.abs(dist1 - dist2), dist1, out=np.ones_like(dist1), where=dist1 != 0)  # (n, d), error rate
    res = np.max(err, axis=-1)  # (n, )
    return res


def flipover(matches, pairs, pts1, pts2):
    """pairs: (n, 2), return (n, ) boolean array"""
    global thresh_flip
    if len(matches) < 2:
        return np.zeros(len(pairs), dtype=bool)
    v1_1 = pts1[matches[-2][0]] - pts1[matches[-1][0]]
    v1_2 = pts1[pairs[:, 0]] - pts1[matches[-1][0]]
    v2_1 = pts2[matches[-2][1]] - pts2[matches[-1][1]]
    v2_2 = pts2[pairs[:, 1]] - pts2[matches[-1][1]]
    n1 = np.cross(v1_1, v1_2) / np.linalg.norm(v1_1) / np.linalg.norm(v1_2)
    n2 = np.cross(v2_1, v2_2) / np.linalg.norm(v2_1) / np.linalg.norm(v2_2)
    flags = np.bitwise_and(n1[:, 2] * n2[:, 2] < 0, np.abs(n1[:, 2] - n2[:, 2]) > thresh_flip)
    return flags


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


def tree_search(pts1, pts2, Mh12, **dbg):
    n1, n2 = Mh12.shape
    matches = []
    rlt = []
    Me11 = np.linalg.norm(pts1[:, np.newaxis, :] - pts1, axis=-1)
    Me22 = np.linalg.norm(pts2[:, np.newaxis, :] - pts2, axis=-1)

    part_indices = (
        np.argpartition(np.reshape(Mh12, -1), N_good)[:N_good] if N_good < n1 * n2 else np.arange(n1 * n2, dtype=int)
    )
    pairs_good = np.array(np.unravel_index(part_indices, Mh12.shape)).T
    # pairs_good = np.argwhere(Mh12 < 0.1)

    pairs_simi = np.argwhere(Mh12 < thresh_des)
    if len(pairs_simi) == 0:
        return np.array([]), 1

    for i, j in pairs_good:
        matches.append((i, j))
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
            matches.append(tuple(pairs[best]))

        if len(matches) > len(rlt):
            rlt = matches
        matches = []
        if len(rlt) == D:
            break

    rlt = np.asarray(rlt)
    if len(rlt) < 3:
        return np.array([]), 1
    else:
        dists1 = Me11[rlt[:, 0][:, np.newaxis], rlt[:, 0][np.newaxis, :]]  # (d, d)
        dists2 = Me22[rlt[:, 1][:, np.newaxis], rlt[:, 1][np.newaxis, :]]  # (d, d)
        diff = np.abs(np.divide(dists1 - dists2, dists1, out=np.zeros_like(dists1), where=dists1 != 0))
        return np.array(rlt), np.max(diff).item()


def SPP_detect(img, mask):
    global detector
    tmp = (img * (mask[:, :, np.newaxis] > 0)).astype(np.uint8)
    feat = detector.extract(numpy_image_to_torch(tmp))
    return feat["keypoints"].cpu().numpy().astype(np.int32).squeeze(), feat["descriptors"].cpu().numpy().squeeze()


def match_features(match_data: util.MatchData, cache_id=None):
    """match imgs_src and img_dst in match_data and store the result in it
    imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3), (H, W, 3)
    """
    global detector, CACHE
    assert len(match_data.imgs_src) > 0, "imgs_src is empty"
    """ load from match_data """
    imgs_src, clds_src, masks_src = match_data.imgs_src, match_data.clds_src, match_data.masks_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst
    # kp_dst, des_dst = detector.detectAndCompute(img_dst, mask_dst)
    uv_dst, des_dst = SPP_detect(img_dst, mask_dst)

    if len(uv_dst) == 0:
        print("No keypoints found in img2")
        return
    pts_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    """ find the keypoints and descriptors with detector """
    if masks_src is None:
        masks_src = [None] * len(imgs_src)
    matches_list = []
    uvs_src = []
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        # kp_src, des_src = (
        #     CACHE[(cache_id, i)] if (cache_id, i) in CACHE else detector.detectAndCompute(img_src, mask_src)
        # )
        uv_src, des_src = CACHE[(cache_id, i)] if (cache_id, i) in CACHE else SPP_detect(img_src, mask_src)

        if cache_id is not None:
            CACHE[(cache_id, i)] = (uv_src, des_src)

        if len(uv_src) == 0:
            matches_list.append(([], 1))
            continue
        pts_src = cld_src[uv_src[:, 1], uv_src[:, 0]]
        uvs_src.append(uv_src)

        # """ L1-normalized SIFT """
        # des_src = des_src / np.sum(des_src, axis=-1, keepdims=True)
        # des_dst = des_dst / np.sum(des_dst, axis=-1, keepdims=True)
        # Mh12 = np.linalg.norm(des_src[:, np.newaxis, :] - des_dst[np.newaxis, :, :], axis=-1)
        Mh12 = np.linalg.norm(des_src[:, np.newaxis, :] - des_dst[np.newaxis, :, :], axis=-1)

        """ <Tune>
            N1 and N2: plot to see whether keypoints are enough
            thresh_des: find a suitable threshold for descriptor distance
        """
        # global thresh_des
        # util.plot_keypoints(img_src, img_dst, uv_src, uv_dst, Mh12, thresh_des)

        matches, cost = tree_search(pts_src, pts_dst, Mh12)
        matches_list.append((matches, cost))
        if len(matches) == D:
            break

        """ visualization """
        # if len(matches) < 3:
        #     print(f"\timgs_src[{i}]: matches NOT found.")
        # else:
        #     print(f"\timgs_src[{i}]: matches found. depth {len(matches)}, cost {cost:.3f}")
        #     util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])

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
