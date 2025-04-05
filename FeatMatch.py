""" use ORB/SIFT detector to match features between two images """

import numpy as np
import cv2
import util
import time


_ham_tab = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
cache = {}
detector: cv2.SIFT = cv2.SIFT_create()
detector.setContrastThreshold(0.03)

N1 = 500
N2 = 500
N_good = 25  # number of good matches
D = 24  # max search depth
thresh_des = 100  # threshold for descriptor distance, used to judge two descriptors' similarity
thresh_cost = 0.08  # if the maximum cost of adding `m` to matches is less than this, accept `m`
thresh_flip = 0.05  # threshold for flipover judgement


def GetHamMat(des1, des2):
    """compute hamming distance matrix `Mh` between two descriptors
    Mh[i, j] == HamDist(des1[i], des2[j])
    """
    global _ham_tab
    ## broadcast des1 and des2
    des1_ = des1[:, np.newaxis, :]
    des2_ = des2[np.newaxis, :, :]
    ## compute xor result
    xor_result = des1_ ^ des2_
    ## compute hamming distance
    hamming_distances = _ham_tab[xor_result]
    ## sum along the last axis to get the hamming distance matrix
    Mh = np.sum(hamming_distances, axis=-1)
    return Mh


def tree_search(pts1, pts2, Mh12):
    n1, n2 = Mh12.shape
    matches = []
    rlt = []

    def cost(m):
        """m: (n, 2), matches: (d, 2)"""
        if len(matches) == 0:
            return 0
        m0, m1 = m[:, 0], m[:, 1]  # (n, )
        i, j = np.array(matches).T  # (d, )
        dist1 = Me11[m0[:, np.newaxis], i]
        dist2 = Me22[m1[:, np.newaxis], j]
        err = np.divide(np.abs(dist1 - dist2), dist1, out=np.ones_like(dist1), where=dist1 != 0)  # (n, d), error rate
        res = np.max(err, axis=-1)  # (n, )
        return res

    def flipover(m):
        """m: (n, 2), return (n, ) boolean array"""
        if len(matches) < 2:
            return np.zeros(len(m), dtype=bool)
        v1 = pts1[matches[-2][0]] - pts1[matches[-1][0]]
        v1_ = pts1[m[:, 0]] - pts1[matches[-1][0]]
        v2 = pts2[matches[-2][1]] - pts2[matches[-1][1]]
        v2_ = pts2[m[:, 1]] - pts2[matches[-1][1]]
        n1 = np.cross(v1, v1_) / np.linalg.norm(v1) / np.linalg.norm(v1_)
        n2 = np.cross(v2, v2_) / np.linalg.norm(v2) / np.linalg.norm(v2_)
        ## flipover threshold may need to be adjusted
        flags = np.bitwise_and(n1[:, 2] * n2[:, 2] < 0, np.abs(n1[:, 2] - n2[:, 2]) > thresh_flip)
        return flags

    Me11 = np.linalg.norm(pts1[:, np.newaxis, :] - pts1, axis=-1)
    Me22 = np.linalg.norm(pts2[:, np.newaxis, :] - pts2, axis=-1)
    part_indices = (
        np.argpartition(np.reshape(Mh12, -1), N_good)[:N_good] if N_good < n1 * n2 else np.arange(n1 * n2, dtype=int)
    )
    good = np.array(np.unravel_index(part_indices, Mh12.shape)).T

    for i, j in good:
        matches.append((i, j))
        ## step(), search for the next match
        while True:
            if len(matches) == D:
                break
            dists = Mh12
            pairs = np.argwhere(dists < thresh_des)
            if len(pairs) == 0:
                break
            matches_alt = np.column_stack((pairs[:, 0], pairs[:, 1]))  # (n, 2)

            ## filter with distance cost
            costs = cost(matches_alt)  # (n, )
            ind = np.argwhere(costs < thresh_cost).flatten()
            matches_alt = matches_alt[ind]
            costs = costs[ind]
            if len(matches_alt) == 0:
                break
            ## filter out the matches that will flip the normal
            flags = flipover(matches_alt)
            matches_alt = matches_alt[~flags]
            costs = costs[~flags]
            if len(matches_alt) == 0:
                break
            ## get the best match
            best = np.argmin(costs)
            matches.append(tuple(matches_alt[best]))

        if len(matches) > len(rlt):
            rlt = matches.copy()
        matches.clear()
        if len(rlt) == D:
            break

    rlt = np.asarray(rlt)
    if len(rlt) < 3:
        return np.array([]), 1
    else:
        dists1 = Me11[rlt[:, 0][:, np.newaxis], rlt[:, 0]]
        dists2 = Me22[rlt[:, 1][:, np.newaxis], rlt[:, 1]]
        diff = np.abs(np.divide(dists1 - dists2, dists1, out=np.zeros_like(dists1), where=dists1 != 0))
        return np.array(rlt), np.max(diff).item()


def match_features(match_data: util.MatchData, cache_id=None):
    """match imgs_src and img_dst in match_data and store the result in it
    imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3), (H, W, 3)
    """
    global detector
    assert len(match_data.imgs_src) > 0, "imgs_src is empty"
    """ load from match_data """
    imgs_src, clds_src, masks_src = match_data.imgs_src, match_data.clds_src, match_data.masks_src
    img_dst, cld_dst, mask_dst = match_data.img_dst, match_data.cld_dst, match_data.mask_dst
    kp_dst, des_dst = detector.detectAndCompute(img_dst, mask_dst)
    if len(kp_dst) == 0:
        print("No keypoints found in img2")
        return
    uv_dst = np.array([k.pt for k in kp_dst], dtype=np.int32)
    pts_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    """ find the keypoints and descriptors with detector """
    if masks_src is None:
        masks_src = [None] * len(imgs_src)
    matches_list = []
    uvs_src = []
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        kp_src, des_src = (
            cache[(cache_id, i)] if (cache_id, i) in cache else detector.detectAndCompute(img_src, mask_src)
        )

        if cache_id is not None:
            cache[(cache_id, i)] = (kp_src, des_src)

        if len(kp_src) == 0:
            matches_list.append(([], 1))
            continue
        uv_src = np.array([k.pt for k in kp_src], dtype=np.int32)
        pts_src = cld_src[uv_src[:, 1], uv_src[:, 0]]
        uvs_src.append(uv_src)

        """ L1-normalized SIFT """
        des_src = des_src / np.sum(des_src, axis=-1, keepdims=True)
        des_dst = des_dst / np.sum(des_dst, axis=-1, keepdims=True)
        Mh12 = np.linalg.norm(des_src[:, np.newaxis, :] - des_dst[np.newaxis, :, :], axis=-1)

        """ <Tune>
            N1 and N2: plot to see whether keypoints are enough
            thresh_des: find a suitable threshold for descriptor distance
        """
        # util.plot_keypoints(img_src, img_dst, kp_src, kp_dst, Mh12, thresh_des)

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
