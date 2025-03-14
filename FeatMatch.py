## use ORB to match features between two images
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib import patches
import pickle
import cProfile

import util

np.random.seed(0)
_ham_tab = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


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
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+380+310")
    fig.tight_layout()
    plt.show()
    return


def HamDist(a, b):
    rlt = 0
    for i in range(len(a)):
        rlt += _ham_tab[a[i] ^ b[i]]
    return rlt


def tree_search(pts1, pts2, Mh12):
    N_good = 100  # number of good matches
    D = 16  # search depth
    K = 256
    thresh_ham = 100  # threshold for hamming distance
    thresh_loss = 0.1  # if the maximum loss of adding `m` to matches is less than this, accept `m`
    n1, n2 = Mh12.shape
    matches = []

    def loss(m):
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
        """ m: (n, 2), return (n, ) boolean array"""
        if len(matches) < 2:
            return np.zeros(len(m), dtype=bool)
        v1 = pts1[matches[-2][0]] - pts1[matches[-1][0]]
        v1_ = pts1[m[:, 0]] - pts1[matches[-1][0]]
        v2 = pts2[matches[-2][1]] - pts2[matches[-1][1]]
        v2_ = pts2[m[:, 1]] - pts2[matches[-1][1]]
        n1 = np.cross(v1, v1_) / np.linalg.norm(v1) / np.linalg.norm(v1_)
        n2 = np.cross(v2, v2_) / np.linalg.norm(v2) / np.linalg.norm(v2_)
        flags = np.bitwise_and(n1[:, 2] * n2[:, 2] < 0, np.abs(n1[:, 2] - n2[:, 2]) > 0.1)
        return flags

    Me11 = np.linalg.norm(pts1[:, np.newaxis, :] - pts1, axis=-1)
    Me22 = np.linalg.norm(pts2[:, np.newaxis, :] - pts2, axis=-1)
    part_indices = np.argpartition(np.reshape(Mh12, -1), N_good)[:N_good]
    good = np.array(np.unravel_index(part_indices, Mh12.shape)).T

    for i, j in good:
        matches.append((i, j))
        ## search for the next match
        while True:
            if len(matches) == D:
                break
            ## sample K indices from kp1
            pts1_ind = np.random.choice(n1, min(K, n1), replace=False)
            ## find all the points in kp2 that has Mh12 < thresh_ham
            dists = Mh12[pts1_ind, :]  # (K, L2)
            pairs = np.argwhere(dists < thresh_ham)
            if len(pairs) == 0:
                break
            matches_alt = np.column_stack((pts1_ind[pairs[:, 0]], pairs[:, 1])) # (n, 2)

            ## filter with distance loss
            losses = loss(matches_alt) # (n, )
            ind = np.argwhere(losses < thresh_loss).flatten()
            matches_alt = matches_alt[ind]
            losses = losses[ind]
            if len(matches_alt) == 0:
                break
            ## filter out the matches that will flip the normal
            flags = flipover(matches_alt)
            matches_alt = matches_alt[~flags]
            if len(matches_alt) == 0:
                break
            ## get the best match
            best = np.argmin(losses[~flags])
            matches.append(tuple(matches_alt[best]))

        if len(matches) == D:
            break
        else:
            # print("clear matches, re-search.")
            matches.clear()

    if len(matches) < D:
        print("Not enough matches found")
    return np.array(matches)


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


def match_features(imgs_src, clds_src, img_dst, cld_dst, mask2=None):
    """imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3), (H, W, 3)
    """
    detector: cv2.ORB = cv2.ORB_create()

    detector.setMaxFeatures(500)
    kp_dst, des_dst = detector.detectAndCompute(img_dst, mask2)
    if len(kp_dst) == 0:
        print("No keypoints found in img2")
        return
    uv_dst = np.array([k.pt for k in kp_dst], dtype=np.int32)
    pts_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    ## find the keypoints and descriptors with detector
    detector.setMaxFeatures(600)
    for _, (img_src, cld_src) in enumerate(zip(imgs_src, clds_src)):
        kp_src, des_src = detector.detectAndCompute(img_src, None)
        if len(kp_src) == 0:
            continue
        uv_src = np.array([k.pt for k in kp_src], dtype=np.int32)
        pts_src = cld_src[uv_src[:, 1], uv_src[:, 0]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.tight_layout()
        ax1.set_title(f"keypoints: {len(pts_src)}")
        ax1.imshow(cv2.drawKeypoints(img_src, kp_src, None))
        ax2.set_title(f"keypoints: {len(pts_dst)}")
        ax2.imshow(cv2.drawKeypoints(img_dst, kp_dst, None))
        plt.show()

        Mh12 = GetHamMat(des_src, des_dst)
        matches = tree_search(pts_src, pts_dst, Mh12)
        if len(matches) != 0:
            print(f"matches found: {matches}")
            plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])
            break


if __name__ == "__main__":
    """ load model images """
    imgs_src, clds_src, poses_src = pickle.load(open("ycbv/20-extra_large_clamp.pt", "rb"))

    """ load scene image """
    img_dst = cv2.imread("bop_data/ycbv/test/000048/rgb/000001.png", cv2.IMREAD_COLOR_RGB)
    depth2 = cv2.imread("bop_data/ycbv/test/000048/depth/000001.png", cv2.IMREAD_UNCHANGED)
    mask2 = cv2.imread("bop_data/ycbv/test/000048/mask/000001_000004.png", cv2.IMREAD_UNCHANGED)
    cld_dst = util.depth2cld(depth2 * 0.0001, [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0])

    """ match features """
    match_features(imgs_src, clds_src, img_dst, cld_dst, mask2)

    # imgs_src, clds_src, poses_src = pickle.load(open("carbinet.pt", "rb"))
    # imgs_dst, clds_dst, poses_dst = pickle.load(open("carbinet_eval2.pt", "rb"))

    # img_dst = imgs_dst[3]
    # cld_dst = clds_dst[3]
    # mask = np.zeros(img_dst.shape[:2], dtype=np.uint8)
    # mask[50:300, 200:400] = 255
    # match_features(imgs_src, clds_src, img_dst, cld_dst, mask)
