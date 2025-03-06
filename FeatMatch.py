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
        l = patches.ConnectionPatch(xyA=pt1, xyB=pt2, axesA=ax1, axesB=ax2, coordsA="data", coordsB="data", color="red")
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


def tree_search(Me11, Me22, Mh12):
    N_good = 100  # number of good matches
    D = 16  # search depth
    K = 32
    thresh_ham = 100
    thresh_loss = 0.1  # if the average loss of adding `m` to matches is less than this, accept `m`
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

    part_indices = np.argpartition(np.reshape(Mh12, -1), N_good)[:N_good]
    good = np.array(np.unravel_index(part_indices, Mh12.shape)).T

    for i, j in good:
        matches.append((i, j))
        # search for the next match
        while True:
            if len(matches) == D:
                break
            # sample K indices from kp1
            indices = np.random.choice(n1, min(K, n1), replace=False)
            # find all the points in kp2 that has Mh12 < thresh_ham
            dists = Mh12[indices, :]  # (K, L2)
            closest_indices = np.argwhere(dists < thresh_ham)
            if len(closest_indices) == 0:
                break
            tmp = np.column_stack((indices[closest_indices[:, 0]], closest_indices[:, 1]))

            # find the best match
            losses = loss(tmp)
            best_idx = np.argmin(losses)
            print(f"best loss: {losses[best_idx]:.3f}")
            if losses[best_idx] < thresh_loss:
                # match point found
                matches.append(tmp[best_idx])
            else:
                # no match point among these L points, maybe `start` is wrong
                break
        if len(matches) == D:
            break
        else:
            print("clear matches, re-search.")
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
    detector.setMaxFeatures(300)

    uv_ex = []
    ## find the keypoints and descriptors with detector
    pts_src, des_src = [], []
    for i, (img_src, cld_src) in enumerate(zip(imgs_src, clds_src)):
        kp, des = detector.detectAndCompute(img_src, None)
        if len(kp) == 0:
            continue
        uv = np.array([k.pt for k in kp], dtype=np.int32)
        uv_ex.extend([(i, u[0], u[1]) for u in uv])
        pts_src.append(cld_src[uv[:, 1], uv[:, 0]])
        des_src.append(des)
        plt.figure()
        plt.imshow(cv2.drawKeypoints(img_src, kp, None))
    pts_src = np.concatenate(pts_src, axis=0)  # (n1, 3), source points
    des_src = np.concatenate(des_src, axis=0)  # (n1, 32), source descriptors

    detector.setMaxFeatures(500)
    kp, des_dst = detector.detectAndCompute(img_dst, mask2)
    if len(kp) == 0:
        print("No keypoints found in img2")
        return
    uv = np.array([kp[i].pt for i in range(len(kp))], dtype=np.int32)
    pts_dst = cld_dst[uv[:, 1], uv[:, 0]]
    print(f"Number of keypoints: src: {len(pts_src)}, img2: {len(pts_dst)}")
    plt.figure()
    plt.imshow(cv2.drawKeypoints(img_dst, kp, None))
    plt.show()

    Mh12 = GetHamMat(des_src, des_dst)
    Me11 = np.linalg.norm(pts_src[:, np.newaxis, :] - pts_src, axis=-1)
    Me22 = np.linalg.norm(pts_dst[:, np.newaxis, :] - pts_dst, axis=-1)

    matches = tree_search(Me11, Me22, Mh12)
    ## parse matches
    dic = {}
    for i, j in matches:
        idx_img = uv_ex[i][0]
        if idx_img not in dic:
            dic[idx_img] = []
        dic[idx_img].append((uv_ex[i][1:], uv[j]))
    for idx_img, matches in dic.items():
        img_src = imgs_src[idx_img]
        cld_src = clds_src[idx_img]
        plot_matches(img_src, img_dst, *zip(*matches))


if __name__ == "__main__":
    """load model images"""
    imgs_src, clds_src, poses_src = pickle.load(open("cookies.pt", "rb"))
    for i, cld in enumerate(clds_src):
        clds_src[i] = util.transform(cld, poses_src[i])

    """ load scene image """
    img_dst = cv2.imread("bop_data/hope/val/000002/rgb/000000.png", cv2.IMREAD_COLOR_RGB)
    depth2 = cv2.imread("bop_data/hope/val/000002/depth/000000.png", cv2.IMREAD_UNCHANGED)
    mask2 = cv2.imread("bop_data/hope/val/000002/mask/000000_000008.png", cv2.IMREAD_UNCHANGED)
    cld_dst = util.depth2cld(depth2 * 0.001, [1390.53, 0.0, 964.957, 0.0, 1386.99, 522.586, 0.0, 0.0, 1.0])
    # img_dst = cv2.medianBlur(img_dst, 5)
    # imgs_dst, clds_dst, poses_dst = pickle.load(open("carbinet_eval2.pt", "rb"))
    # img_dst, cld_dst = imgs_dst[1], clds_dst[1]
    # mask2 = np.zeros(img_dst.shape[:2], dtype=np.uint8)
    # mask2[80:270, 220:390] = 255
    match_features(imgs_src, clds_src, img_dst, cld_dst, mask2)
