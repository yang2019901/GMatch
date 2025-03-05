## use ORB to match features between two images
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib import patches
import pickle
import cProfile
from scipy.spatial.transform import Rotation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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
    m = 100  # number of start
    D = 8  # search depth
    K = 32
    thresh_ham = 80
    thresh_loss = 0.1  # if the average loss of adding `m` to matches is less than this, accept `m`
    L1, L2 = Mh12.shape
    matches = []

    def loss(m):
        if len(matches) == 0:
            return 0
        m0, m1 = m[:, 0], m[:, 1]  # (Li, )
        i, j = np.array(matches).T  # (Di, )
        d1 = Me11[m0[:, np.newaxis], i]
        d2 = Me22[m1[:, np.newaxis], j]
        err = np.divide(np.abs(d1 - d2), d1, out=np.ones_like(d1), where=d1 != 0)  # (Li, Di), error rate
        res = np.sum(err, axis=-1)  # (Li, )
        return res

    part_indices = np.argpartition(np.reshape(Mh12, -1), m)[:m]
    good = np.array(np.unravel_index(part_indices, Mh12.shape)).T

    for i, j in good:
        matches.append((i, j))
        d_sum = 0
        # search for the next match
        while True:
            if len(matches) == D:
                break
            # sample K indices from kp1
            indices = np.random.choice(L1, min(K, L1), replace=False)
            # find all the points in kp2 that has Mh12 < thresh_ham
            dists = Mh12[indices, :]  # (K, L2)
            closest_indices = np.argwhere(dists < thresh_ham)
            if len(closest_indices) == 0:
                break
            tmp = np.column_stack((indices[closest_indices[:, 0]], closest_indices[:, 1]))

            # find the best match
            losses = loss(tmp)
            best_idx = np.argmin(losses)
            print(f"best loss: {losses[best_idx] / len(matches): .3f}")
            if losses[best_idx] / len(matches) < thresh_loss:
                # match point found
                matches.append(tmp[best_idx])
                d_sum += losses[best_idx]
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
    print(f"Average loss: {d_sum * 2 / (D * (D - 1)):.3f}")
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


def match_features(imgs_src, clds_src, img_dst, cld_dst):
    """imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3), (H, W, 3)
    """
    detector: cv2.ORB = cv2.ORB_create()
    detector.setMaxFeatures(200)

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
    pts_src = np.concatenate(pts_src, axis=0)  # (n1, 3), source points
    des_src = np.concatenate(des_src, axis=0)  # (n1, 32), source descriptors

    detector.setMaxFeatures(400)
    kp, des_dst = detector.detectAndCompute(img_dst, None)
    uv = np.array([kp[i].pt for i in range(len(kp))], dtype=np.int32)
    pts_dst = cld_dst[uv[:, 1], uv[:, 0]]
    print(f"Number of keypoints: src: {len(pts_src)}, img2: {len(pts_dst)}")
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
    imgs_src, clds_src, poses_src = pickle.load(open("box.pt", "rb"))
    imgs_dst, clds_dst, poses_dst = pickle.load(open("box_eval.pt", "rb"))
    for i in range(len(imgs_src)):
        imgs_src[i] = cv2.medianBlur(imgs_src[i], 7)
    img_dst, cld_dst = imgs_dst[0], clds_dst[0]
    img_dst = cv2.medianBlur(img_dst, 7)
    match_features(imgs_src, clds_src, img_dst, cld_dst)
