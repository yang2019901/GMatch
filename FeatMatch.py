## use ORB to match features between two images
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib import patches
from util import transform, depth2cld
import pickle
import cProfile
from scipy.spatial.transform import Rotation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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
    K1 = 16
    K2 = 16
    L1, L2 = Mh12.shape
    matches = []
    thresh_loss = 0.2  # if the average loss of adding `m` to matches is less than this, accept `m`

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
            # sample L indices from kp1
            indices = np.random.choice(L1, min(K1, L1), replace=False)
            dists = Mh12[indices, :]  # (K1, L2)
            closest_indices = np.argpartition(dists, K2, axis=1)[:, :K2]  # (K1, K2)
            indices = np.repeat(indices[:, np.newaxis], K2, axis=1)  # (K1*K2, )
            tmp = np.column_stack((indices.flatten(), closest_indices.flatten()))  # (K1*K2, 2)

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


def cluster_keypoints(kp1, cld1, des1, th_e=0.01, th_h=20):
    uv1 = np.array([kp.pt for kp in kp1], dtype=np.int32)
    pts1 = cld1[uv1[:, 1], uv1[:, 0]]
    Me11 = np.linalg.norm(pts1[:, np.newaxis, :] - pts1, axis=-1)
    Mh11 = GetHamMat(des1, des1)
    Z = linkage(squareform(Me11), method="single")
    clusters = fcluster(Z, t=th_e, criterion="distance")
    return clusters, Z


def visualize_clusters(kp1, clusters, img):
    cluster_ids = np.unique(clusters)
    print(len(cluster_ids))
    colors = plt.cm.get_cmap("tab20", len(cluster_ids))

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for cluster_id in cluster_ids:
        cluster_points = [kp1[i].pt for i in range(len(kp1)) if clusters[i] == cluster_id]
        cluster_points = np.array(cluster_points)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(cluster_id), label=f"Cluster {cluster_id}")
    plt.legend()
    plt.show()


def match_features(imgs_src, clds_src, img_dst, cld_dst):
    """imgs_src, clds_src: (N, H, W, 3)
    img_dst, cld_dst: (H, W, 3), (H, W, 3)
    """
    detector: cv2.ORB = cv2.ORB_create()
    detector.setMaxFeatures(100)

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

    detector.setMaxFeatures(500)
    kp, des_dst = detector.detectAndCompute(img_dst, None)
    uv = np.array([kp[i].pt for i in range(len(kp))], dtype=np.int32)
    pts_dst = cld_dst[uv[:, 1], uv[:, 0]]
    print(f"Number of keypoints: src: {len(pts_src)}, img2: {len(pts_dst)}")
    # plt.imshow(cv2.drawKeypoints(img_dst, kp, None))
    # plt.show()

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
    imgs, clds, poses = pickle.load(open("jar.pt", "rb"))
    for i, cld in enumerate(clds):
        clds[i] = transform(cld, poses[i])

    img1, cld1 = imgs[2], clds[2]

    img2 = cv2.imread("bop_data/ycbv/test/000050/rgb/000001.png", cv2.IMREAD_COLOR_RGB)
    depth2 = cv2.imread("bop_data/ycbv/test/000050/depth/000001.png", cv2.IMREAD_UNCHANGED)
    mask2 = cv2.imread("bop_data/ycbv/test/000050/mask/000001_000000.png", cv2.IMREAD_UNCHANGED)

    intrin = np.array([1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]).reshape(3, 3)
    cld2 = depth2cld(depth2 * 0.0001, intrin)

    match_features(img1, img2, cld1, cld2, mask2=mask2)
