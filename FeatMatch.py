## use ORB to match features between two images
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib import patches
import open3d as o3d
import pickle
import cProfile
from scipy.spatial.transform import Rotation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

np.random.seed(0)
_ham_tab = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def plot_matches(img1, img2, kp1, kp2, matches):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax2.imshow(img2)
    for i, j in matches:
        pt1 = kp1[i].pt
        pt2 = kp2[j].pt
        pt2 = (pt2[0] + img1.shape[1], pt2[1])
        l = patches.ConnectionPatch(xyA=pt1, xyB=pt2, axesA=ax1, axesB=ax2)
        fig.add_artist(l)
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+380+310")
    fig.tight_layout()
    plt.show()
    return


def plot_matches_3d(pcd1, pcd2, m_pts1, m_pts2):
    # 创建一个新的 LineSet 对象
    L = len(m_pts1)
    lines = [(i, i + L) for i in range(L)]
    colors = [(1, 0, 0) for _ in range(L)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((m_pts1, m_pts2)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # 可视化点云和连线
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd1, pcd2, line_set, axes])


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
    thresh_loss = 0.05  # if the average loss of adding `m` to matches is less than this, accept `m`

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


def get_pcd(pts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
    if color is not None:
        c = np.asarray(color, dtype=np.float32) / 255 if color.dtype==np.uint8 else color
        pcd.colors = o3d.utility.Vector3dVector(c.reshape(-1, 3))
    return pcd


def match_features(imgs_src, clds_src, img_dst, cld_dst):
    detector: cv2.ORB = cv2.ORB_create()
    detector.setMaxFeatures(100)

    ## find the keypoints and descriptors with detector
    pts_src, des_src = [], []
    for img_src, cld_src in zip(imgs_src, clds_src):
        kp, des = detector.detectAndCompute(img_src, None)
        uv = np.array([kp[i].pt for i in range(len(kp))], dtype=np.int32)
        if len(uv) == 0:
            continue
        pts_src.append(cld_src[uv[:, 1], uv[:, 0]])
        des_src.append(des)
    pts_src = np.concatenate(pts_src, axis=0)  # (n1, 3), source points
    des_src = np.concatenate(des_src, axis=0)  # (n1, 32), source descriptors

    detector.setMaxFeatures(200)
    kp, des_dst = detector.detectAndCompute(img_dst, None)
    uv = np.array([kp[i].pt for i in range(len(kp))], dtype=np.int32)
    pts_dst = cld_dst[uv[:, 1], uv[:, 0]]
    print(f"Number of keypoints: src: {len(pts_src)}, img2: {len(pts_dst)}")

    Mh12 = GetHamMat(des_src, des_dst)
    Me11 = np.linalg.norm(pts_src[:, np.newaxis, :] - pts_src, axis=-1)
    Me22 = np.linalg.norm(pts_dst[:, np.newaxis, :] - pts_dst, axis=-1)

    matches = tree_search(Me11, Me22, Mh12)
    pcd1 = get_pcd(clds_src.reshape(-1, 3), imgs_src.reshape(-1, 3))
    pcd2 = get_pcd(cld_dst.reshape(-1, 3), img_dst.reshape(-1, 3))
    plot_matches_3d(pcd1, pcd2, pts_src[matches[:, 0]], pts_dst[matches[:, 1]])


def pose2Mat(pose):
    """pose: [[x, y, z], [qx, qy, qz, qw]]"""
    M = np.eye(4)
    M[:3, 3] = pose[0]
    M[:3, :3] = Rotation.from_quat(pose[1]).as_matrix()
    return M


def transform_clds(clds, poses):
    clds_transformed = []
    for cld, pose in zip(clds, poses):
        M = pose2Mat(pose)
        cld_transformed = cld @ M[:3, :3].T + M[:3, 3]
        clds_transformed.append(cld_transformed)
    return np.array(clds_transformed)


if __name__ == "__main__":
    imgs, clds, poses = pickle.load(open("jar.pt", "rb"))
    clds = transform_clds(clds, poses)
    img1, cld1 = imgs[0], clds[0] + 0.1
    match_features(imgs, clds, img1, cld1)
