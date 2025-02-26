## use orb/sift to match features between two images
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import pickle
import cProfile
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# np.random.seed(0)
_ham_tab = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def plot_matches(img1, img2, kp1, kp2, matches):
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(np.hstack((cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))))
    ax.set_title("Matches")
    for i, j in matches:
        pt1 = kp1[i].pt
        pt2 = kp2[j].pt
        pt2 = (pt2[0] + img1.shape[1], pt2[1])
        ax.plot(pt1[0], pt1[1], "ro")
        ax.plot(pt2[0], pt2[1], "ro")
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "c")
    plt.show()
    return


def HamDist(a, b):
    rlt = 0
    for i in range(len(a)):
        rlt += _ham_tab[a[i] ^ b[i]]
    return rlt


def tree_search(kp1, kp2, Me11, Me22, Mh12):
    m = 100  # number of start
    D = 8  # search depth
    L = 16
    K = 16
    matches = []
    thresh_loss = 0.01  # if the average loss of adding `m` to matches is less than this, accept `m`. unit: meter

    def loss(m):
        if len(matches) == 0:
            return 0
        m0, m1 = m[:, 0], m[:, 1]
        i, j = np.array(matches).T
        res = np.sum(np.abs(Me11[m0[:, np.newaxis], i] - Me22[m1[:, np.newaxis], j]), axis=-1)
        return res

    part_indices = np.argpartition(np.reshape(Mh12, -1), m)[:m]
    good = np.array(np.unravel_index(part_indices, Mh12.shape)).T

    for i, j in good:
        matches.append((i, j))
        plot_matches(img1, img2, kp1, kp2, matches)
        d_sum = 0
        # search for the next match
        while True:
            if len(matches) == D:
                break
            # sample L indices from kp1
            indices = np.random.choice(len(kp1), min(L, len(kp1)), replace=False)
            dists = Mh12[indices, :] # (L, n2)
            closest_indices = np.argpartition(dists, K, axis=1)[:, :K] # (L, K)
            indices = np.repeat(indices[:, np.newaxis], K, axis=1)
            tmp = np.column_stack((indices.flatten(), closest_indices.flatten()))

            # find the best match
            losses = loss(tmp)
            best_idx = np.argmin(losses)
            print(f"best loss: {losses[best_idx] / len(matches): .3f}")
            if losses[best_idx] / len(matches) < thresh_loss:
                # match point found
                matches.append(tmp[best_idx])
                plot_matches(img1, img2, kp1, kp2, matches)
                d_sum += losses[best_idx]
            else:
                # no match point among these L points, maybe `start` is wrong
                break
        if len(matches) == D:
            break
        else:
            matches.clear()

    if len(matches) < D:
        print("Not enough matches found")
    print(f"Average loss: {d_sum * 2 / (D * (D - 1)):.3f}")
    return [cv2.DMatch(i, j, 0) for i, j in matches]


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
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for cluster_id in cluster_ids:
        cluster_points = [kp1[i].pt for i in range(len(kp1)) if clusters[i] == cluster_id]
        cluster_points = np.array(cluster_points)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(cluster_id), label=f"Cluster {cluster_id}")
    plt.legend()
    plt.show()


def match_features(img1, img2, cld1, cld2, mask1=None, mask2=None):
    detector: cv2.ORB = cv2.ORB_create()
    detector.setMaxFeatures(200)

    ## find the keypoints and descriptors with detector
    # mask1 = np.zeros(img1.shape[:2], dtype=np.uint8)
    # mask1[0:380, 150:500] = 255
    kp1, des1 = detector.detectAndCompute(img1, mask1)
    kp2, des2 = detector.detectAndCompute(img2, mask2)
    print(f"Number of keypoints in img1: {len(kp1)}, img2: {len(kp2)}")

    # clusters, Z = cluster_keypoints(kp1, cld1, des1)
    # visualize_clusters(kp1, clusters, img1)

    Mh12 = GetHamMat(des1, des2)
    uv1 = np.array([kp.pt for kp in kp1], dtype=np.int32)
    uv2 = np.array([kp.pt for kp in kp2], dtype=np.int32)
    pts1 = cld1[uv1[:, 1], uv1[:, 0]]
    pts2 = cld2[uv2[:, 1], uv2[:, 0]]
    Me11 = np.linalg.norm(pts1[:, np.newaxis, :] - pts1, axis=-1)
    Me22 = np.linalg.norm(pts2[:, np.newaxis, :] - pts2, axis=-1)

    t0 = time.time()
    cProfile.runctx("matches = tree_search(kp1, kp2, Me11, Me22, Mh12)", globals(), locals(), "profile_data.prof")
    print(f"Time: {time.time() - t0}")
    exit(0)

    # draw the keypoints
    img1_with_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_with_keypoints = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

    # show the keypoints
    _, ax = plt.subplots(1, 3, width_ratios=[1, 1, 2])
    # hide the axes
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    ax[0].imshow(cv2.cvtColor(img1_with_keypoints, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(img2_with_keypoints, cv2.COLOR_BGR2RGB))
    ax[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # img1 = cv2.imread("img4.jpg", 0)
    # img2 = cv2.imread("img1.jpg", 0)

    # img1 = cv2.resize(img1, (480, 640))
    # img2 = cv2.resize(img2, (480, 640))

    imgs, clds, poses = pickle.load(open("carbinet.pt", "rb"))
    img1, img2, cld1, cld2 = imgs[0], imgs[2], clds[0], clds[2]
    match_features(img1, img2, cld1, cld2)
