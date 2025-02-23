## use orb/sift to match features between two images
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import pickle

np.random.seed(0)


def plot_matches(img1, img2, kp1, kp2, matches):
    # 创建一个图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # 在img1和img2上绘制关键点和匹配线
    ax.imshow(np.hstack((cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))))
    ax.set_title("Matches")

    for i, j in matches:
        pt1 = kp1[i].pt
        pt2 = kp2[j].pt
        pt2 = (pt2[0] + img1.shape[1], pt2[1])  # 调整pt2的x坐标以匹配拼接图像

        # 绘制关键点
        ax.plot(pt1[0], pt1[1], "ro")
        ax.plot(pt2[0], pt2[1], "ro")

        # 绘制匹配线
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "c")

    plt.show()


def HamDist(a, b):
    return np.sum([bin(a[i] ^ b[i]).count("1") for i in range(len(a))])


def tree_search(kp1, kp2, des1, des2, cld1, cld2):
    D = 8  # search depth
    L = 8
    K = 8
    matches = []
    thresh_loss = 0.01  # if the average loss of adding `m` to matches is less than this, accept `m`. unit: meter

    def loss(m):
        res = 0
        for i, j in matches:
            p1_1 = cld1[round(kp1[i].pt[1]), round(kp1[i].pt[0])]  # pair1 in img1
            p1_2 = cld2[round(kp2[j].pt[1]), round(kp2[j].pt[0])]  # pair1 in img2
            p2_1 = cld1[round(kp1[m[0]].pt[1]), round(kp1[m[0]].pt[0])]  # pair2 in img1
            p2_2 = cld2[round(kp2[m[1]].pt[1]), round(kp2[m[1]].pt[0])]  # pair2 in img2
            res += np.abs(np.linalg.norm(p1_1 - p2_1) - np.linalg.norm(p1_2 - p2_2))
        return res

    good = []
    for i in range(len(kp1)):
        for j in range(len(kp2)):
            if HamDist(des1[i], des2[j]) < 40:
                good.append((i, j))

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
            tmp = []
            # find top K closest keypoints in kp2
            for idx in indices:
                dists = [HamDist(des1[idx], des2[j]) for j in range(len(kp2))]
                closest = np.argsort(dists)[:K]
                tmp += [(idx, j) for j in closest]
            # find the best match
            best = min(tmp, key=loss)
            print(f"best loss: {loss(best) / len(matches): .3f}")
            if loss(best) / len(matches) < thresh_loss:
                # match point found
                matches.append(best)
                plot_matches(img1, img2, kp1, kp2, matches)
                d_sum += loss(best)
            else:
                # no match point among these L points, maybe `start` is wrong
                break
        if len(matches) == D:
            break
        else:
            matches.clear()

    print(f"Average loss: {d_sum / D}")
    return [cv2.DMatch(i, j, 0) for i, j in matches]


def match_features(img1, img2, cld1, cld2, mask1=None, mask2=None):
    detector: cv2.ORB = cv2.ORB_create()
    # detector.setMaxFeatures(500)
    # find the keypoints and descriptors with detector
    kp1, des1 = detector.detectAndCompute(img1, mask1)
    kp2, des2 = detector.detectAndCompute(img2, mask2)
    print(f"Number of keypoints in img1: {len(kp1)}, img2: {len(kp2)}")

    matches = tree_search(kp1, kp2, des1, des2, cld1, cld2)
    # matches = []

    # def plot(i):
    #     li = np.array([HamDist(des1[i], des2[j]) for j in range(len(kp2))])
    #     # sort the distances
    #     ind = np.argsort(li)
    #     plt.plot(li[ind])
    #     # print(li[ind[:10]])
    #     for j in range(10):
    #         if li[ind[j]] < 40:
    #             matches.append(cv2.DMatch(i, ind[j], 0))
    #             print(f"Matched {i} with {ind[j]}")

    # for i in range(len(kp1)):
    #     plot(i)

    img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(img)
    plt.show()
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
    img1, img2, cld1, cld2 = imgs[0], imgs[1], clds[0], clds[1]

    match_features(img1, img2, cld1, cld2)
