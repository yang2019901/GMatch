"""Implements GMatch to match keypoints extracted by SIFT detector."""

import numpy as np
import open3d as o3d
import cv2
import time, logging
import sys, os

from . import gmatch_cpp, util


logger = logging.getLogger(__file__)
logger.setLevel(logging.WARN)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[%(levelname)5s] [%(funcName)s] %(message)s"))
logger.addHandler(h)
logger.propagate = False

# cache for keypoints and features of imgs_src
CACHE = {}


""" SIFT settings """
detector: cv2.SIFT = cv2.SIFT_create()
detector.setContrastThreshold(0.03)
thresh_feat = 0.65  # threshold for feature distance, used to judge the similarity of two feature vectors
feat_mat = lambda feat1, feat2: sift_mat(feat1, feat2, rootsift=True)  # feature distance matrix


""" GMatch settings """
T = 24  # T pairs with the highest feature similarity will be used to Branch-and-Bound
L = 24  # max search length
# threshold for geometric cost, applied to 3d distance error ratio when attempting to add a pair to matches.
thresh_geom_ratio = 0.1
# threshold for geometric cost, accounts for point cloud noise
thresh_geom_abs = 0.01
# threshold for flipover judgement
thresh_flip = 0.8


def sift_mat(feat1, feat2, rootsift):
    """Compute feature distance matrix `Mf` for SIFT, whose metric is Euclidean distance.

    Note: RootSIFT is used here.

    Args:
        feat1: (n1, 128), integer stored in float32
        feat2: (n2, 128), integer stored in float32
    Returns:
        Mf: (n1, n2), float32
    """
    feat1_ = feat1 / np.sum(feat1, axis=-1, keepdims=True)
    feat2_ = feat2 / np.sum(feat2, axis=-1, keepdims=True)
    if rootsift:
        feat1_ = np.sqrt(feat1_)
        feat2_ = np.sqrt(feat2_)
    Mf = np.linalg.norm(feat1_[:, np.newaxis, :] - feat2_[np.newaxis, :, :], axis=-1)
    return Mf


def SIFT_detect(img, mask):
    """Extract keypoints and SIFT features from the image.

    Args:
        img: (H, W, 3), uint8
        mask: (H, W), uint8
    Returns:
        uv: (n, 2), int32
        feat: (n, 128), float32
    """
    global detector

    _dim = 128 # SIFT feature dimension
    _type = np.float32 # SIFT feature type

    # extract the keypoints and features with sift descriptor
    kp, feat = detector.detectAndCompute(img, mask)

    if len(kp) == 0:
        uv = np.zeros((0, 2), dtype=np.int32)
        feat = np.zeros((0, _dim), dtype=_type)
    else:
        uv = np.array([k.pt for k in kp], dtype=np.int32).reshape(-1, 2) # shepe: (n2, 2), int32
        # remove duplicated keypoints
        uv, ind = np.unique(uv, axis=0, return_index=True)
        feat = feat[ind]
    return uv, feat


def search(pts1, pts2, Mf12):
    """Branch-and-Bound search with geometric constraints (distance matrix and flip-over removal)

    Args:
        pts1: (n1, 3), float32
        pts2: (n2, 3), float32
        Mf12: (n1, n2), feature distance matrix; float32 for SIFT
    Returns:
        (matches, cost): matches is (d, 2), int32; cost is float32 in [0, 1] where 1 means no matches found.
    """
    matches, cost = gmatch_cpp.gmatch_search_bnb(
        pts1, pts2, Mf12, thresh_feat, L, T, thresh_geom_ratio, thresh_geom_abs, thresh_flip
    )
    return np.reshape(matches, (-1, 2)).astype(np.int32), cost


def Match(match_data: util.MatchData, cache_id=None, debug=-1):
    """Match keypoints and features between destination and each source images.

    Extract data from `match_data`, then match each of source images and destination image in match_data and store the result in it;

    Args:
        match_data: util.MatchData, see util.py for details.
        cache_id: any immutable, optional. if provided, the keypoints and features for the source image will be cached with `cache_id`.
        debug: -1, 0, 1, 2, bigger value means more debug info and -1 means none
    Returns:
        None, the result is stored in `match_data`.
    """
    global CACHE
    assert len(match_data.imgs_src) > 0, "imgs_src is empty"

    EMPTY_MATCHES = np.zeros((0, 2), dtype=np.int32)

    # imgs_src: (N, H, W, 3); clds_src: (N, H, W, 3); masks_src: (N, H, W); can be list
    imgs_src, clds_src, masks_src = (
        match_data.imgs_src,
        match_data.clds_src,
        match_data.masks_src,
    )
    # img_dst: (H, W, 3); cld_dst: (H, W, 3); mask_dst: (H, W)
    img_dst, cld_dst, mask_dst = (
        match_data.img_dst,
        match_data.cld_dst,
        match_data.mask_dst,
    )

    # extract destination keypoints and features
    uv_dst, feat_dst = SIFT_detect(img_dst, mask_dst)

    # filter out points with invalid depth
    mask = cld_dst[uv_dst[:, 1], uv_dst[:, 0], 2] > 1e-3
    uv_dst = uv_dst[mask]
    feat_dst = feat_dst[mask]
    pts_dst = cld_dst[uv_dst[:, 1], uv_dst[:, 0]]

    matches_list = []
    uvs_src = []
    for i, (img_src, cld_src, mask_src) in enumerate(zip(imgs_src, clds_src, masks_src)):
        if (cache_id, i) in CACHE:
            uv_src, feat_src = CACHE[(cache_id, i)]
        else:
            # extract source keypoints and features
            uv_src, feat_src = SIFT_detect(img_src, mask_src)

            # filter out points with invalid depth
            mask = cld_src[uv_src[:, 1], uv_src[:, 0], 2] > 1e-3
            uv_src = uv_src[mask]
            feat_src = feat_src[mask]

            if cache_id is not None:
                CACHE[(cache_id, i)] = (uv_src, feat_src)

        pts_src = cld_src[uv_src[:, 1], uv_src[:, 0]]
        uvs_src.append(uv_src)

        # Feature Distance Matrix (for visual similarity)
        Mf12 = feat_mat(feat_src, feat_dst)

        if debug >= 2:
            util.plot_keypoints(img_src, img_dst, uv_src, uv_dst, Mf12, thresh_feat)

        matches, cost = search(pts_src, pts_dst, Mf12)
        matches_list.append((matches, cost))

        # visualization
        if debug >= 1:
            if len(matches) < 3:
                logger.info(f"\timgs_src[{i}]: matches NOT found.")
            else:
                logger.info(f"\timgs_src[{i}]: matches found. depth {len(matches)}, cost {cost:.3f}")
                util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])

        if len(matches) == L:
            break

    # take max depth matches as the best
    match_data.matches_list, match_data.cost_list = zip(*matches_list)
    match_data.uvs_src = uvs_src
    match_data.uv_dst = uv_dst
    match_data.idx_best = np.argmax([len(matches) for matches, _ in matches_list])

    # visualization
    if debug >= 0:
        if len(match_data.matches_list[match_data.idx_best]) > 0:
            img_src = imgs_src[match_data.idx_best]
            uv_src = uvs_src[match_data.idx_best]
            matches = matches_list[match_data.idx_best][0]
            util.plot_matches(img_src, img_dst, uv_src[matches[:, 0]], uv_dst[matches[:, 1]])
    return
