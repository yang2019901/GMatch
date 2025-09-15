# A python version of RANSAC-based registration using NumPy.
# This code is a modified version of Open3D's RegistrationRANSACBasedOnCorrespondence.

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import logging


@dataclass
class RegistrationResult:
    fitness: float
    inlier_rmse: float
    transformation: np.ndarray
    correspondence_set: Optional[np.ndarray] = None  # (K_inlier, 2) 内点 correspondence

    def is_better_ransac_than(self, other: "RegistrationResult") -> bool:
        """first fitness then inlier_rmse"""
        if self.fitness != other.fitness:
            return self.fitness > other.fitness
        return self.inlier_rmse < other.inlier_rmse


def compute_transformation_point_to_point(
    source_points: np.ndarray, target_points: np.ndarray, correspondences: np.ndarray
) -> np.ndarray:
    """
    Kabsch method (i.e. SVD)
    correspondences: (n, 2), each row is [src_idx, tgt_idx]
    """
    src_pts = source_points[correspondences[:, 0]]  # (n, 3)
    tgt_pts = target_points[correspondences[:, 1]]  # (n, 3)

    src_centroid = np.mean(src_pts, axis=0)
    tgt_centroid = np.mean(tgt_pts, axis=0)

    src_centered = src_pts - src_centroid
    tgt_centered = tgt_pts - tgt_centroid

    W = tgt_centered.T @ src_centered
    U, _, Vt = np.linalg.svd(W)

    # ensure determinant is positive (aka, SO(3) matrix)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = tgt_centroid - R @ src_centroid

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def evaluate_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    correspondences: np.ndarray,
    transformation: np.ndarray,
    max_correspondence_distance: float,
) -> Tuple[float, float]:
    """
    calculate inlier count and RMSE for correspondences under the given transformation.
    return: (fitness, inlier_rmse)
    """
    src_indices = correspondences[:, 0]
    tgt_indices = correspondences[:, 1]

    src_pts = source_points[src_indices]  # (K, 3)
    tgt_pts = target_points[tgt_indices]  # (K, 3)

    src_homo = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    transformed_src = (transformation @ src_homo.T).T[:, :3]  # (K, 3)

    diff = transformed_src - tgt_pts
    distances_sq = np.sum(diff**2, axis=1)

    inlier_mask = distances_sq < (max_correspondence_distance**2)
    inlier_count = np.sum(inlier_mask)

    fitness = inlier_count / len(correspondences)
    inlier_rmse = np.sqrt(np.mean(distances_sq[inlier_mask])) if inlier_count > 0 else np.inf

    return fitness, inlier_rmse, inlier_mask


def registration_ransac_based_on_correspondence(
    source_points: np.ndarray,
    target_points: np.ndarray,
    correspondences: np.ndarray,
    max_correspondence_distance: float,
    estimation_method: Callable = compute_transformation_point_to_point,
    ransac_n: int = 3,
    checkers: Optional[list] = None,  # not implemented here
    max_iteration: int = 1000,
    confidence: float = 0.999,
) -> RegistrationResult:
    """
    Modified python version of open3d.pipelines.registration.registration_ransac_based_on_correspondence().
    what's different: this version traverses given correspondences to find inliers instead of transforming source_points and finding nearest neighbors with KD-Tree.
    """
    if ransac_n < 3 or len(correspondences) < ransac_n or max_correspondence_distance <= 0:
        return RegistrationResult(fitness=0.0, inlier_rmse=np.inf, transformation=np.eye(4))

    best_result = RegistrationResult(fitness=0.0, inlier_rmse=np.inf, transformation=np.eye(4))
    est_k_global = max_iteration
    total_validation = 0

    K = len(correspondences)

    # RANSAC main loop
    for itr in range(max_iteration):

        if itr >= est_k_global:
            break

        # Step 1: sample from correspondences
        sampled_indices = np.random.choice(K, size=ransac_n, replace=False)
        sampled_corres = correspondences[sampled_indices]

        # Step 2: estimate transformation with Kabsch method
        try:
            transformation = estimation_method(source_points, target_points, sampled_corres)
        except np.linalg.LinAlgError:
            continue

        # Step 3: checker (omitted for simplicity)
        # if checkers: ...
        #   for checker in checkers: ...

        # Step 4 & 5: evaluate fitness and rmse of the transformation
        fitness, inlier_rmse, inlier_mask = evaluate_correspondences(
            source_points, target_points, correspondences, transformation, max_correspondence_distance
        )

        current_result = RegistrationResult(
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            transformation=transformation,
            correspondence_set=correspondences[inlier_mask],
        )

        # Step 6: update best result
        if current_result.is_better_ransac_than(best_result):
            best_result = current_result

            # update est_k_global based on current fitness
            if fitness > 0:
                est_k_local_d = np.log(1.0 - confidence) / np.log(1.0 - fitness**ransac_n)
                est_k_local_d = max(1, est_k_local_d)
            else:
                est_k_local_d = est_k_global

            if est_k_local_d < est_k_global:
                est_k_global = int(np.ceil(est_k_local_d))

        total_validation += 1

    logging.debug(
        f"RANSAC exits after {total_validation} validations. "
        f"Best fitness: {best_result.fitness:.3f}, RMSE: {best_result.inlier_rmse:.3e}"
    )

    return best_result
