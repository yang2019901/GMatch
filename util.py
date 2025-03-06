import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d


def pose2mat(pose):
    """pose: [[x, y, z], [qx, qy, qz, qw]], M: 4x4"""
    M = np.eye(4)
    M[:3, 3] = pose[0]
    M[:3, :3] = Rotation.from_quat(pose[1]).as_matrix()
    return M


def mat2pose(M):
    """pose: [[x, y, z], [qx, qy, qz, qw]], M: 4x4"""
    return [M[:3, 3], Rotation.from_matrix(M[:3, :3]).as_quat()]


def transform(pts, pose):
    """pts: (..., 3), pose: [[x, y, z], [qx, qy, qz, qw]], return: (..., 3)"""
    _pts = np.array(pts)
    M = pose2mat(pose)
    return _pts @ M[:3, :3].T + M[:3, 3]


def depth2cld(depth, intrisic):
    intrin = np.array(intrisic).reshape(3, 3)
    z = depth
    u, v = np.meshgrid(range(z.shape[1]), range(z.shape[0]))
    uv = np.stack((u, v, np.ones_like(u)), axis=-1)
    pts = np.linalg.inv(intrin) @ uv.reshape(-1, 3).T * z.reshape(-1)
    return pts.T.reshape(z.shape[0], z.shape[1], 3)


def vis_cld(clds, colors=None, poses=None):
    if poses is not None:
        _clds = np.array([transform(cld, pose) for cld, pose in zip(clds, poses)])
    else:
        _clds = np.array(clds)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_clds.reshape(-1, 3))
    if colors is not None:
        _colors = np.array(colors).reshape(-1, 3) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(_colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, axis])
