import numpy as np
from scipy.spatial.transform import Rotation

def transform_points(points, pose):
    return points.dot(Rotation.from_rotvec(pose[:3]).as_matrix().T) + pose[3:]

def plot_3d_landmark(verts, campose, intrinsics):
    lm_3d_trans = transform_points(verts, campose)

    # project to image plane
    lms_3d_trans_proj = intrinsics.dot(lm_3d_trans.T).T
    lms_projected = (
        lms_3d_trans_proj[:, :2] / np.tile(lms_3d_trans_proj[:, 2], (2, 1)).T
    )

    return lms_projected, lms_3d_trans_proj

def transform_vertices(img, pose, vertices, global_intrinsics=None):
        (h, w,_) = img.shape
        if global_intrinsics is None:
            global_intrinsics = np.array(
                [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
            )

        projected_lms = np.zeros_like(vertices)
        projected_lms[:, :2], lms_3d_trans_proj = plot_3d_landmark(
            vertices, pose, global_intrinsics
        )
        projected_lms[:, 2] = lms_3d_trans_proj[:, 2] * -1

        range_x = np.max(projected_lms[:, 0]) - np.min(projected_lms[:, 0])
        range_y = np.max(projected_lms[:, 1]) - np.min(projected_lms[:, 1])

        s = (h + w) / pose[5]
        projected_lms[:, 2] *= s
        projected_lms[:, 2] += (range_x + range_y) * 3

        return projected_lms
