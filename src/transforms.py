import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def interpolate_poses(poses_times, poses, interp_times):
    """
    Interpolates poses using slerp.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
    """
    assert len(poses_times) == len(poses)
    assert len(poses_times) > 1
    assert poses.shape[1] == 7  # [x, y, z, qx, qy, qz, qw]

    # Convert poses to quaternions
    quats = poses[:, 3:]
    rots = Rotation.from_quat(quats)

    # Create slerp object
    slerp = Slerp(poses_times, rots)

    # Interpolate quaternions
    interp_rots = slerp(interp_times)

    # Convert interpolated quaternions to poses
    interp_poses = np.zeros((len(interp_times), 7))
    interp_poses[:, 3:] = interp_rots.as_quat()

    # Interpolate positions
    for i in range(3):
        interp_poses[:, i] = np.interp(interp_times, poses_times, poses[:, i])

    return interp_poses
