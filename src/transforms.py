import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def interpolate_poses(poses_times, poses, target_times):
    """
    Interpolates poses using slerp.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
    """
    assert len(poses_times) == len(poses)
    assert len(poses_times) > 0
    assert poses.shape[1] == 7  # [x, y, z, qx, qy, qz, qw]

    # repeat poses for time moments that are outside the range of poses_times
    if target_times[0] < poses_times[0]:
        poses_times = np.insert(poses_times, 0, target_times[0])
        poses = np.insert(poses, 0, poses[0], axis=0)
    if target_times[-1] > poses_times[-1]:
        poses_times = np.append(poses_times, target_times[-1])
        poses = np.append(poses, poses[-1].reshape(1, 7), axis=0)

    # Convert poses to quaternions
    quats = poses[:, 3:]
    rots = Rotation.from_quat(quats)

    # Create slerp object
    slerp = Slerp(poses_times, rots)

    # Interpolate quaternions
    interp_rots = slerp(target_times)

    # Convert interpolated quaternions to poses
    interp_poses = np.zeros((len(target_times), 7))
    interp_poses[:, 3:] = interp_rots.as_quat()

    # Interpolate positions
    for i in range(3):
        interp_poses[:, i] = np.interp(target_times, poses_times, poses[:, i])

    return interp_poses
