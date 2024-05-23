import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

def control_to_numpy(controls, t0, T, dt):
    '''We need control sequence of length T, but have much fewer control samples.
    We need to reassign this sparse control to simulation dense control.'''
    control_np = np.zeros((T, 2))
    control_idx = 0
    for sim_idx in range(T):
        sim_t = sim_idx * dt
        next_control_t = controls[control_idx + 1][0] - t0
        while sim_t >= next_control_t and (control_idx + 2) < len(controls):  # if sim_t ahead of next_control_idx, advance the control_idx
            control_idx += 1
            next_control_t = controls[control_idx + 1][0] - t0
        control_np[sim_idx] = controls[control_idx][1]
    return control_np

def flipper_angles_to_numpy(flipper_angles, t0, T, dt):
    '''We need flipper angle sequence of length T, but have much fewer flipper angle samples.
    We need to reassign this sparse flipper angle to simulation dense flipper angle.'''
    flipper_angles_np = np.zeros((T, 4))
    flipper_angle_idx = 0
    for sim_idx in range(T):
        sim_t = sim_idx * dt
        next_flipper_angle_t = flipper_angles[flipper_angle_idx + 1][0] - t0
        while sim_t >= next_flipper_angle_t and (flipper_angle_idx + 2) < len(flipper_angles):  # if sim_t ahead of next_flipper_angle_idx, advance the flipper_angle_idx
            flipper_angle_idx += 1
            next_flipper_angle_t = flipper_angles[flipper_angle_idx + 1][0] - t0
        flipper_angles_np[sim_idx] = flipper_angles[flipper_angle_idx][1]
    return flipper_angles_np

def parse_csv_traj(path, origin=(0,0,0)):
    '''open the poses csv in trajectory path and extract a list consisting of timestamps and poses.
    The poses are moved to begin at origin. The timestamps are relative with respect to first pose.'''
    poses = list(csv.reader(open(path + '/poses.csv', 'r')))
    traj = []
    t0 = float(poses[0][0])  # initial time
    pos0 = [float(n) for n in poses[0][1:4]]  # initial position
    for time_pose in poses:
        t = float(time_pose[0]) - t0  # relative timestamp
        body_q = np.array([float(n) for n in time_pose[1:]])
        body_q[:3] = body_q[:3] + origin - pos0  # shift the origin
        traj.append([t, body_q])  # relative timestamp and 6DOF pose as [x, y, z, qx, qy, qz, qw]
    return traj, t0, pos0

def parse_csv_control(path):
    '''open control csv in trajectory path and extract a list of timestamps and controls. The timestamps are absolute.'''
    controls = list(csv.reader(open(path, 'r')))
    control = []
    for time_control in controls:
        t = float(time_control[0])
        u = [float(n) for n in time_control[1:]]
        control.append([t, u])  # 2DOF control as [left, right]
    return control

def combine_transforms(t1, t2):
    pos1, quat1 = t1[:3], t1[3:]
    pos2, quat2 = t2[:3], t2[3:]
    pos = pos1 + R.from_quat(quat1).apply(pos2)
    quat = (R.from_quat(quat1) * R.from_quat(quat2)).as_quat()
    return np.concatenate([pos, quat])