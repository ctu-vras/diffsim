import numpy as np
import matplotlib.pyplot as plt


def states_to_line(states):
    points = []
    for s in states:
        pos = s.body_q.numpy()[0][:3]
        points.append(pos)
    return points


def traj_to_line(traj):
    points = []
    for (t, body_q) in traj:
        pos = body_q[:3]
        points.append(pos)
    return points


def traj_to_posquat(traj):
    pos = []
    quat = []
    for (t, body_q) in traj:
        pos.append(body_q[:3])
        quat.append(body_q[3:])
    return pos, quat


def get_heightmap_vis_ids(shp):
    # create triangles from the points in the grid
    heightmap_vis_indices = []
    for i in range(shp[0] - 1):
        for j in range(shp[1] - 1):
            heightmap_vis_indices.append([i * shp[1] + j, (i + 1) * shp[1] + j, (i + 1) * shp[1] + j + 1])
            heightmap_vis_indices.append([i * shp[1] + j, (i + 1) * shp[1] + j + 1, i * shp[1] + j + 1])
    return heightmap_vis_indices


def generate_force_vis(points, forces, scale=0.001):
    force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
    line_pts = np.zeros((len(points)*2, 3))
    line_pts[::2] = [0, 0, -10]
    line_pts[1::2] = [0, 0, -11]
    indices = np.arange(len(points)*2)
    for i in range(len(points)):
        if force_norms[i] > 1e-3:
            line_pts[2*i] = points[i]
            line_pts[2*i+1] = points[i] + forces[i]*scale
    return line_pts, indices


def plot_maps(gt, opt, valid):
    plt.figure(figsize=(15, 5))
    diff = gt - opt
    gt_mean = np.mean(gt)
    gt = gt - gt_mean
    opt = opt - gt_mean
    diff[~valid] = 0
    gt[~valid] = 0
    opt[~valid] = 0
    rng = np.max(np.abs(diff))
    plt.subplot(1,3,1)
    plt.imshow(gt, vmin=-rng, vmax=rng, cmap='seismic')
    plt.subplot(1,3,2)
    plt.imshow(opt, vmin=-rng, vmax=rng, cmap='seismic')
    plt.subplot(1,3,3)
    plt.imshow(diff, vmin=-rng, vmax=rng, cmap='seismic')
    plt.colorbar()
    plt.show()