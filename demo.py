import numpy as np
from SimpleSim import TrackSimulator
import warp as wp
from time import time

wp.init()

num_robots = 16  # number of simulated robots (in parallel)
T = 10_000  # number of simulation timesteps to perform
device = "cuda"

shp = (128, 128)
np_heightmaps = [np.zeros(shp, dtype=np.float32) for _ in range(num_robots)]
np_kfs = [0.5 * np.ones(shp, dtype=np.float32) for _ in range(num_robots)]
res = [0.1 for _ in range(num_robots)]

init_poses = np.zeros((num_robots, 7))
init_poses[:, 2] = 1.2  # z coordinate
init_poses[:, :2] = np.clip(np.random.randn(num_robots, 2), -1, 1)  # z coordinate
init_poses[:, 6] = 1.0  # quaternion w

# track_vels = 0.5*np.ones((num_robots, T, 2)) + 0.05*np.random.randn(num_robots, 1, 2)
track_vels = 0.5*np.ones((num_robots, T, 2))
track_vels[0, :, 0] = 0.6
flipper_angles = np.zeros((num_robots, T, 4))
# flipper_angles[0, :, 0] = 0.5

simulator = TrackSimulator(np_heightmaps, np_kfs, res, T, use_renderer=True, device=device)

simulator.set_control(track_vels, flipper_angles)
simulator.set_init_poses(init_poses)

for i in range(10):
    start = time()
    body_q = simulator.simulate(render=False, use_graph=True)
    body_q_np = body_q.numpy()
    print('simulation took ', time() - start, ' s')
    # print('body_q: ', body_q_np)

