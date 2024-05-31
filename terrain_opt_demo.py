import torch
import warp as wp
import numpy as np
import os
from PIL import Image
from utils import sample_augmentation, img_transform, normalize_img, load_calib
from DiffSim import DiffSim
from TerrainEncoder import compile_model
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
wp.init()  # init warp!


def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def get_cam_data(img_paths, cameras, calib_path, terrain_encoder_cfg, device='cuda'):
    imgs = []
    rots = []
    trans = []
    post_rots = []
    post_trans = []
    intrins = []
    calib = load_calib(calib_path)
    for cam, img_path in zip(cameras, img_paths):
        img = Image.open(img_path)
        K = calib[cam]['camera_matrix']['data']
        K = np.asarray(K, dtype=np.float32).reshape((3, 3))
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = sample_augmentation(terrain_encoder_cfg)
        img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                   resize=resize,
                                                   resize_dims=resize_dims,
                                                   crop=crop,
                                                   flip=flip,
                                                   rotate=rotate)
        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        # rgb and intrinsics
        img = normalize_img(img)
        K = torch.as_tensor(K)
        # extrinsics
        T_robot_cam = calib['transformations'][f'T_base_link__{cam}']['data']
        T_robot_cam = np.asarray(T_robot_cam, dtype=np.float32).reshape((4, 4))
        rot = torch.as_tensor(T_robot_cam[:3, :3])
        tran = torch.as_tensor(T_robot_cam[:3, 3])
        imgs.append(img)
        rots.append(rot)
        trans.append(tran)
        intrins.append(K)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
    cam_data = [torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
    cam_data = [torch.as_tensor(i[None], dtype=torch.float32, device=device) for i in cam_data]

    return cam_data


def init_dphys(torch_hms, res, use_renderer=False, device='cuda'):
    num_robots = len(torch_hms)
    # instantiate dphys
    dphys = DiffSim(torch_hms, res, use_renderer=use_renderer, device=device)
    # set the simulated time horizon
    dphys.set_T(5000, T_s=300)

    # define target poses with timesteps for each robot
    num_poses = 30
    poses0 = np.zeros((num_poses, 7))
    poses0[:, 6] = 1  # quaternion w
    poses0[:, 0] = np.arange(num_poses) / num_poses * 1  # x coordinate
    poses0[:, 2] = 1.0
    timesteps0 = (dphys.T * np.arange(num_poses) / num_poses).astype(int)
    poses = [poses0[:] for _ in range(num_robots)]
    timesteps = [timesteps0[:] for _ in range(num_robots)]

    # define control input for each robot
    controls0 = 0.2 * np.ones((dphys.T, num_robots, 2))
    flipper_angles0 = np.zeros((dphys.T, num_robots, 4))
    controls = [controls0[:] for _ in range(num_robots)]
    flipper_angles = [flipper_angles0[:] for _ in range(num_robots)]

    # print('setting ground truth trajectories')
    dphys.set_target_poses(timesteps, poses)

    # print('setting controls')
    dphys.set_control(controls, flipper_angles)

    if use_renderer:
        dphys.render_heightmaps()
        dphys.render_traj(poses0[:, :3])

    return dphys


def main():
    num_robots = 1
    device = "cuda"
    use_cuda_graph = True
    use_renderer = True

    # get camera data
    terrain_encoder_cfg = read_yaml('config/lss_cfg_tradr.yaml')
    img_paths = sorted([os.path.join('data_sample/tradr/images/', f) for f in os.listdir('data_sample/tradr/images/') if f.endswith('.png')])
    cameras = sorted(['camera_front', 'camera_left', 'camera_right', 'camera_rear_left', 'camera_rear_right'])
    calib_path = 'data_sample/tradr/calibration/'
    cam_data = get_cam_data(img_paths, cameras, calib_path, terrain_encoder_cfg, device=device)
    print('cam_data', [i.shape for i in cam_data])

    # instantiate terrain encoder
    terrain_encoder = compile_model(terrain_encoder_cfg['grid_conf'], terrain_encoder_cfg['data_aug_conf'])
    terrain_encoder.to(device)
    terrain_encoder.train()

    # initialize heightmaps
    xbound = terrain_encoder_cfg['grid_conf']['xbound']
    ybound = terrain_encoder_cfg['grid_conf']['ybound']
    grid_res = xbound[2]
    shp = (int((xbound[1] - xbound[0]) / grid_res), int((ybound[1] - ybound[0]) / grid_res))
    torch_hms = [torch.zeros(shp, dtype=torch.float32, device=device, requires_grad=True) for _ in range(num_robots)]
    res = [grid_res for _ in range(num_robots)]  # heightmap resolutions

    # initialize diffsim
    dphys = init_dphys(torch_hms, res, use_renderer=use_renderer, device=device)

    sgd_iters = 2000
    optimizer = torch.optim.Adam(terrain_encoder.parameters(), lr=1e-6, weight_decay=1e-7)

    for i in range(sgd_iters):
        dphys.init_shoot_states()  # load initial states for the shooter

        # predict heightmaps
        torch_hms = terrain_encoder(*cam_data)
        # set new heightmaps
        dphys.update_heightmaps(torch_hms)
        # forward pass
        body_q, loss = dphys.simulate_and_backward(use_graph=use_cuda_graph)
        print('loss: ', loss.numpy())

        optimizer.step()
        optimizer.zero_grad(set_to_none=False)  # necessary, since tape.zero() does not reach the torch tensor for some reason

        if use_renderer and i % 20 == 0:
            print('iter', i)
            dphys.save_shoot_init_vels()  # save states for shooter
            dphys.simulate_single()  # simulate a single long trajectory for testing
            dphys.render_states('current', color=(1.0, 0.0, 0.0))
            dphys.render_simulation(pause=False)


if __name__ == '__main__':
    main()
