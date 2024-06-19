import copy
import os
import numpy as np
import torch
import torchvision
from scipy.interpolate import griddata
from skimage.draw import polygon
from torch.utils.data import Dataset
from ..utils import img_transform, normalize_img, ego_to_cam, get_only_in_img_mask, sample_augmentation
from ..utils import position, normalize, load_calib, read_yaml
from ..config import DPhysConfig
from .coco import COCO_CATEGORIES
import cv2
import albumentations as A
from PIL import Image
import open3d as o3d


__all__ = [
    'data_dir',
    'RobinGas',
    'RobinGasPoints',
    'robingas_seq_paths',
]


data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

robingas_seq_paths = {
    'husky': [
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-10-27-15-33-57'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-09-27-10-33-15'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-09-27-15-01-44'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-09-23-12-38-31'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-06-30-15-58-37'),
    ],
    'marv': [
        os.path.join(data_dir, 'RobinGas/marv/ugv_2022-08-12-16-37-03'),
        os.path.join(data_dir, 'RobinGas/marv/ugv_2022-08-12-15-18-34'),
    ],
    'tradr': [
        os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-10-20-14-30-57'),
        os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-10-20-14-05-42'),
        os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-10-20-13-58-22'),
        # os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-06-30-11-30-57'),
    ],
    'husky_oru': [
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-02-33_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-09-06_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-24-37_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-37-14_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-44-56_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-54-42_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-02-07-10-47-13_0'),  # no radar
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-04-27-15-02-12_0'),
        # os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-05-01-15-48-29_0'),  # localization must be fixed
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-05-24-13-21-28_0'),  # no radar
    ],
}


def transform_cloud(cloud, Tr):
    assert isinstance(cloud, np.ndarray) or isinstance(cloud, torch.Tensor), type(cloud)
    assert isinstance(Tr, np.ndarray) or isinstance(Tr, torch.Tensor), type(Tr)
    if isinstance(cloud, np.ndarray) and cloud.dtype.names is not None:
        points = position(cloud)
        points = transform_cloud(points, Tr)
        cloud = cloud.copy()
        cloud['x'] = points[:, 0]
        cloud['y'] = points[:, 1]
        cloud['z'] = points[:, 2]
        return cloud
    assert cloud.ndim == 2
    assert cloud.shape[1] == 3  # (N, 3)
    cloud_tr = Tr[:3, :3] @ cloud.T + Tr[:3, 3:]
    return cloud_tr.T


def rot2rpy(R):
    assert isinstance(R, torch.Tensor) or isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    if isinstance(R, np.ndarray):
        R = torch.as_tensor(R)
    roll = torch.atan2(R[2, 1], R[2, 2])
    pitch = torch.atan2(-R[2, 0], torch.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = torch.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

def rpy2rot(roll, pitch, yaw):
    roll = torch.as_tensor(roll)
    pitch = torch.as_tensor(pitch)
    yaw = torch.as_tensor(yaw)
    RX = torch.tensor([[1, 0, 0],
                       [0, torch.cos(roll), -torch.sin(roll)],
                       [0, torch.sin(roll), torch.cos(roll)]], dtype=torch.float32)

    RY = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                       [0, 1, 0],
                       [-torch.sin(pitch), 0, torch.cos(pitch)]], dtype=torch.float32)

    RZ = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                       [torch.sin(yaw), torch.cos(yaw), 0],
                       [0, 0, 1]], dtype=torch.float32)
    return RZ @ RY @ RX


def estimate_heightmap(points, d_min=1., d_max=6.4, grid_res=0.1,
                       h_max_above_ground=1., robot_clearance=0.,
                       hm_interp_method='nearest',
                       fill_value=0., robot_radius=None, return_filtered_points=False,
                       map_pose=np.eye(4)):
    assert points.ndim == 2
    assert points.shape[1] >= 3  # (N x 3)
    assert len(points) > 0
    assert isinstance(d_min, (float, int)) and d_min >= 0.
    assert isinstance(d_max, (float, int)) and d_max >= 0.
    assert isinstance(grid_res, (float, int)) and grid_res > 0.
    assert isinstance(h_max_above_ground, (float, int)) and h_max_above_ground >= 0.
    assert hm_interp_method in ['linear', 'nearest', 'cubic', None]
    assert fill_value is None or isinstance(fill_value, (float, int))
    assert robot_radius is None or isinstance(robot_radius, (float, int)) and robot_radius > 0.
    assert isinstance(return_filtered_points, bool)
    assert map_pose.shape == (4, 4)

    # remove invalid points
    mask_valid = np.isfinite(points).all(axis=1)
    points = points[mask_valid]

    # gravity aligned points
    roll, pitch, yaw = rot2rpy(map_pose[:3, :3])
    R = rpy2rot(roll, pitch, 0.).cpu().numpy()
    points_grav = points @ R.T

    # filter points above ground
    mask_h = points_grav[:, 2] + robot_clearance <= h_max_above_ground

    # filter point cloud in a square
    mask_sq = np.logical_and(np.abs(points[:, 0]) <= d_max, np.abs(points[:, 1]) <= d_max)

    # points around robot
    if robot_radius is not None:
        mask_cyl = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= robot_radius / 2.
    else:
        mask_cyl = np.zeros(len(points), dtype=bool)

    # combine and apply masks
    mask = np.logical_and(mask_h, mask_sq)
    mask = np.logical_and(mask, ~mask_cyl)
    points = points[mask]
    if len(points) == 0:
        if return_filtered_points:
            return None, None
        return None

    # create a grid
    n = int(2 * d_max / grid_res)
    xi = np.linspace(-d_max, d_max, n)
    yi = np.linspace(-d_max, d_max, n)
    x_grid, y_grid = np.meshgrid(xi, yi)

    if hm_interp_method is None:
        # estimate heightmap
        z_grid = np.full(x_grid.shape, fill_value=fill_value)
        mask_meas = np.zeros_like(z_grid)
        for i in range(len(points)):
            xp, yp, zp = points[i]
            # find the closest grid point
            idx_x = np.argmin(np.abs(xi - xp))
            idx_y = np.argmin(np.abs(yi - yp))
            # update heightmap
            if z_grid[idx_y, idx_x] == fill_value or zp > z_grid[idx_y, idx_x]:
                z_grid[idx_y, idx_x] = zp
                mask_meas[idx_y, idx_x] = 1.
        mask_meas = mask_meas.astype(np.float32)
    else:
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        z_grid = griddata((X, Y), Z, (xi[None, :], yi[:, None]),
                          method=hm_interp_method, fill_value=fill_value)
        mask_meas = np.full(z_grid.shape, 1., dtype=np.float32)

    z_grid = z_grid.T
    mask_meas = mask_meas.T
    heightmap = {'x': np.asarray(x_grid, dtype=np.float32),
                 'y': np.asarray(y_grid, dtype=np.float32),
                 'z': np.asarray(z_grid, dtype=np.float32),
                 'mask': mask_meas}

    if return_filtered_points:
        return heightmap, points

    return heightmap


def filter_range(cloud, min, max, log=False, only_mask=False):
    """Keep points within range interval."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    assert isinstance(min, (float, int)), min
    assert isinstance(max, (float, int)), max
    assert min <= max, (min, max)
    min = float(min)
    max = float(max)
    if min <= 0.0 and max == np.inf:
        return cloud
    if cloud.dtype.names:
        cloud = cloud.ravel()
    x = position(cloud)
    r = np.linalg.norm(x, axis=1)
    mask = (min <= r) & (r <= max)

    if log:
        print('%.3f = %i / %i points kept (range min %s, max %s).'
              % (mask.sum() / len(cloud), mask.sum(), len(cloud), min, max))

    if only_mask:
        return mask

    filtered = cloud[mask]
    return filtered

class RobinGasBase(Dataset):
    """
    Class to wrap traversability data generated using lidar odometry.

    The data is stored in the following structure:
    - <path>
        - clouds
            - <id>.npz
            - ...
        - images
            - <id>_<camera_name>.png
            - ...
        - trajectories
            - <id>.csv
            - ...
        - calibration
            - cameras
                - <camera_name>.yaml
                - ...
            - transformations.yaml
        - terrain
            - <id>.npy
            - ...
        - poses
            - lidar_poses.csv
            - ...

    A sample of the dataset contains:
    - point cloud (N x 3), where N is the number of points
    - height map (H x W)
    - trajectory (T x 4 x 4), where the horizon T is the number of poses
    """

    def __init__(self, path, dphys_cfg=DPhysConfig()):
        super(Dataset, self).__init__()
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.cloud_path = os.path.join(path, 'clouds')
        # assert os.path.exists(self.cloud_path)
        self.radar_cloud_path = os.path.join(path, 'radar_clouds')
        # assert os.path.exists(self.radar_cloud_path)
        self.traj_path = os.path.join(path, 'trajectories')
        # global pose of the robot (initial trajectory pose on a map) path (from SLAM)
        self.poses_path = os.path.join(path, 'poses', 'lidar_poses.csv')
        # assert os.path.exists(self.traj_path)
        self.calib_path = os.path.join(path, 'calibration')
        # assert os.path.exists(self.calib_path)
        self.dphys_cfg = dphys_cfg
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.ts, self.poses = self.get_poses(return_stamps=True)
        # get camera names
        self.cameras = self.get_camera_names()

    def get_ids(self):
        ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        ids = np.sort(ids)
        return ids

    @staticmethod
    def pose2mat(pose):
        T = np.eye(4)
        T[:3, :4] = pose.reshape((3, 4))
        return T

    def get_poses(self, return_stamps=False):
        if not os.path.exists(self.poses_path):
            print(f'Poses file {self.poses_path} does not exist')
            return None
        data = np.loadtxt(self.poses_path, delimiter=',', skiprows=1)
        stamps, Ts = data[:, 0], data[:, 1:13]
        lidar_poses = np.asarray([self.pose2mat(pose) for pose in Ts], dtype=np.float32)
        # poses of the robot in the map frame
        Tr_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=np.float32).reshape((4, 4))
        Tr_lidar_robot = np.linalg.inv(Tr_robot_lidar)
        poses = lidar_poses @ Tr_lidar_robot
        if return_stamps:
            return stamps, poses
        return poses

    def get_pose(self, i):
        return self.poses[i]

    def get_camera_names(self):
        cams_yaml = os.listdir(os.path.join(self.path, 'calibration/cameras'))
        cams = [cam.replace('.yaml', '') for cam in cams_yaml]
        if 'camera_up' in cams:
            cams.remove('camera_up')
        return sorted(cams)

    def get_traj(self, i, n_frames=100):
        # n_frames equals to the number of future poses (trajectory length)
        ind = self.ids[i]
        Tr_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=np.float32).reshape((4, 4))
        # load data from csv file
        csv_path = os.path.join(self.traj_path, '%s.csv' % ind)
        if os.path.exists(csv_path):
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            stamps, poses = data[:, 0], data[:, 1:13]
            poses = np.asarray([self.pose2mat(pose) for pose in poses])
            # transform to robot frame
            poses = Tr_robot_lidar @ poses
        else:
            # get trajectory as sequence of `n_frames` future poses
            all_poses = self.get_poses(return_stamps=False)
            all_ids = list(self.get_ids())
            il = all_ids.index(ind)
            if n_frames is None:
                poses = copy.copy(all_poses)
                stamps = np.asarray(copy.copy(self.ts), dtype=np.float32)
            else:
                ir = np.clip(il + n_frames, 0, len(all_ids))
                poses = all_poses[il:ir]
                stamps = np.asarray(copy.copy(self.ts[il:ir]))
            assert len(poses) > 0, f'No poses found for trajectory {ind}'
            poses = np.linalg.inv(poses[0]) @ poses

        traj = {
            'stamps': stamps, 'poses': poses,
        }

        return traj

    def get_states_traj(self, i, start_from_zero=False):
        traj = self.get_traj(i)
        poses = traj['poses']

        if start_from_zero:
            # transform poses to the same coordinate frame as the height map
            Tr = np.linalg.inv(poses[0])
            poses = np.asarray([np.matmul(Tr, p) for p in poses])
            poses[:, 2, 3] -= self.calib['clearance']
            # count time from 0
            tstamps = traj['stamps']
            tstamps = tstamps - tstamps[0]

        poses = np.asarray(poses, dtype=np.float32)
        tstamps = np.asarray(tstamps, dtype=np.float32)

        xyz = torch.as_tensor(poses[:, :3, 3])
        rot = torch.as_tensor(poses[:, :3, :3])

        n_states = len(xyz)
        tt = torch.tensor(tstamps)[None].T

        dps = torch.diff(xyz, dim=0)
        dt = torch.diff(tt, dim=0)
        theta = torch.atan2(dps[:, 1], dps[:, 0]).view(-1, 1)
        theta = torch.cat([theta[:1], theta], dim=0)

        vel = torch.zeros_like(xyz)
        vel[:-1] = dps / dt
        omega = torch.zeros_like(xyz)
        omega[:-1, 2:3] = torch.diff(theta, dim=0) / dt  # + torch.diff(angles, dim=0)[:, 2:3] / dt

        forces = torch.zeros((n_states, 3, 10))
        states = (xyz.view(n_states, 3, 1),
                  rot.view(n_states, 3, 3),
                  vel.view(n_states, 3, 1),
                  omega.view(n_states, 3, 1),
                  forces.view(n_states, 3, 10))
        return states

    def get_raw_cloud(self, i):
        ind = self.ids[i]
        cloud_path = os.path.join(self.cloud_path, '%s.npz' % ind)
        assert os.path.exists(cloud_path), f'Cloud path {cloud_path} does not exist'
        cloud = np.load(cloud_path)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud

    def get_lidar_cloud(self, i):
        cloud = self.get_raw_cloud(i)
        # remove nans from structured array with fields x, y, z
        cloud = cloud[~np.isnan(cloud['x'])]
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud
    
    def get_raw_radar_cloud(self, i):
        ind = self.ids[i]
        cloud_path = os.path.join(self.radar_cloud_path, '%s.npz' % ind)
        assert os.path.exists(cloud_path), f'Cloud path {cloud_path} does not exist'
        cloud = np.load(cloud_path)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud
    
    def get_radar_cloud(self, i):
        cloud = self.get_raw_radar_cloud(i)
        # remove nans from structured array with fields x, y, z
        cloud = cloud[~np.isnan(cloud['x'])]
        # close by points contain noise
        cloud = filter_range(cloud, 3.0, np.inf)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__hugin_radar']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_cloud(self, i, points_source='lidar'):
        assert points_source in ['lidar', 'radar', 'lidar_radar']
        if points_source == 'lidar':
            return self.get_lidar_cloud(i)
        elif points_source == 'radar':
            return self.get_radar_cloud(i)
        else:
            lidar_points = self.get_lidar_cloud(i)
            radar_points = self.get_radar_cloud(i)
            cloud = np.concatenate((lidar_points[['x', 'y', 'z']], radar_points[['x', 'y', 'z']]))
            return cloud

    def get_traj_dphyics_terrain(self, i):
        ind = self.ids[i]
        p = os.path.join(self.path, 'terrain', 'traj', 'dphysics', '%s.npy' % ind)
        terrain = np.load(p)['height']
        return terrain

    def get_footprint_traj_points(self, i, robot_size=(0.7, 1.0)):
        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, self.dphys_cfg.grid_res)
        y = np.arange(-width / 2, width / 2, self.dphys_cfg.grid_res)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        footprint0 = np.stack([x, y, z], axis=-1).reshape((-1, 3))

        Tr_base_link__base_footprint = np.asarray(self.calib['transformations']['T_base_link__base_footprint']['data'],
                                                  dtype=float).reshape((4, 4))
        traj = self.get_traj(i)
        poses = traj['poses']
        poses_footprint = poses @ Tr_base_link__base_footprint

        trajectory_points = []
        for Tr in poses_footprint:
            footprint = transform_cloud(footprint0, Tr)
            trajectory_points.append(footprint)
        trajectory_points = np.concatenate(trajectory_points, axis=0)
        return trajectory_points

    def estimate_heightmap(self, points, **kwargs):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.dphys_cfg.d_min, d_max=self.dphys_cfg.d_max,
                                    grid_res=self.dphys_cfg.grid_res,
                                    h_max_above_ground=self.dphys_cfg.h_max_above_ground,
                                    robot_clearance=self.calib['clearance'],
                                    hm_interp_method=self.dphys_cfg.hm_interp_method, **kwargs)
        return height

    def get_sample(self, i):
        cloud = self.get_cloud(i)
        traj = self.get_traj(i)
        height = self.estimate_heightmap(position(cloud), fill_value=0.)
        return cloud, traj, height

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample

        ds = copy.deepcopy(self)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
            ds.poses = [self.poses[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
            ds.poses = self.poses[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ids)


class RobinGas(RobinGasBase):
    """
    A dataset for traversability estimation from camera and lidar data.

    A sample of the dataset contains:
    - image (3 x H x W)
    - rotation matrix (3 x 3)
    - translation vector (3)
    - intrinsic matrix (3 x 3)
    - post rotation matrix (3 x 3)
    - post translation vector (3)
    - lidar height map (2 x H x W)
    - trajectory height map (2 x H x W)
    - map pose (4 x 4)
    """

    def __init__(self,
                 path,
                 lss_cfg,
                 dphys_cfg=DPhysConfig(),
                 is_train=False,
                 only_front_cam=False):
        super(RobinGas, self).__init__(path, dphys_cfg)
        self.is_train = is_train
        self.only_front_cam = only_front_cam
        self.cameras = self.cameras[:1] if only_front_cam else self.cameras

        self.lss_cfg = lss_cfg
        # initialize image augmentations
        self.img_augs = self.get_img_augs()

    def get_img_augs(self):
        if self.is_train:
            return A.Compose([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, always_apply=False, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.Blur(blur_limit=7, p=0.5),
                    A.GaussNoise(var_limit=(10, 50), p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                                 p=0.5),
                    # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5),
                    A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.5),
                    # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
                    A.RandomToneCurve(scale=0.1, p=0.5),
            ])
        else:
            return None

    def get_raw_image(self, i, camera=None):
        if camera is None:
            camera = self.cameras[0]
        ind = self.ids[i]
        img_path = os.path.join(self.path, 'images', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        img = Image.open(img_path)
        return img

    def get_raw_img_size(self, i=0, cam=None):
        if cam is None:
            cam = self.cameras[0]
        img = self.get_raw_image(i, cam)
        img = np.asarray(img)
        return img.shape[0], img.shape[1]

    def get_image(self, i, camera=None):
        if camera is None:
            camera = self.cameras[0]
        img = self.get_raw_image(i, camera)
        for key in self.calib.keys():
            if camera in key:
                camera = key
                break
        K = self.calib[camera]['camera_matrix']['data']
        r, c = self.calib[camera]['camera_matrix']['rows'], self.calib[camera]['camera_matrix']['cols']
        K = np.asarray(K, dtype=np.float32).reshape((r, c))
        return img, K

    def get_images_data(self, i):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []

        for cam in self.cameras:
            img, K = self.get_image(i, cam)
            # if self.is_train:
            #     img = self.img_augs(image=np.asarray(img))['image']

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = sample_augmentation(self.lss_cfg, is_train=self.is_train)
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
            T_robot_cam = self.calib['transformations'][f'T_base_link__{cam}']['data']
            T_robot_cam = np.asarray(T_robot_cam, dtype=np.float32).reshape((4, 4))
            rot = torch.as_tensor(T_robot_cam[:3, :3])
            tran = torch.as_tensor(T_robot_cam[:3, 3])

            imgs.append(img)
            rots.append(rot)
            trans.append(tran)
            intrins.append(K)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        inputs = [torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                  torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
        inputs = [torch.as_tensor(i, dtype=torch.float32) for i in inputs]

        return inputs

    def seg_label_to_color(self, seg_label):
        coco_colors = [(np.array(color['color'])).tolist() for color in COCO_CATEGORIES] + [[0, 0, 0]]
        seg_label = np.asarray(seg_label)
        # transform segmentation labels to colors
        size = [s for s in seg_label.shape] + [3]
        seg_color = np.zeros(size, dtype=np.uint8)
        for color_i, color in enumerate(coco_colors):
            seg_color[seg_label == color_i] = color
        return seg_color

    def get_seg_label(self, i, camera=None):
        if camera is None:
            camera = self.cameras[0]
        id = self.ids[i]
        seg_path = os.path.join(self.path, 'images/seg/', '%s_%s.npy' % (id, camera))
        assert os.path.exists(seg_path), f'Image path {seg_path} does not exist'
        seg = Image.fromarray(np.load(seg_path))
        size = self.get_raw_img_size(i, camera)
        transform = torchvision.transforms.Resize(size)
        seg = transform(seg)
        return seg
    
    def get_semantic_cloud(self, i, classes=None, vis=False, points_source='lidar'):
        coco_classes = [i['name'].replace('-merged', '').replace('-other', '') for i in COCO_CATEGORIES] + ['void']
        if classes is None:
            classes = np.copy(coco_classes)
        # ids of classes in COCO
        selected_labels = []
        for c in classes:
            if c in coco_classes:
                selected_labels.append(coco_classes.index(c))

        lidar_points = position(self.get_cloud(i, points_source=points_source))
        points = []
        labels = []
        for cam in self.cameras[::-1]:
            seg_label_cam = self.get_seg_label(i, camera=cam)
            seg_label_cam = np.asarray(seg_label_cam)

            K = self.calib[cam]['camera_matrix']['data']
            K = np.asarray(K, dtype=np.float32).reshape((3, 3))
            E = self.calib['transformations'][f'T_base_link__{cam}']['data']
            E = np.asarray(E, dtype=np.float32).reshape((4, 4))
    
            lidar_points = torch.as_tensor(lidar_points)
            E = torch.as_tensor(E)
            K = torch.as_tensor(K)
    
            cam_points = ego_to_cam(lidar_points.T, E[:3, :3], E[:3, 3], K).T
            mask = get_only_in_img_mask(cam_points.T, seg_label_cam.shape[0], seg_label_cam.shape[1])
            cam_points = cam_points[mask]

            # colorize point cloud with values from segmentation image
            uv = cam_points[:, :2].numpy().astype(int)
            seg_label_cam = seg_label_cam[uv[:, 1], uv[:, 0]]
    
            points.append(lidar_points[mask].numpy())
            labels.append(seg_label_cam)

        points = np.concatenate(points)
        labels = np.concatenate(labels)
        colors = self.seg_label_to_color(labels)
        assert len(points) == len(colors)

        # mask out points with labels not in selected classes
        mask = np.isin(labels, selected_labels)
        points = points[mask]
        colors = colors[mask]

        if vis:
            colors = normalize(colors)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        return points, colors

    def get_geom_height_map(self, i, cached=True, dir_name=None, points_source='lidar', **kwargs):
        """
        Get height map from lidar point cloud.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :param kwargs: additional arguments for height map estimation
        :return: height map (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, f'terrain_{2*self.dphys_cfg.d_max}x{2*self.dphys_cfg.d_max}', 'lidar')
        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
        if cached and os.path.exists(file_path):
            lidar_hm = np.load(file_path, allow_pickle=True).item()
        else:
            cloud = self.get_cloud(i, points_source=points_source)
            points = position(cloud)
            lidar_hm = self.estimate_heightmap(points, **kwargs)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, lidar_hm)
        height = lidar_hm['z']
        mask = lidar_hm['mask']
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def get_terrain_height_map(self, i, method='footprint', cached=False, dir_name=None, points_source='lidar'):
        """
        Get height map from trajectory points.
        :param i: index of the sample
        :param method: method to estimate height map from trajectory points
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :param obstacle_classes: classes of obstacles to include in the height map
        :return: heightmap (2 x H x W), where 2 is the number of channels (z and mask)
        """
        assert method in ['dphysics', 'footprint']
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'traj', 'footprint')
        if method == 'dphysics':
            height = self.get_traj_dphyics_terrain(i)
            h, w = (int(2 * self.dphys_cfg.d_max // self.dphys_cfg.grid_res),
                    int(2 * self.dphys_cfg.d_max // self.dphys_cfg.grid_res))
            # Optimized height map shape is 256 x 256. We need to crop it to 128 x 128
            H, W = height.shape
            if H == 256 and W == 256:
                # print(f'Height map shape is {H} x {W}). Cropping to 128 x 128')
                # select only the h x w area from the center of the height map
                height = height[int(H // 2 - h // 2):int(H // 2 + h // 2),
                                int(W // 2 - w // 2):int(W // 2 + w // 2)]
            # poses in grid coordinates
            poses = self.get_traj(i)['poses']
            poses_grid = poses[:, :2, 3] / self.dphys_cfg.grid_res + np.asarray([w / 2, h / 2])
            poses_grid = poses_grid.astype(int)
            # crop poses to observation area defined by square grid
            poses_grid = poses_grid[(poses_grid[:, 0] > 0) & (poses_grid[:, 0] < w) &
                                    (poses_grid[:, 1] > 0) & (poses_grid[:, 1] < h)]

            # visited by poses dilated height map area mask
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[poses_grid[:, 0], poses_grid[:, 1]] = 1
            mask = cv2.dilate(mask, kernel, iterations=2)
        else:
            assert method == 'footprint'
            file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
            if cached and os.path.exists(file_path):
                hm_rigid = np.load(file_path, allow_pickle=True).item()
            else:
                seg_points, _ = self.get_semantic_cloud(i, classes=self.lss_cfg['obstacle_classes'],
                                                        points_source=points_source, vis=False)
                traj_points = self.get_footprint_traj_points(i)
                points = np.concatenate((seg_points, traj_points), axis=0)
                hm_rigid = self.estimate_heightmap(points, robot_radius=None)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.save(file_path, hm_rigid)
            height = hm_rigid['z']
            mask = hm_rigid['mask']
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def front_height_map_mask(self):
        camera = self.cameras[0]
        K = self.calib[camera]['camera_matrix']['data']
        r, c = self.calib[camera]['camera_matrix']['rows'], self.calib[camera]['camera_matrix']['cols']
        K = np.asarray(K, dtype=np.float32).reshape((r, c))

        # get fov from camera intrinsics
        img_h, img_w = self.lss_cfg['data_aug_conf']['H'], self.lss_cfg['data_aug_conf']['W']
        fx, fy = K[0, 0], K[1, 1]

        fov_x = 2 * np.arctan2(img_h, 2 * fx)
        fov_y = 2 * np.arctan2(img_w, 2 * fy)

        # camera frustum mask
        d = self.dphys_cfg.d_max
        res = self.dphys_cfg.grid_res
        h, w = 2 * d / res, 2 * d / res
        h, w = int(h), int(w)
        mask = np.zeros((h, w), dtype=np.float32)

        to_grid = lambda x: np.array([x[1], x[0]]) / res + np.array([h // 2, w // 2])
        A = to_grid([0, 0])
        B = to_grid([d * np.tan(fov_y / 2), d])
        C = to_grid([-d * np.tan(fov_y / 2), d])

        # select triangle
        rr, cc = polygon([A[0], B[0], C[0]], [A[1], B[1], C[1]], mask.shape)
        mask[rr, cc] = 1.

        return mask

    def get_sample(self, i):
        img, rot, tran, intrins, post_rots, post_trans = self.get_images_data(i)
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        if self.only_front_cam:
            mask = self.front_height_map_mask()
            hm_geom[1] = hm_geom[1] * torch.from_numpy(mask)
            hm_terrain[1] = hm_terrain[1] * torch.from_numpy(mask)
        return img, rot, tran, intrins, post_rots, post_trans, hm_geom, hm_terrain


class RobinGasPoints(RobinGas):
    def __init__(self, path, lss_cfg, dphys_cfg=DPhysConfig(), is_train=True, only_front_cam=False, points_source='lidar'):
        super(RobinGasPoints, self).__init__(path, lss_cfg,
                                             dphys_cfg=dphys_cfg, is_train=is_train, only_front_cam=only_front_cam)
        assert points_source in ['lidar', 'radar', 'lidar_radar']
        self.points_source = points_source

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        hm_geom = self.get_geom_height_map(i, points_source=self.points_source)
        hm_terrain = self.get_terrain_height_map(i, points_source=self.points_source)
        if self.only_front_cam:
            mask = self.front_height_map_mask()
            hm_geom[1] = hm_geom[1] * torch.from_numpy(mask)
            hm_terrain[1] = hm_terrain[1] * torch.from_numpy(mask)
        points = torch.as_tensor(position(self.get_cloud(i, points_source=self.points_source))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, hm_geom, hm_terrain, points


def compile_data(robot='tradr', seq_i=None, small=False):
    dphys_cfg = DPhysConfig()
    dphys_cfg_path = os.path.join(data_dir, '../config/dphys_cfg.yaml')
    assert os.path.isfile(dphys_cfg_path), 'Config file %s does not exist' % dphys_cfg_path
    dphys_cfg.from_yaml(dphys_cfg_path)

    lss_cfg_path = os.path.join(data_dir, f'../config/lss_cfg_{robot}.yaml')
    assert os.path.isfile(lss_cfg_path)
    lss_cfg = read_yaml(lss_cfg_path)

    if seq_i is not None:
        path = robingas_seq_paths[robot][seq_i]
    else:
        path = np.random.choice(robingas_seq_paths[robot])

    ds = RobinGas(path=path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)
    if small:
        ds = ds[np.random.choice(len(ds), 32, replace=False)]

    return ds
