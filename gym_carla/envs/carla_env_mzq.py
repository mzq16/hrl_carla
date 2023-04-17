from __future__ import division
import copy
from queue import Queue
import numpy as np
import random
import time
import pickle
import psutil
from numpy.core.shape_base import block
from gym_carla.envs.dynamic_weather import Weather
import gym
from gym import spaces
import sys
from gym_carla.envs.egovehicle_r import EgoVehicle
# from gym_carla.envs.egovehicle import EgoVehicle
from gym_carla.envs.othervehicle import OtherVehicle
from gym_carla.envs.otherwalker import OtherWalker
from gym_carla.envs.zombie_walker_handler import ZombieWalkerHandler
from gym_carla.envs.hud import HUD
from gym_carla.envs.reward import Reward
from gym_carla.envs import utils
import pygame
import cv2 as cv
import argparse
import copy
import os
import pickle
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
#from torchpack.utils.tqdm import tqdm
from tqdm import tqdm
import math
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models.fusion_models import bevfusion
from mmdet3d.models import build_model, build_fusion_model, FUSIONMODELS
from mmcv.utils import Registry, build_from_cfg
import torch.nn as nn
from collections import deque
import carla


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

class CarlaEnv_mzq(gym.Env):
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 2000, 
        seed: int = 2021, 
        carla_map: str = None, 
        file_path: str = 'Town10HD_Opt.h5', 
        bool_render: bool = False, 
        num_vehicles: int = 100,
        num_walkers: int = 200,
        use_bev_fusion: bool = False,
        device: torch.device = torch.device(2),
        num_des: int = 1,
        is_meta: bool  = True,
        ):
        # initialize client with timeout(include world and map)
        #print('connecting to Carla server...')
        self._init_client(carla_map, host, port, seed)
        self.device = device
        self.num_des = num_des
        self.eval_info = []
        ''''''
        # set bev-fusion model
        self.use_bev_fusion = use_bev_fusion
        if use_bev_fusion:
            self.cfg = self.get_cfg()
            self.temp_data = self.get_temp_data(path='template.pkl', device=device)
            self.bev_model = self.get_bev_model(cfg=self.cfg, device=device)
        

        # print('Carla server connected!')
        # get settings
        # self._init_settings = self._world.get_settings()
        
        self.is_render = bool_render

        # set action and observation space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            camera=gym.spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),
            full_obs=gym.spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8),
            bev_render=gym.spaces.Box(low=0, high=255, shape=(192, 192, 3), dtype=np.uint8),
            bev_mask=gym.spaces.Box(low=0, high=255, shape=(15, 192, 192), dtype=np.uint8),
            # control
            throttle=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            steer=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            brake=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            gear=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # 0-5 in rl_wrapper preprocess_obs() normalization
            # velocity vel:m/s, acc:m/s^2
            acc_xy=spaces.Box(low=-1e3, high=1e3, shape=(2,), dtype=np.float32),
            vel_xy=spaces.Box(low=-1e2, high=1e2, shape=(2,), dtype=np.float32),
        )

        # set Weather
        # weather = Weather(self._world.get_weather())
        self._world.set_weather(carla.WeatherParameters.ClearNoon)
        
        # set ego vehicle
        self._ev_handle = EgoVehicle(self._client, port, multi_cam=use_bev_fusion, num_des=num_des, is_meta=is_meta)
        self._ev_handle.init_bev_observation(file_path)

        # set other vehicles
        self._ov_handle = OtherVehicle(self._client, port, num_vehicles=num_vehicles)

        # set other walkers
        self._ow_handle = OtherWalker(self._client, num_walkers=num_walkers)
        #self._ZombieWalkerHandler = ZombieWalkerHandler(self._client, number_walkers=num_walkers)

        self._history_queue = deque(maxlen=20)
        self._history_idx = [-16, -11, -6, -1]
        # HUD
        #self.hud = HUD(320, 720)
        # self.clock = pygame.time.Clock()

        # Reward
        # Reward class need ego vehicle's destination, route trace to create  
        self._reward_handle = None

        # set pygame init
        if self.is_render:
            self._screen = utils.init_render(1680, 1680)
        self._obs = {}
        self._info = {}

        self.epoch_statistic = {'d':0}
        self.epoch = 0


    def step(self, action):
        sensor_number = 0
        self._info = {}
        
        # 1.control(auto)
        throttle, brake, steer = utils.action_to_control(action)
        ev_control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self._ev_handle.apply_control(ev_control)
        
        # after action, world tick()
        self.frame = self._world.tick()
        snap_shot = self._world.get_snapshot()
        timestamp = snap_shot.timestamp
        #self.hud.on_world_tick(timestamp)
        self._timestamp = utils.timestamp_tran(self._timestamp, timestamp)


        # 2.get observation(next state)
        if self.use_bev_fusion:
            sensor_N = 8
        else:
            sensor_N = 3
        while sensor_number < sensor_N: # 6 camera, lidar, GPS
            while not self._ev_handle.sensor_queue.empty():
                data, name = self._ev_handle.sensor_queue.get(True, 1)
                if data.frame == self.frame:
                    self._obs[name] = data
                    sensor_number += 1
        ''''''
        # get info 
        if self.use_bev_fusion:
            sensor_info = {}
            sensor_info['frame'] = self.frame
            tmp_sensor = self._ev_handle._camera
            tmp_info = self.get_sensor_info(tmp_sensor,'camera')
            sensor_info['camera_0'] = tmp_info

            tmp_sensor = self._ev_handle._lidar
            tmp_info = self.get_sensor_info(tmp_sensor,'lidar')
            sensor_info['lidar'] = tmp_info
            for i in range(5): # without the camera front
                tmp_sensor = self._ev_handle._other_camera_list[i]
                tmp_info = self.get_sensor_info(tmp_sensor,'camera')
                sensor_info[f'camera_{i+1}'] = tmp_info
            
            # get bev data
            cam, lidar_np = self.get_cam_lidar(self._obs)
            bev_data = utils.get_bev_data(info=sensor_info, cam=cam, lidar_np=lidar_np, my_data=self.temp_data, device=self.device)
            with torch.inference_mode():
                output=self.bev_model(**bev_data)
            lidar_render = bev_data['points'][0].cpu().numpy()
            self._obs['output'] = output
            self._obs['lidar_render'] = lidar_render
        
        obs_bev_dict = self._ev_handle.get_bev_observation(50.0, False)
        if 'rendered' in obs_bev_dict:
            img = obs_bev_dict['rendered']
            self._obs['bev_render'] = img
        if 'masks' in obs_bev_dict:
            array = obs_bev_dict['masks']
            self._obs['bev_mask'] = array
        if 'ev_history_masks' in obs_bev_dict:
            array = obs_bev_dict['ev_history_masks']
            self._obs['bev_ev_history_mask'] = array
        if self.use_bev_fusion:    
            if 'image_map' in obs_bev_dict:
                array = obs_bev_dict['image_map']
                self._obs['image_map'] = array
        
            image_map = self.get_mask_from_bev(output=output, bev_mask_map=self._obs['image_map'])
            self._obs['image_map'] = image_map
            c_vehicle_masks = self.get_history_mask()
            array = obs_bev_dict['masks']
            array[4:8] = c_vehicle_masks
            self._obs['bev_mask'] = array
        
        state_v, state_ctl = self._ev_handle.get_state()
        self._obs['velocity'] = state_v
        self._obs['control'] = state_ctl
        
        # 3.calculate reward
        # 4.done? terminate or not.
        reward, done, reward_info = self._reward_handle.tick(self._timestamp)
        # self._info['timestamp']  = timestamp # copy got some problems. 'Pickling of "carla.libcarla.Timestamp" instances is not enabled'
                                               # but self._info['timestamp']  = _timestamp is ok
        
        self._info['timestamp'] = self._timestamp
        self._info['reward'] = reward_info
        self._info['text'] = reward_info['text']
        ev_loc = self._ev_handle._ego_vehicle.get_location()
        self._info['text'].append(f'ev_loc: {ev_loc.x:5.2f}, {ev_loc.y:5.2f}')
        self.add_history(reward_info)
        if done:
            #self._info['episode_stat'] = self.epoch_statistic
            self.get_statistic(reward_info)
        
        return self._obs, reward, done, self._info

    def reset(self):
        self._ev_handle.reset()
        self._ov_handle.reset()
        self._ow_handle.reset()
        #self._ZombieWalkerHandler.clean()
        #self._ZombieWalkerHandler.reset()
        if self._reward_handle is not None:
            self._reward_handle.destroy()
        self._reward_handle = Reward(self._ev_handle._ego_vehicle, self._ev_handle.get_route_trace(), self._ev_handle.get_trafficlight_manager())
        # self._reward_handle.reset()
        self._history_queue.clear()
        self.frame = self._world.tick()
        snap_shot = self._world.get_snapshot()
        timestamp = snap_shot.timestamp
        self._timestamp = {
        'step': 0,
        'frame': 0,
        'relative_wall_time': 0.0,
        'wall_time': timestamp.platform_timestamp,
        'relative_simulation_time': 0.0,
        'simulation_time': timestamp.elapsed_seconds,
        'start_frame': timestamp.frame,
        'start_wall_time': timestamp.platform_timestamp,
        'start_simulation_time': timestamp.elapsed_seconds
        }
        self.init_statistic()
        #obs = self.gen_init_obs()
        obs, reward, done, info = self.step([0, 0])
        return obs, reward, done, info

    def close(self):
        self._ev_handle.destroy()
        self._ov_handle.destroy()
        self._ow_handle.destroy()
        #self._ZombieWalkerHandler.clean()
        if self._reward_handle is not None:
            self._reward_handle.destroy()
        pygame.quit()
        self.set_synchronous_mode(False, 20)

    def render(self, **kwargs):
        # camera and BEV render
        if 'camera' in self._obs:
            camera_surface = utils.display_surface(self._obs['camera'])
            self._screen.blit(camera_surface, (0, 0))
        if 'bev' in self._obs:
            bev_img = self._obs['bev']
            bev_surface = pygame.surfarray.make_surface(bev_img.swapaxes(0, 1))
            self._screen.blit(bev_surface, (640, 0))
        if 'full_obs' in self._obs:
            full_bev_img = self._obs['full_obs']
            full_bev_surface = pygame.surfarray.make_surface(full_bev_img.swapaxes(0, 1))
            self._screen.blit(full_bev_surface, (840, 0))
        # HUD render
        #self.hud.render(self._screen)
        pygame.display.flip()

    def _init_client(self, carla_map, host, port, seed=2021):
        # initialize client
        client = carla.Client(host, port)
        client.set_timeout(60.0)
        self._client = client

        # set map and world
        if carla_map is None:
            self._world = client.get_world()
            #print("do not load a new map")
        else:
            self._world = client.load_world(carla_map)
            self._map = self._world.get_map()
            print("load new map: {}".format(carla_map))

        # get traffic manager
        tm_port = port + 6000
        while self.is_port_used(tm_port):
            tm_port += 1
        self._tm = self._client.get_trafficmanager(tm_port)

        # init sync and no rendering
        #print('set sync')
        self.set_synchronous_mode(sync_mode=True, fps=10)
        #print('set no rendering')
        self.set_no_rendering_mode(rendering_mode=True)

        # init statistic history
        self.init_statistic()
        
    def set_synchronous_mode(self, sync_mode=True, fps=10):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync_mode
        settings.fixed_delta_seconds = 1.0 / fps
        self._world.apply_settings(settings)
        self._tm.set_synchronous_mode(sync_mode)

    def set_no_rendering_mode(self, rendering_mode=True):
        settings = self._world.get_settings()
        settings.no_rendering_mode = rendering_mode
        self._world.apply_settings(settings)

    def init_statistic(self):
        self.history_buffer = {}
        self.history_buffer['collisions_layout'] = []
        self.history_buffer['collisions_vehicle'] = []
        self.history_buffer['collisions_walkers'] = []
        self.history_buffer['collisions_other'] = []
        self.history_buffer['runredlight_info'] = []

    def get_cfg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default='../testBEVfusion/bevfusion-main/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml')
        parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
        parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
        parser.add_argument("--bbox-score", type=float, default=None)
        parser.add_argument("--map-score", type=float, default=0.5)
        parser.add_argument("--out-dir", type=str, default="viz/test")
        args, opts = parser.parse_known_args()
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg = Config(recursive_eval(configs), filename=args.config)
        return cfg
        
    def get_bev_model(self, cfg, device='cuda:2'):
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.cuda.set_device(device=device)
        model = build_from_cfg(cfg=cfg.model, registry=FUSIONMODELS)
        checkpoint = '../testBEVfusion/bevfusion-main/pretrained/bevfusion-det.pth'
        ckpt = load_checkpoint(model, checkpoint, map_location="cpu")
        model = model.to(device)
        model = model.eval()
        return model
    
    def get_temp_data(self, path='template.pkl', device='cuda:2'):
        with open(path,'rb') as f:
            my_data = pickle.load(f)
        for key in my_data.keys():
            if isinstance(my_data[key], torch.Tensor):
                my_data[key] = my_data[key].to(device)
            if isinstance(my_data[key], list):
                if isinstance(my_data[key][0], torch.Tensor):
                    my_data[key] = [m.to(device) for m in my_data[key]]
        return my_data

    def get_statistic(self, info):
        epoch_statistic = {}
        eval_info = {}
        route_info = info['route_info']
        route_complete_in_m = info['route_info']['route_completed_in_m']
        route_complete_in_km = float(route_complete_in_m / 1000.0) + 1e-2 # in case divide by zero
        n_collision_layout = int(len(self.history_buffer['collisions_layout']))
        n_collisions_vehicle = int(len(self.history_buffer['collisions_vehicle']))
        n_collisions_walkers = int(len(self.history_buffer['collisions_walkers']))
        n_collisions_other = int(len(self.history_buffer['collisions_other']))
        n_run_red_light = int(len(self.history_buffer['runredlight_info']))
        n_collision = n_collision_layout + n_collisions_vehicle + n_collisions_walkers + n_collisions_other + n_run_red_light
        epoch_statistic['collisions_layout_km'] = n_collision_layout /route_complete_in_km
        epoch_statistic['n_collisions_vehicle_km'] = n_collisions_vehicle /route_complete_in_km
        epoch_statistic['n_collisions_walkers_km'] = n_collisions_walkers /route_complete_in_km
        epoch_statistic['n_collisions_other_km'] = n_collisions_other /route_complete_in_km
        epoch_statistic['n_run_red_light_km'] = n_run_red_light /route_complete_in_km
        epoch_statistic['n_collision_km'] = n_collision /route_complete_in_km
        self._info['episode_stat'] = epoch_statistic

        eval_info['n_collision'] = n_collision
        eval_info['n_collision_layout'] = n_collision_layout 
        eval_info['n_collisions_vehicle'] = n_collisions_vehicle
        eval_info['n_collisions_walkers'] = n_collisions_walkers
        eval_info['n_collisions_other'] = n_collisions_other
        eval_info['n_run_red_light'] = n_run_red_light
        eval_info['route_complete_in_m'] = route_complete_in_m
        eval_info['route_wrong'] = route_info['is_wrong']
        eval_info['route_deviation'] = route_info['is_deviation']
        eval_info['full_route_length_in_m'] = route_info['full_route_length_in_m']
        eval_info['route_finished'] = route_info['is_route_finished']
        self.eval_info.append(eval_info)
        self._info['eval_info'] = eval_info
        

    def get_eval_done(self):
        return len(self.eval_info) == self.num_des

    def get_sensor_info(self, sensor, type_name='camera'):
        info = {}
        if type_name == 'camera':
            intrinsic_matrix = self.get_intrinsic_matrix(sensor)
            extrinsix_matrix = sensor.get_transform().get_inverse_matrix()
            info['in'] = intrinsic_matrix
            info['ex'] = extrinsix_matrix
        elif type_name == 'lidar':
            extrinsix_matrix = sensor.get_transform().get_inverse_matrix()
            info['ex'] = extrinsix_matrix
        else:
            raise TypeError('wrong sensor type')
        return info

    def get_cam_lidar(self, obs):
        cam = {}
        if 'camera' in obs:
            cam['camera_0'] = utils.preprocess_img(obs['camera'])
        for i in range(1,6):
            name = f'camera_{i}'
            cam[name] = utils.preprocess_img(obs[name])
        lidar_np = utils.preprocess_lidar(obs['lidar'])
        # left-handed to right-handed

        return cam, lidar_np
    

    def get_mask_from_bev(self, output, bev_mask_map):
        mask = np.zeros([self._ev_handle._width, self._ev_handle._width], dtype=np.uint8)
        ex_matrix = self._ev_handle.get_ego_vehicle().get_transform().get_matrix()
        bboxes = self.get_bbox_from_bev(output)
        vehicles_list = []
        if bboxes is not None and len(bboxes) > 0:
            coords = bboxes.corners[:, [0, 3, 7, 4, 0]]
            #coords = bboxes.corners[:, :, [1, 0, 2]]  # change x,y 
            if isinstance(coords, torch.Tensor):
                coords = coords.numpy()
            coords = np.pad(coords,((0,0),(0,0),(0,1)), 'constant', constant_values=1) # expand dim
            vehicles_list.append(coords)
            
            for index in range(coords.shape[0]):
                tmp_corners = []
                for i in range(5):
                    tmp_corner = np.matmul(ex_matrix, coords[index][i].T) # ego vehicle coord -> global coord
                    #tmp_corners.append(tmp_corner[:2])
                    tmp_corners.append(carla.Location(x=tmp_corner[0],y=tmp_corner[1]))
                tmp_corner_inpixel = np.array([[self._ev_handle._world_to_pixel(corner)] for corner in tmp_corners])
                corners_warped = cv.transform(tmp_corner_inpixel, self._ev_handle.M_warp)
                
                cv.fillConvexPoly(bev_mask_map, np.round(corners_warped).astype(np.int32), color=(0, 0, 255))
                cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        self._history_queue.append(mask.astype(np.bool))
        return bev_mask_map

    def get_history_mask(self):
        q_size = len(self._history_queue)
        vehicle_masks = []
        for idx in self._history_idx:
            idx = max(idx, -1 * q_size)
            mask = self._history_queue[idx]
            vehicle_masks.append(mask)
        c_vehicle_history = np.array([m * 255 for m in vehicle_masks])
        return c_vehicle_history

    def get_bbox_from_bev(self, output):
        bboxes = output[0]["boxes_3d"].tensor.numpy()
        scores = output[0]["scores_3d"].numpy()
        labels = output[0]["labels_3d"].numpy()
        indices = scores >= 0.25
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
        bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        return bboxes

    @staticmethod
    def get_intrinsic_matrix(camera):
        # Intrinsic Matrix
        IM = np.identity(3)
        WINDOW_WIDTH = float(camera.attributes['image_size_x'])
        WINDOW_HEIGHT = float(camera.attributes['image_size_y'])
        FOV = float(camera.attributes['fov'])
        IM[0, 2] = WINDOW_WIDTH / 2.0
        IM[1, 2] = WINDOW_HEIGHT / 2.0
        IM[0, 0] = IM[1, 1] = WINDOW_WIDTH / (2.0 * math.tan(FOV * math.pi / 360.0))
        return IM


    def add_history(self, info):
        # collision statistic
        if info['collision_info']:
            if info['collision_info']['collision_type'] == 0:
                self.history_buffer['collisions_layout'].append(info['collision_info'])
            if info['collision_info']['collision_type'] == 1:
                self.history_buffer['collisions_vehicle'].append(info['collision_info'])
            if info['collision_info']['collision_type'] == 2:
                self.history_buffer['collisions_walkers'].append(info['collision_info'])
            if info['collision_info']['collision_type'] == -1:
                self.history_buffer['collisions_other'].append(info['collision_info'])
        if info['runredlight_info']:
            self.history_buffer['runredlight_info'].append(info['runredlight_info'])
            

    def get_world(self):
        return self._world

    def get_ego_vehicle(self):
        return self._ev_handle.get_ego_vehicle()

    def get_obs(self):
        return self._obs
    
    def set_render(self, bool):
        self.is_render = bool
        if self.is_render:
            self._screen = utils.init_render(1280, 1280)

    def gen_init_obs(self):
        obs = {}
        obs['bev'] = np.zeros((192,192,3))
        obs['camera'] = np.zeros((600,800,3))
        return obs

    @staticmethod
    def record_and_info(name, obs):
        # obs is dict
        if name in obs:
            obs[name].save_to_disk('_out/%06d-%s.png' % (obs[name].frame, name))
            print(obs[name].frame, name)
    
    @staticmethod
    def is_port_used(port):
        return port in [conn.laddr.port for conn in psutil.net_connections()]
