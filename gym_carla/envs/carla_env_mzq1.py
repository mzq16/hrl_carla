from __future__ import division
import copy
from queue import Queue
import numpy as np
import random
import time
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

import carla


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
        ):
        # initialize client with timeout(include world and map)
        #print('connecting to Carla server...')
        self._init_client(carla_map, host, port, seed)
        #print('Carla server connected!')
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
        self._ev_handle = EgoVehicle(self._client, port)
        self._ev_handle.init_bev_observation(file_path)

        # set other vehicles
        self._ov_handle = OtherVehicle(self._client, port, num_vehicles=num_vehicles)

        # set other walkers
        #self._ow_handle = OtherWalker(self._client)
        self._ZombieWalkerHandler = ZombieWalkerHandler(self._client, number_walkers=num_walkers)

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
        
        self._info = {}
        snap_shot = self._world.get_snapshot()
        
        timestamp = snap_shot.timestamp
        #self.hud.on_world_tick(timestamp)
        self._timestamp = utils.timestamp_tran(self._timestamp, timestamp)
        # 1.control(auto)
        throttle, brake, steer = utils.action_to_control(action)
        ev_control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self._ev_handle.apply_control(ev_control)
        
        # 2.get observation(next state)
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
        self._ev_handle.sensor_queue.put((None, 'EOF'))
        while self._ev_handle.sensor_queue.qsize() > 1:
            data, name = self._ev_handle.sensor_queue.get()
            self._obs[name] = data
        assert self._ev_handle.sensor_queue.qsize() == 1, 'the sensor queue length is wrong' 
        
        #full_obs = self._ev_handle.get_full_obs()
        #full_obs_render = cv.resize(full_obs, (512,512))
        #self._obs['full_obs'] = full_obs_render
        #if 'camera' in self._obs:
        #    time.sleep(0.5)
        #    array = utils.preprocess_img(self._obs['camera'])   
        #    self._obs['camera'] = array
        
        
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

        _, name = self._ev_handle.sensor_queue.get()
        assert name == 'EOF', 'the final signal of sensor queue is wrong'
        self._world.tick()
        return self._obs, reward, done, self._info

    def reset(self):
        
        #  reset
        self._ev_handle.reset()
        #if self.epoch == 0:
         #   self.epoch += 1
        self._ov_handle.reset()
        #self._ow_handle.reset()
        #self._ZombieWalkerHandler.clean()
        self._ZombieWalkerHandler.reset()
        if self._reward_handle is not None:
            self._reward_handle.destroy()
        self._reward_handle = Reward(self._ev_handle._ego_vehicle, self._ev_handle.get_route_trace(), self._ev_handle.get_trafficlight_manager())
        # self._reward_handle.reset()
        self._world.tick()
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
        #self._ow_handle.destroy()
        self._ZombieWalkerHandler.clean()
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

    def get_statistic(self, info):
        epoch_statistic = {}
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
