from os import stat
import gym
from sentry_sdk import start_transaction
from sklearn import preprocessing
from gym_carla.envs import utils
from gym_carla.envs.carla_env_mzq import CarlaEnv_mzq
import numpy as np
import cv2
import math

COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (187, 187, 186)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)

class RLBEVWrapper(gym.Wrapper):
    def __init__(self, env: CarlaEnv_mzq):
        # super(RLBEVWrapper, self).__init__(env)  this code is the same as the following
        self.env = env
        self.action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self._obs = {}
        state_low, state_high = self.preprocess_statespace()
        # change observate space
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
            birdview=gym.spaces.Box(low=0, high=255, shape=(16, 192, 192), dtype=np.uint8)
        )

    def step(self, action):
        
        self._raw_obs = None
        obs, reward, done, info = self.env.step(action)
        #self._raw_obs = obs
        # tranfer obs to fit observation space
        '''
        if 'camera' in obs:
            array = utils.preprocess_img(obs['camera'])
            self._obs['camera'] = array
        if 'rendered' in obs:
            img = obs['rendered']
            self._obs['bev'] = img
        '''
        obs_dict = self.preprocess_obs(obs)
        self._raw_obs = {
            'bev_render': obs['bev_render'],
            'info': info,
            'state': obs_dict['state'],
            'camera': obs['camera'],
        }
        self._raw_obs['info'] = info
        return obs_dict, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        obs_dict = self.preprocess_obs(obs)
        '''
        print('set sync')
        self.env.set_synchronous_mode(sync_mode=True, fps=10)
        print('set no rendering')
        self.env.set_no_rendering_mode(rendering_mode=True)
        '''
        return obs_dict

    def render(self, mode="rgb_array"):
        #self.env.render()
        # in callback has 'set attr(action_value)'
        self._raw_obs['action'] = self.action
        self._raw_obs['action_value'] = self.action_value
        self._raw_obs['action_log_probs'] = self.action_log_probs
        self._raw_obs['action_mu'] = self.action_mu
        self._raw_obs['action_sigma'] = self.action_sigma
        im = self.im_render(self._raw_obs)
        #im = self._raw_obs['bev_render']
        return im

    def close(self):
        self.env.close()

    @staticmethod
    def im_render(obs):
        im_birdview = obs['bev_render']
        im_camera = obs['camera']
        im_camera=cv2.resize(im_camera,(192,192))
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*3, c], dtype=np.uint8)
        im[:h, w:2*w] = im_birdview
        im[:h, :w] = im_camera
        
        #action = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        action_str = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        mean_str = np.array2string(obs['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(obs['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(obs['state'], precision=2, separator=',', suppress_small=True)

        txt_t = f'step:{obs["info"]["timestamp"]["step"]:5}, frame:{obs["info"]["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{obs["action_value"]:5.2f} p:{obs["action_log_probs"]:5.2f}'
        im = cv2.putText(im, txt_1, (2*w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (2*w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mean_str} b{sigma_str}'
        im = cv2.putText(im, txt_3, (2*w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i, txt in enumerate(obs['reward_debug']['debug_texts'] +
                                obs['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i_txt, txt in enumerate(obs['info']['text']):
                im = cv2.putText(im, txt, (2*w, 36+(i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return im

    @staticmethod
    def preprocess_obs(obs, train=True):
        state_list = []
        obs_dict = {}
        if 'bev_mask' in obs:
            bev = obs['bev_mask']

        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        if 'control' in obs:
            state_list.append(obs['control'].throttle)
            state_list.append(obs['control'].steer)
            state_list.append(obs['control'].brake)
            state_list.append(obs['control'].gear / 5.0)
        if 'velocity' in obs:
            vel_x = obs['velocity'].x
            vel_y = obs['velocity'].y
            state_list.append(vel_x)
            state_list.append(vel_y)
        state = np.array(state_list)
        
        if not train:
            bev = np.expand_dims(bev, 0)
            state = np.expand_dims(state, 0)

        obs_dict['birdview'] = bev
        obs_dict['state'] = state.astype(np.float32)

        return obs_dict
    
    def preprocess_statespace(self):
        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        state_space = []
        state_space.append(self._observation_space['throttle'])
        state_space.append(self._observation_space['steer'])
        state_space.append(self._observation_space['brake'])
        state_space.append(self._observation_space['gear'])
        state_space.append(self._observation_space['vel_xy'])
        state_low = np.concatenate([s.low for s in state_space])
        state_high = np.concatenate([s.high for s in state_space])
        return state_low, state_high

class RLdiaynWrapper(gym.Wrapper):
    def __init__(self, env: CarlaEnv_mzq, number_z=50):
        # super(RLBEVWrapper, self).__init__(env)  this code is the same as the following
        self.env = env
        self.number_z = number_z
        self.action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self._obs = {}
        state_low, state_high = self.preprocess_statespace()
        # change observate space
        # may auto add batch dim, i dont know
        # env_vec will auto add batch dim
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
            birdview=gym.spaces.Box(low=0, high=255, shape=(16, 192, 192), dtype=np.uint8),
            z_onehot=gym.spaces.Box(low=0, high=1, shape=(number_z,), dtype=np.int32)
        )

    def step(self, action):
        
        self._raw_obs = None
        obs, reward, done, info = self.env.step(action)
        #self._raw_obs = obs
        # tranfer obs to fit observation space
        '''
        if 'camera' in obs:
            array = utils.preprocess_img(obs['camera'])
            self._obs['camera'] = array
        if 'rendered' in obs:
            img = obs['rendered']
            self._obs['bev'] = img
        '''
        obs_dict = self.preprocess_obs(obs, self.number_z)
        self._raw_obs = {
            'bev_render': obs['bev_render'],
            'info': info,
            'state': obs_dict['state']
        }
        self._raw_obs['info'] = info
        return obs_dict, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        obs_dict = self.preprocess_obs(obs, self.number_z)
        '''
        print('set sync')
        self.env.set_synchronous_mode(sync_mode=True, fps=10)
        print('set no rendering')
        self.env.set_no_rendering_mode(rendering_mode=True)
        '''
        return obs_dict

    def render(self, mode="rgb_array"):
        #self.env.render()
        # in callback has 'set attr(action_value)'
        self._raw_obs['action'] = self.action
        self._raw_obs['action_value'] = self.action_value
        self._raw_obs['action_log_probs'] = self.action_log_probs
        self._raw_obs['action_mu'] = self.action_mu
        self._raw_obs['action_sigma'] = self.action_sigma
        self._raw_obs['z'] = self.z
        im = self.im_render(self._raw_obs)
        #im = self._raw_obs['bev_render']
        return im

    def close(self):
        self.env.close()

    @staticmethod
    def im_render(obs):
        im_birdview = obs['bev_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview
        
        #action = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        action_str = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        mu_str = np.array2string(obs['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(obs['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(obs['state'], precision=2, separator=',', suppress_small=True)
        z_str = str(obs['z'])
        txt_t = f'step:{obs["info"]["timestamp"]["step"]:5}, frame:{obs["info"]["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{obs["action_value"]:5.2f} p:{obs["action_log_probs"]:5.2f}'
        im = cv2.putText(im, txt_1, (w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mu_str} b{sigma_str} z{z_str}'
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i, txt in enumerate(obs['reward_debug']['debug_texts'] +
                                obs['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i_txt, txt in enumerate(obs['info']['text']):
                im = cv2.putText(im, txt, (w, 36+(i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return im

    @staticmethod
    def preprocess_obs(obs, number_z, train=True):
        state_list = []
        obs_dict = {}
        if 'bev_mask' in obs:
            bev = obs['bev_mask']

        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        if 'control' in obs:
            state_list.append(obs['control'].throttle)
            state_list.append(obs['control'].steer)
            state_list.append(obs['control'].brake)
            state_list.append(obs['control'].gear / 5.0)
        if 'velocity' in obs:
            vel_x = obs['velocity'].x
            vel_y = obs['velocity'].y
            state_list.append(vel_x)
            state_list.append(vel_y)
        state = np.array(state_list)
        
        if not train:
            bev = np.expand_dims(bev, 0)
            state = np.expand_dims(state, 0)

        obs_dict['birdview'] = bev
        obs_dict['state'] = state.astype(np.float32)
        obs_dict['z_onehot'] = np.zeros((number_z,))
        obs_dict['z_onehot'][0] = 1
        return obs_dict
    
    def preprocess_statespace(self):
        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        state_space = []
        state_space.append(self._observation_space['throttle'])
        state_space.append(self._observation_space['steer'])
        state_space.append(self._observation_space['brake'])
        state_space.append(self._observation_space['gear'])
        state_space.append(self._observation_space['vel_xy'])
        state_low = np.concatenate([s.low for s in state_space])
        state_high = np.concatenate([s.high for s in state_space])
        return state_low, state_high

class RLdiayn_egov_Wrapper(gym.Wrapper):
    def __init__(self, env: CarlaEnv_mzq, number_z=50, number_timestep=1):
        # super(RLBEVWrapper, self).__init__(env)  this code is the same as the following
        self.env = env
        self.number_z = number_z
        self.number_timestep = number_timestep
        self.action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        
        self._obs = {}
        state_low, state_high = self.preprocess_statespace()
        # change observate space
        # may auto add batch dim, i dont know
        # env_vec will auto add batch dim
        # not batch dim, may be the n_env dim
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
            birdview=gym.spaces.Box(low=0, high=255, shape=(number_timestep,192, 192), dtype=np.uint8),
            z_onehot=gym.spaces.Box(low=0, high=1, shape=(number_z,), dtype=np.int32)
        )

    def step(self, action):
        self._raw_obs = None
        obs, reward, done, info = self.env.step(action)
        # self._raw_obs = obs
        # tranfer obs to fit observation space
        obs_dict = self.preprocess_obs(obs, self.number_z)
        self._raw_obs = {
            'bev_render': obs['bev_render'],
            'info': info,
            'state': obs_dict['state'],
        }
        self._raw_obs['info'] = info
        return obs_dict, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        obs_dict = self.preprocess_obs(obs, self.number_z)
        '''
        print('set sync')
        self.env.set_synchronous_mode(sync_mode=True, fps=10)
        print('set no rendering')
        self.env.set_no_rendering_mode(rendering_mode=True)
        '''
        return obs_dict

    def render(self, mode="rgb_array"):
        # self.env.render()
        # in callback has 'set attr(action_value)'
        self._raw_obs['action'] = self.action
        self._raw_obs['action_value'] = self.action_value
        self._raw_obs['action_log_probs'] = self.action_log_probs
        self._raw_obs['action_mu'] = self.action_mu
        self._raw_obs['action_sigma'] = self.action_sigma
        self._raw_obs['z'] = self.z
        im = self.im_render(self._raw_obs)
        # im = self._raw_obs['bev_render']
        return im

    def close(self):
        self.env.close()

    @staticmethod
    def im_render(obs):
        im_birdview = obs['bev_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview
        
        # action = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        action_str = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        mu_str = np.array2string(obs['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(obs['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(obs['state'], precision=2, separator=',', suppress_small=True)
        z_str = str(obs['z'])
        txt_t = f'step:{obs["info"]["timestamp"]["step"]:5}, frame:{obs["info"]["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{obs["action_value"]:5.2f} p:{obs["action_log_probs"]:5.2f}'
        im = cv2.putText(im, txt_1, (w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mu_str} b{sigma_str} z{z_str}'
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i, txt in enumerate(obs['reward_debug']['debug_texts'] +
                                obs['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i_txt, txt in enumerate(obs['info']['text']):
                im = cv2.putText(im, txt, (w, 36+(i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return im

    
    def preprocess_obs(self, obs, number_z, train=True, only_ego=True):
        state_list = []
        obs_dict = {}
        bev_only = None
        if 'bev_mask' in obs:
            bev = obs['bev_mask']
        if 'bev_ev_history_mask' in obs:
            bev_only = obs['bev_ev_history_mask'][-self.number_timestep:]
        #if only_ego:
        #    bev_only = bev[3]   # ego vehicle mask
        #    bev_only = np.expand_dims(bev_only, 0)

        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        if 'control' in obs:
            state_list.append(obs['control'].throttle)
            state_list.append(obs['control'].steer)
            state_list.append(obs['control'].brake)
            state_list.append(obs['control'].gear / 5.0)
        if 'velocity' in obs:
            vel_x = obs['velocity'].x
            vel_y = obs['velocity'].y
            state_list.append(vel_x)
            state_list.append(vel_y)
        state = np.array(state_list)
        
        if not train:
            bev = np.expand_dims(bev, 0)
            state = np.expand_dims(state, 0)

        if bev_only is not None:
            obs_dict['birdview'] = bev_only
        else:
            obs_dict['birdview'] = bev
        obs_dict['state'] = state.astype(np.float32)
        obs_dict['z_onehot'] = np.zeros((number_z,))
        obs_dict['z_onehot'][0] = 1
        return obs_dict
    
    def preprocess_statespace(self):
        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        state_space = []
        state_space.append(self._observation_space['throttle'])
        state_space.append(self._observation_space['steer'])
        state_space.append(self._observation_space['brake'])
        state_space.append(self._observation_space['gear'])
        state_space.append(self._observation_space['vel_xy'])
        state_low = np.concatenate([s.low for s in state_space])
        state_high = np.concatenate([s.high for s in state_space])
        return state_low, state_high

class RLdiayn_egov_transformer_Wrapper(gym.Wrapper):
    def __init__(self, env: CarlaEnv_mzq, number_z=50, number_timestep=1):
        # super(RLBEVWrapper, self).__init__(env)  this code is the same as the following
        self.env = env
        self.number_z = number_z
        self.number_timestep = number_timestep
        self.action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self._width = 192
        self._obs = {}
        state_low, state_high = self.preprocess_statespace()
        # change observate space
        # may auto add batch dim, i dont know
        # env_vec will auto add batch dim
        # not batch dim, may be the n_env dim
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
            birdview=gym.spaces.Box(low=0, high=255, shape=(number_timestep, 192, 192), dtype=np.uint8),
            z=gym.spaces.Box(low=-1, high=number_z, shape=(number_timestep, ), dtype=np.int32)
        )

    def step(self, action):
        self._raw_obs = None
        obs, reward, done, info = self.env.step(action)
        # self._raw_obs = obs
        # tranfer obs to fit observation space
        obs_dict = self.preprocess_obs(obs=obs, number_z=self.number_z, number_timestep=self.number_timestep)
        tmp_img = np.zeros((192, 192, 3), dtype=np.uint8)
        tmp_img[:,:,0] = obs_dict['birdview'][0]
        tmp_img[:,:,1] = obs_dict['birdview'][0]
        tmp_img[:,:,2] = obs_dict['birdview'][0]
        self._raw_obs = {
            'bev_render': obs['bev_render'],
            'bev_ego': tmp_img,
            'info': info,
            'state': obs_dict['state'],
        }
        self._raw_obs['info'] = info
        return obs_dict, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        obs_dict = self.preprocess_obs(obs=obs, number_z=self.number_z, number_timestep=self.number_timestep)
     
        return obs_dict

    def render(self, mode="rgb_array"):
        # self.env.render()
        # in callback has 'set attr(action_value)'
        self._raw_obs['action'] = self.action
        self._raw_obs['action_value'] = self.action_value
        self._raw_obs['action_log_probs'] = self.action_log_probs
        self._raw_obs['action_mu'] = self.action_mu
        self._raw_obs['action_sigma'] = self.action_sigma
        
        self._raw_obs['z'] = self.z
        im = self.im_render(self._raw_obs)
        # im = self._raw_obs['bev_render']
        return im

    def close(self):
        self.env.close()

    @staticmethod
    def im_render(obs):
        im_birdview = obs['bev_ego']
        # im_birdview = obs['bev_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        

        im[:h, :w] = im_birdview
        
        # action = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        action_str = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        mu_str = np.array2string(obs['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(obs['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(obs['state'], precision=2, separator=',', suppress_small=True)
        z_str = str(obs['z'])
        txt_t = f'step:{obs["info"]["timestamp"]["step"]:5}, frame:{obs["info"]["timestamp"]["frame"]:5}'
        # im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{obs["action_value"]:5.2f} p:{obs["action_log_probs"]:5.2f}'
        # im = cv2.putText(im, txt_1, (w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        im = cv2.putText(im, txt_1, (w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        # im = cv2.putText(im, txt_2, (w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        im = cv2.putText(im, txt_2, (w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mu_str} b{sigma_str} z{z_str}'
        # im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i, txt in enumerate(obs['reward_debug']['debug_texts'] +
                                obs['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        '''
        for i_txt, txt in enumerate(obs['info']['text']):
                im = cv2.putText(im, txt, (w, 36+(i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        

    
    def preprocess_obs(self, obs, number_z=20, number_timestep=5, train=True):
        state_list = []
        obs_dict = {}
        
        if 'bev_ev_history_mask' in obs:
            bev_only = obs['bev_ev_history_mask'][-self.number_timestep:] # 'bev_only', dtype: bool
            image = np.zeros(bev_only.shape, dtype=np.uint8)                
            road_mask = obs['bev_mask'][1]                                # road mask, dtype: int64  
            road_mask = np.array(road_mask, dtype=bool)                
            for i in range(self.number_timestep):
                image[i][road_mask] = 128 
                image[i][bev_only[i]] = 255
            
        # state order: control / velocity -> throttle, steer, brake, gear / vel_x, vel_y
        if 'control' in obs:
            state_list.append(obs['control'].throttle)
            state_list.append(obs['control'].steer)
            state_list.append(obs['control'].brake)
            state_list.append(obs['control'].gear / 5.0)
        if 'velocity' in obs:
            vel_x = obs['velocity'].x
            vel_y = obs['velocity'].y
            state_list.append(vel_x)
            state_list.append(vel_y)
        state = np.array(state_list)
        
        # other operations
        if not train:
            state = np.expand_dims(state, 0)

        obs_dict['birdview'] = image
        # obs_dict['bev_ego'] = image[0].reshape(1, 192, 192)
        obs_dict['state'] = state.astype(np.float32)
        obs_dict['z'] = np.zeros((self.number_timestep, )) - 1
        obs_dict['z'][0] = 0
        return obs_dict
    
    def preprocess_statespace(self):
        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        state_space = []
        state_space.append(self._observation_space['throttle'])
        state_space.append(self._observation_space['steer'])
        state_space.append(self._observation_space['brake'])
        state_space.append(self._observation_space['gear'])
        state_space.append(self._observation_space['vel_xy'])
        state_low = np.concatenate([s.low for s in state_space])
        state_high = np.concatenate([s.high for s in state_space])
        return state_low, state_high

class RL_meta_Wrapper(gym.Wrapper):
    def __init__(self, env: CarlaEnv_mzq, number_z=50, number_timestep=1):
        # super(RLBEVWrapper, self).__init__(env)  this code is the same as the following
        self.env = env
        self.number_z = number_z
        self.number_timestep = number_timestep
        self.action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        
        self._obs = {}
        state_low, state_high = self.preprocess_statespace()
        # change observate space
        # may auto add batch dim, i dont know
        # env_vec will auto add batch dim
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
            total_birdview=gym.spaces.Box(low=0, high=255, shape=(16,192, 192), dtype=np.uint8),
            z=gym.spaces.Box(low=-1, high=number_z, shape=(number_timestep,), dtype=np.int32),
            birdview=gym.spaces.Box(low=0, high=255, shape=(number_timestep,192, 192), dtype=np.uint8),
        )

    def step(self, action):
        self._raw_obs = None
        obs, reward, done, info = self.env.step(action)
        # self._raw_obs = obs
        # tranfer obs to fit observation space
        obs_dict = self.preprocess_obs(obs, self.number_z)
        self._raw_obs = {
            'bev_render': obs['bev_render'],
            'info': info,
            'state': obs_dict['state'],
        }
        self._raw_obs['info'] = info
        return obs_dict, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        obs_dict = self.preprocess_obs(obs, self.number_z)
        '''
        print('set sync')
        self.env.set_synchronous_mode(sync_mode=True, fps=10)
        print('set no rendering')
        self.env.set_no_rendering_mode(rendering_mode=True)
        '''
        return obs_dict

    def render(self, mode="rgb_array"):
        # self.env.render()
        # in callback has 'set attr(action_value)'
        self._raw_obs['action'] = self.action
        self._raw_obs['action_value'] = self.action_value
        self._raw_obs['z'] = self.z
        self._raw_obs['q_values']=self.q_values
        self._raw_obs['ter'] = self.ter
        im = self.im_render(self._raw_obs)
        # im = self._raw_obs['bev_render']
        return im

    def close(self):
        self.env.close()

    @staticmethod
    def im_render(obs):
        im_birdview = obs['bev_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*3, c], dtype=np.uint8)
        #im = np.zeros([h, w*2, c], dtype=np.uint8)

        im[:h, :w] = im_birdview
        
        # action = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        action_str = np.array2string(obs['action'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(obs['state'], precision=2, separator=',', suppress_small=True)
        z_str = str(obs['z'])
        ter_str = str(obs['ter'])
        a_v_str = str(obs["action_value"])
        txt_t = f'step:{obs["info"]["timestamp"]["step"]:5}, frame:{obs["info"]["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} act_v:{a_v_str} '
        im = cv2.putText(im, txt_1, (w, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (w, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_3 = f'z{z_str} ter{ter_str}'
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for i_txt, txt in enumerate(obs['info']['text']):
                im = cv2.putText(im, txt, (w, 36+(i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # write q_values
        for l in range(len(obs['q_values'])//3):
            i = 3 * l
            tmp_txt = f'{i}: {obs["q_values"][i]:3.2f}  {i+1}: {obs["q_values"][i+1]:3.2f}  {i+2}: {obs["q_values"][i+2]:3.2f}'
            im = cv2.putText(im, tmp_txt, (2*w, 12+l*17), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        tmp_txt = f''
        for j in range(len(obs['q_values'])%3):
            tmp_txt = tmp_txt + f'{i+3+j}: {obs["q_values"][i+3+j]:3.2f}  '
        im = cv2.putText(im, tmp_txt, (2*w, 12+(l+1)*17), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
        return im

    def preprocess_obs(self, obs, number_z, only_ego=True):
        ''' ego
        state_list = []
        obs_dict = {}
        if 'bev_mask' in obs:
            bev = obs['bev_mask']
        if 'bev_ev_history_mask' in obs:
            bev_only = obs['bev_ev_history_mask'][-self.number_timestep:]

        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        if 'control' in obs:
            state_list.append(obs['control'].throttle)
            state_list.append(obs['control'].steer)
            state_list.append(obs['control'].brake)
            state_list.append(obs['control'].gear / 5.0)
        if 'velocity' in obs:
            vel_x = obs['velocity'].x
            vel_y = obs['velocity'].y
            state_list.append(vel_x)
            state_list.append(vel_y)
        state = np.array(state_list)

        obs_dict['birdview'] = bev_only
        obs_dict['total_birdview'] = bev
        obs_dict['state'] = state.astype(np.float32)
        obs_dict['z_onehot'] = np.zeros((number_z,))
        obs_dict['z_onehot'][0] = 1
        '''
        state_list = []
        obs_dict = {}
        if 'bev_mask' in obs:
            bev = obs['bev_mask'] 
        if 'bev_ev_history_mask' in obs:
            bev_only = obs['bev_ev_history_mask'][-self.number_timestep:] # 'bev_only', dtype: bool
            image = np.zeros(bev_only.shape, dtype=np.uint8)                
            road_mask = obs['bev_mask'][1]                                # road mask, dtype: int64  
            road_mask = np.array(road_mask, dtype=bool)                
            for i in range(self.number_timestep):
                image[i][road_mask] = 128 
                image[i][bev_only[i]] = 255
            
        # state order: control / velocity -> throttle, steer, brake, gear / vel_x, vel_y
        if 'control' in obs:
            state_list.append(obs['control'].throttle)
            state_list.append(obs['control'].steer)
            state_list.append(obs['control'].brake)
            state_list.append(obs['control'].gear / 5.0)
        if 'velocity' in obs:
            vel_x = obs['velocity'].x
            vel_y = obs['velocity'].y
            state_list.append(vel_x)
            state_list.append(vel_y)
        state = np.array(state_list)

        obs_dict['birdview'] = image
        obs_dict['total_birdview'] = bev
        obs_dict['state'] = state.astype(np.float32)
        obs_dict['z'] = np.zeros((self.number_timestep, )) - 1
        obs_dict['z'][0] = 0
        return obs_dict
    
    def preprocess_statespace(self):
        # state order: control,vel -> throttle, steer, brake, gear, vel_x, vel_y
        state_space = []
        state_space.append(self._observation_space['throttle'])
        state_space.append(self._observation_space['steer'])
        state_space.append(self._observation_space['brake'])
        state_space.append(self._observation_space['gear'])
        state_space.append(self._observation_space['vel_xy'])
        state_low = np.concatenate([s.low for s in state_space])
        state_high = np.concatenate([s.high for s in state_space])
        return state_low, state_high