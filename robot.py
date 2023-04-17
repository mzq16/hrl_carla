import logging
import numpy as np
import os
from agent.ppo import PPO
from agent.ppo_policy import PpoPolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
import server_utils
import torch
from tqdm import tqdm

def start_server(cfg):
    server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
    server_manager.start()
    a=input()
    server_manager.stop()

class Robot(object):
    def __init__(self, env: VecEnv, ppo_args=None, ppopolicy_args=None, device=torch.device(1)):
        '''
        robot use ppo model, init ppo model need ppo policy and env, and some args,     
                    ppo model contains -> 'learn' -> 'data rollout', 'train'  
        ppo policy, 

        '''
        self._logger = logging.getLogger(__name__)
        self._env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._ppo = None
        self._ppo_args = ppo_args
        self._ppopolicy_args = ppopolicy_args
        self._ckpt = None
        self._start_timesteps = 0
        self._device = device

    def build_ppo(self):
        # check ckpt
        self.check_ckpt()

        # prepare ppo policy
        if self._ckpt is None:
            self._policy = PpoPolicy(self._env.observation_space, self._env.action_space, **self._ppopolicy_args, device=self._device)
        else:
            self._policy = PpoPolicy(self._env.observation_space, self._env.action_space, **self._ppopolicy_args, device=self._device)
            self._policy, self._train_cfg = self._policy.load(self._ckpt, device=self._device)
        
        # prepare ppo by using env and ppo policy
        if self._ppo_args is not None:
            self._ppo = PPO(self._policy, self._env, **self._ppo_args, start_num_timesteps=self._start_timesteps)
        else:
            self._ppo = PPO(self._policy, self._env, start_num_timesteps=self._start_timesteps)

    def learn(self, total_timesteps, callback, seed):
        self._ppo.learn(total_timesteps=total_timesteps, callback=callback, seed=seed)

    def check_ckpt(self, dic_path='ppo_data/ckpt'):
        ckpt_list = os.listdir(dic_path)
        if ckpt_list:
            ckpt_list[0].split('_')[1].split('.')[0]
            max_ckpt = max(ckpt_list, key = lambda x: int(x.split('_')[1].split('.')[0]))
            self._ckpt = os.path.join(dic_path,max_ckpt)
            self._start_timesteps = int(max_ckpt.split('_')[1].split('.')[0])
            self._logger.info('resume checkpoint latest ' + max_ckpt)
        else:
            self._logger.info('no exit ckpt, start a new train')

    
            
        

    