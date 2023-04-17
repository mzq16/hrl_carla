import gym
import numpy as np
import torchaudio
from zmq import device
from SAC.diayn import my_diayn
import carla
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from gym_carla.rlbev_wrapper import RLdiaynWrapper
import gym
from SAC.sac_policy import my_SACPolicy
import torch
from stable_baselines3.sac.policies import SACPolicy
import os
import time
import server_utils
import subprocess
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from tqdm import tqdm
from stable_baselines3.common.callbacks import CallbackList
from gym_carla.wandb_callback_diayn import WandbCallback


def propre_envconfig(num_env=2):
    env_configs = []
    for i in range(num_env):
        port = 2000 + i * 5
        env_cfg = {'gpu':0, 'port':port}
        env_configs.append(env_cfg)
    return env_configs

if __name__ == '__main__':
    cfg = {}
    cfg['carla_sh_path'] = '/opt/carla-simulator/CarlaUE4.sh'
    env_configs = propre_envconfig(num_env=2)
    cfg['env_configs'] = env_configs
    try:
        server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
        server_manager.start()
    except:
        server_manager.stop()