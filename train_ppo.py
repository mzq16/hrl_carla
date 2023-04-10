import gym
import hydra
import numpy as np
from pyparsing import null_debug_action
import server_utils
import argparse
import logging
from gym_carla.rlbev_wrapper import PPORLBEVWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from gym_carla.wandb.wandb_callback_ppo import WandbCallback
from robot import Robot
import time
import os
import subprocess
import torch
from tqdm import tqdm
from gym_carla.envs.carla_env_mzq import CarlaEnv_mzq
from gym.wrappers.monitoring.video_recorder import ImageEncoder
import pickle
from agent.ppo_policy import PpoPolicy
import warnings
warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.INFO)
log = logging


def start_server(cfg):
    server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
    server_manager.start()
    a=input()
    server_manager.stop()

#@hydra.main(config_path='test_config', config_name='test')
def main(cfg):
    #hydra.initialize(config_path='test_config')
    #test_cfg = hydra.compose(config_name='test')
    try:
        # start server 
        server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
        server_manager.start()
        time.sleep(15)
        device = torch.device(2)
        # env_list = [env_base, env_base2]
        #port_list = []
        #for i in range(len(cfg['env_configs'])):
            #port_list.append(2000 + i * 5)
        env_cfg1 = {
            'port':2000,
            'map_id':2,
            'device':device,
        }
        env_cfg2 = {
            'port':2005,
            'map_id':3,
            'device':device,
        }
        env_cfg = [env_cfg1, env_cfg2]
        #env_cfg = [env_cfg1]
        if len(env_cfg) == 1:
            env_vec = DummyVecEnv([lambda cfg=cfg: env_make(**cfg) for cfg in env_cfg])
        else:
            env_vec = SubprocVecEnv([lambda cfg=cfg: env_make(**cfg) for cfg in env_cfg])

        # init wandb
        wandb_cfg = {
            'wb_project': 'test',
            'wb_name': None,
            'wb_notes': None,
            'wb_tags': None
        }
        wb_callback = WandbCallback(wandb_cfg, env_vec)
        callback = CallbackList([wb_callback])
        
        # init robot
        ppo_args = None
        ppopolicy_args = {'policy_head_arch': [256, 256], 'value_head_arch': [256, 256], 
            'features_extractor_entry_point': 'agent.torch_layers:XtMaCNN', 
            'features_extractor_kwargs': {'states_neurons': [256, 256]}, 
            'distribution_entry_point': 'agent.distributions:BetaDistribution', 'distribution_kwargs': {'dist_init': None}}
        robot = Robot(env_vec, None, ppopolicy_args, device=device)
        robot.build_ppo()
        total_timesteps = 10000000
        robot.learn(total_timesteps, callback=callback, seed=2022)

    except:
        pid = os.getpid()
        subprocess.Popen('kill {}'.format(pid), shell=True)

    finally:
        env_vec.close()
        server_manager.stop()
        
        #exit(0)
        
def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        out_put = {}
        avg_eval_stat = {}
        count = 0
        for eval_info in ep_stat_buffer:
            epoch = f'epoch_{count}'
            out_put[epoch] = eval_info
            count += 1
            for k, v in eval_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_eval_stat:
                    avg_eval_stat[k_avg] += v
                else:
                    avg_eval_stat[k_avg] = v
        out_put['avg_eval'] = avg_eval_stat
        return out_put

def evaluate_policy(env:CarlaEnv_mzq, policy:PpoPolicy, video_folder_path, info_folder_path, min_eval_steps=100, is_use_wandb=False):
    # get the maximum
    max_i = 0
    for filename in os.listdir(video_folder_path):
        if filename.endswith('.mp4'):
            i  = int(filename.split('_')[-1].split('.')[0])
            max_i = max(max_i, i)
    video_name = f'{max_i + 1}' + '.mp4'
    video_path = os.path.join(video_folder_path, video_name)

    for filename in os.listdir(info_folder_path):
        if filename.endswith('.pkl'):
            i  = int(filename.split('_')[-1].split('.')[0])
            max_i = max(max_i, i)
    info_name = f'{max_i + 1}' + '.pkl'
    info_path = os.path.join(info_folder_path, info_name)

    
    policy = policy.eval()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()
    list_render = []
    
    n_step = 0
    env_done = np.array([False]*env.num_envs)
    if is_use_wandb:
        pbar = tqdm(initial=0)
    else:
        pbar = tqdm(initial=0, total=min_eval_steps)
    
    ep_stat_buffer = []
    while len(ep_stat_buffer) < 4:
        actions, values, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
        
        obs, reward, done, info = env.step(actions)
        for i in range(env.num_envs):
            env.set_attr('action', actions[i], indices=i)
            env.set_attr('action_value', values[i], indices=i)
            env.set_attr('action_log_probs', log_probs[i], indices=i)
            env.set_attr('action_mu', mu[i], indices=i)
            env.set_attr('action_sigma', sigma[i], indices=i)

            
        list_render.append(env.render('rgb_array'))

        n_step += 1
        env_done |= done
        for i in np.where(done)[0]:
            ep_stat_buffer.append(info[i]['eval_info'])
        pbar.update(1)
    pbar.close()
    encoder = ImageEncoder(video_path, list_render[0].shape, 20, 20)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

    eval_info = get_avg_ep_stat(ep_stat_buffer)
    with open(info_path,'wb') as f:
        pickle.dump(eval_info, f)
    #if is_use_wandb:
    #    wandb.init(project='sac_eval', name=None, notes=None, tags=None)

def eval(cfg):
    server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
    server_manager.start()
    time.sleep(15)
    device = torch.device(2)
    env_cfg1 = {
        'port':2010,
        'map_id':4,
        'device':device,
        }
    #env_cfg2 = {
    #    'port':2015,
    #    'map_id':5,
    #    'device':device,
    #    }
    env_cfg = [env_cfg1]
    if len(env_cfg) == 1:
        env_vec = DummyVecEnv([lambda cfg=cfg: env_make(**cfg) for cfg in env_cfg])
    else:
        env_vec = SubprocVecEnv([lambda cfg=cfg: env_make(**cfg) for cfg in env_cfg])

    ppopolicy_args = {'policy_head_arch': [256, 256], 'value_head_arch': [256, 256], 
        'features_extractor_entry_point': 'agent.torch_layers:XtMaCNN', 
        'features_extractor_kwargs': {'states_neurons': [256, 256]}, 
        'distribution_entry_point': 'agent.distributions:BetaDistribution', 'distribution_kwargs': {'dist_init': None}}
    robot = Robot(env_vec, None, ppopolicy_args, device=device)
    robot.build_ppo()
    evaluate_policy(env_vec, robot._policy, 'ppo_data/eval_video/ppo', 'ppo_data/info/ppo', is_use_wandb=True)



def env_make(port, map_id=0, device=torch.device(0)):
    map_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town10HD_Opt']
    map_name = map_list[map_id]
    h5file_path = map_name + '.h5'
    host = 'localhost'
    env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022, carla_map=map_name, 
            file_path=h5file_path, num_vehicles=100, num_walkers=100, use_bev_fusion=False, device=device)
    #env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022, carla_map=map_name, file_path=h5file_path)
    #env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022)
    env = PPORLBEVWrapper(env_base)
    return env


def propre_envconfig(num_env=2):
    env_configs = []
    for i in range(num_env):
        port = 2010 + i * 5
        env_cfg = {'gpu':0, 'port':port}
        env_configs.append(env_cfg)
    return env_configs

if __name__ == '__main__':
    cfg = {}
    cfg['carla_sh_path'] = '/home/ubuntu/carla-simulator/CarlaUE4.sh'
    env_configs = propre_envconfig(1)
    cfg['env_configs'] = env_configs
    #start_server(cfg)
    #main(cfg)
    eval(cfg)