from curses import KEY_LEFT
from pickletools import int4
from turtle import forward
import gym
import numpy as np
import torchaudio
from zmq import device
import gym
import carla
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from gym_carla.wandb import wandb_callback_diayn
from gym_carla.rlbev_wrapper import RLdiaynWrapper, RLdiayn_egov_Wrapper
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from gym_carla.wandb.wandb_callback_diayn import WandbCallback
from gym_carla.wandb.wandb_callback_meta import WandbCallback as meta_WandbCallback
from SAC.diayn import my_diayn
from SAC.sac_policy import my_SACPolicy
from SAC.diayn_policy import my_diaynPolicy
import torch
import os
import time
import server_utils
import subprocess
from tqdm import tqdm
import argparse
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import DictReplayBufferSamples



class Robot(object):
    def __init__(self, env: VecEnv, sac_args: dict , base_ckpt_path: str, base_buffer_path: str, number_z: int = 50, train_bool: bool = True):
        self.env = env
        #self.args = args
        self._dir_ckpt = None
        self._number_z = number_z
        self.diayn = my_diayn(
                policy="MultiInputPolicy", 
                env=env, 
                policy_base=SACPolicy, 
                #number_z=number_z,
                **sac_args
        )
        self.check_ckpt(dir_path=base_ckpt_path)
        self.check_buffer(dir_path=base_buffer_path)  

        if (self._dir_ckpt is None) and (not train_bool):
            print('using diayn for meta, but got no ckpt')
            
        if self._dir_ckpt is not None:
            self.diayn = self.diayn.load(path=self._dir_ckpt, env=env, buffer_path=self._dir_buffer,train_bool=train_bool, **sac_args)
        
        if self.diayn._init_setup_model:    
            if self._dir_buffer:
                self.diayn.load_replay_buffer(self._dir_buffer)     
            self.diayn._setup_model()
        # print(self.diayn.policy)

    def learn(self, callback):
        self.diayn._setup_learn(callback=callback, start_timestamp=self._start_timesteps)
        self.diayn.learn(
            total_timesteps=1500000,
            callback=callback,
            number_z=self._number_z,
        )

    def learn_from_meta(self, replay_buffer: DictReplayBuffer, callback: WandbCallback, gredient_step: int, batch_size: int):
        # rollout from meta, only need to train 
        # consider replay buffer 
        # print("learn finetune") 
        assert isinstance(replay_buffer, ReplayBuffer), 'has an unknown buffer'
        #self.diayn.replay_buffer = replay_buffer
        for i in range(gredient_step):
            replay_data = replay_buffer.sample(batch_size=batch_size, env=None)
            replay_data = self.clean_meta_data(replay_buffer=replay_buffer, replay_data=replay_data)
            self.diayn.finetune(gradient_steps=1, callback=callback, replay_data=replay_data)

    def clean_meta_data(self, replay_buffer: DictReplayBuffer, replay_data: DictReplayBufferSamples):
        ind_len, ind = self.find_index(replay_data=replay_data, number_z=20)
        while ind_len > 0:
            tmp_data = replay_buffer.sample(ind_len, env=None)
            tmp_len = 0
            tmp_index = 0
            for i in range(ind_len):
                if tmp_data.actions[i] == self._number_z or tmp_data.actions[i] == self._number_z + 1:
                    tmp_len += 1
                    continue
                replay_data.actions[ind[tmp_index]] = tmp_data.actions[i]
                replay_data.observations['birdview'][ind[tmp_index]] = tmp_data.observations['birdview'][i]
                replay_data.observations['state'][ind[tmp_index]] = tmp_data.observations['state'][i]
                replay_data.observations['z_onehot'][ind[tmp_index]] = tmp_data.observations['z_onehot'][i]
                replay_data.observations['total_birdview'][ind[tmp_index]] = tmp_data.observations['total_birdview'][i]
                tmp_index += 1
            
            ind_len, ind = self.find_index(replay_data=replay_data, number_z=20)
        return replay_data

    def find_index(self, replay_data: DictReplayBufferSamples, number_z:int):
        action = replay_data.actions.squeeze(-1).cpu().numpy()
        b1 = action == number_z
        b2 = action == number_z + 1
        b = b1 + b2
        ind = np.where(b)[0]
        return len(ind), ind
        
    def check_ckpt(self, dir_path='SAC_train_data/diayn_ckpt'):
        ckpt_list = os.listdir(dir_path)
        if ckpt_list:
            ckpt_list[0].split('_')[1].split('.')[0]
            max_ckpt = max(ckpt_list, key = lambda x: int(x.split('_')[1].split('.')[0]))
            self._dir_ckpt = os.path.join(dir_path, max_ckpt)
            self._start_timesteps = int(max_ckpt.split('_')[1].split('.')[0])
            print('resume checkpoint latest ' + max_ckpt)
        else:
            self._start_timesteps = 0
            print('no exist ckpt, start a new train')

    def check_buffer(self, dir_path='SAC_train_data/diayn_buffer'):
        buffer_list = os.listdir(dir_path)
        if len(buffer_list) == 0:
            self._dir_buffer = None
            print('no exist buffer')
        else:
            self._dir_buffer = os.path.join(dir_path, buffer_list[0])
            #self.sac.load_replay_buffer(self._dir_buffer)
    
    def _transfer_env(self, obs: dict, env: VecEnv = None):
        tmp_obs = {}
        target_keys = ['birdview','state','z_onehot']
        for key in obs.keys():
            if key in target_keys:
                tmp_obs[key] = obs[key]
        return tmp_obs

    @staticmethod
    def dict_to_tensor(obs:dict, device):
        return {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in obs.items()}
    
    @staticmethod
    def evaluate_policy(env, policy: my_diaynPolicy, video_path: str, min_eval_steps: int = 1000, tmp_z: int = 0, number_z: int = 50):
        device = torch.device('cuda:1')
        policy = policy.eval()
        
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()
        z_onehot = np.zeros((number_z,))
        
        z_onehot[tmp_z] = 1
        obs['z_onehot'][:] = z_onehot
        o = WandbCallback.dict_to_tensor(obs, device)
        list_render = []
        
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        
        pbar = tqdm(initial=0, total=min_eval_steps)
        while n_step < min_eval_steps:
            mean_actions, log_std, kwargs= policy.actor.get_action_dist_params(o)
            actions, log_probs = policy.actor.action_dist.log_prob_from_params(mean_actions, log_std)
            value, _ = torch.cat(policy.critic(o, actions), dim=1).min(dim=1)
            #value, _ = torch.min(value, dim=1)
            actions = np.array(actions.detach().cpu())
            log_probs = np.array(log_probs.detach().cpu())
            mean_actions = np.array(mean_actions.detach().cpu())
            log_std = np.array(log_std.detach().cpu().exp())
            values = np.array(value.detach().cpu()).reshape(-1,)
            obs, reward, done, info = env.step(actions)
            obs['z_onehot'][:] = z_onehot
            o = WandbCallback.dict_to_tensor(obs, device)

            for i in range(env.num_envs):
                env.set_attr('action', actions[i], indices=i)
                env.set_attr('action_value', values[i], indices=i)
                env.set_attr('action_log_probs', log_probs[i], indices=i)
                env.set_attr('action_mu', mean_actions[i], indices=i)
                env.set_attr('action_sigma', log_std[i], indices=i)
                env.set_attr('z', tmp_z, indices=i)
                
            list_render.append(env.render('rgb_array'))
            n_step += 1
            pbar.update(1)
        pbar.close()

        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()
    
    def forward(self, observation: dict, skill: int = 0, deterministic: bool = True, device: Union[str, int] = "auto"):
        if isinstance(device, str):
            if device == "auto":
                device = torch.device("cuda:1")
        if isinstance(device, int):
            device = torch.device("cuda:{}".format(device))

        z_onehot = np.zeros((self._number_z,))
        z_onehot[skill] = 1

        # for actor there are two kinds of sample 
        # the following one, we cannot chose determination or not, 
        # but the function will use the default value: False (use distribution rsample) 
        # actions_pi, log_prob = self.diayn.policy.actor.action_log_prob(obeservation)
        
        # the other one is use 
        # self.diayn.policy.actor.forward or actor()
        
        observation['z_onehot'][:] = z_onehot

        if not isinstance(list(observation.keys())[0], torch.Tensor):
            observation = WandbCallback.dict_to_tensor(observation, device)

        action = self.diayn.policy.actor.forward(observation, deterministic)
        
        return action

def get_ego_obs(self, obs: dict):
    ego_obs = {}
    keys_list = list(obs.keys()) 
    assert 'ego_history' in keys_list, 'wrong'
    for key in obs.keys():
        if key == 'birdview':
            pass
        else:
            ego_obs[key] = obs[key]

    return ego_obs

def propre_envconfig(num_env=2):
    env_configs = []
    for i in range(num_env):
        port = 2000 + i * 5
        env_cfg = {'gpu':0, 'port':port}
        env_configs.append(env_cfg)
    return env_configs

def env_make(port, map_id=0, number_z=5):
    map_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town10HD_Opt']
    map_name = map_list[map_id]
    h5file_path = map_name + '.h5'
    host = 'localhost'
    env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022, 
        carla_map=map_name, file_path=h5file_path, num_vehicles=0, num_walkers=0)
    # env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022)
    # env = RLdiaynWrapper(env=env_base, number_z=number_z)
    env = RLdiayn_egov_Wrapper(env=env_base, number_z=number_z)
    return env

def main():
    argparser = argparse.ArgumentParser(description='CARLA Automatic Control Data Collection')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--carla_sh_path',
        default='/home/ubuntu/carla-simulator/CarlaUE4.sh',
        help='path of carla server')
    argparser.add_argument(
        '--base_ckpt_path',
        default='SAC_train_data/diayn_ckpt',
        help='path of ckpt dir')
    argparser.add_argument(
        '--base_buffer_path',
        default='SAC_train_data/diayn_buffer',
        help='path of buffer dir')
    argparser.add_argument(
        '--number_z',
        default=5,
        help='number of skill')
    args = argparser.parse_args()
    
    server_env_configs = propre_envconfig(num_env=2)
    
    try:
        server_manager = server_utils.CarlaServerManager(args.carla_sh_path, configs=server_env_configs)
        server_manager.start()
        time.sleep(15)

        env_cfg1 = {
            'port':2000,
            'map_id':0,
            'number_z':args.number_z,
        }
        env_cfg2 = {
            'port':2005,
            'map_id':1,
            'number_z':args.number_z,
        }
        env_cfg = [env_cfg1, env_cfg2]
        if len(env_cfg) == 1:
            env = DummyVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        else:
            env = SubprocVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        
        wandb_cfg = {
            'wb_project': 'diayn_5skill_road&ego_school_iter',
            'wb_name': None,
            'wb_notes': None,
            'wb_tags': None
        }
        wb_callback = WandbCallback(wandb_cfg, env)
        callback = CallbackList([wb_callback])
        policy_args = {
            'features_extractor_entry_point': 'SAC.torch_layers:diaynCNN', 
            'features_extractor_kwargs': {'states_neurons': [256, 256], 
                                            'number_z': args.number_z}, 
            'use_sde': False,
            }
        sac_args = {
            'learning_rate': 1e-5,
            'buffer_size': 10000,
            'train_freq': (10,'step'),
            'gradient_steps': 10,
            'target_update_interval': 1,
            'learning_starts': 100,
            'batch_size': 128,
            'ent_coef': "auto",                    # defualt "auto"
            'add_action': True,
            'policy_kwargs': policy_args,
            'number_z': args.number_z,
            'life_time': 1e5,
        }
        robot = Robot(
            env, 
            number_z=args.number_z, 
            sac_args=sac_args, 
            base_ckpt_path=args.base_ckpt_path, 
            base_buffer_path=args.base_buffer_path)
        robot.learn(callback=callback)

    except:
        pid = os.getpid()
        subprocess.Popen('kill {}'.format(pid), shell=True)

    finally:
        env.close()
        server_manager.stop()
        
def eval(z=None):
    argparser = argparse.ArgumentParser(description='CARLA Automatic Control Data Collection')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--carla_sh_path',
        default='/home/ubuntu/carla-simulator/CarlaUE4.sh',
        help='path of carla server')
    argparser.add_argument(
        '--base_ckpt_path',
        default='SAC_train_data/diayn_ckpt',
        help='path of ckpt dir')
    argparser.add_argument(
        '--base_buffer_path',
        default='SAC_train_data/diayn_buffer',
        help='path of buffer dir')
    argparser.add_argument(
        '--number_z',
        default=5,
        help='number of skill')
    args = argparser.parse_args()
    server_env_configs = propre_envconfig(num_env=2)
    server_manager = server_utils.CarlaServerManager(args.carla_sh_path, configs=server_env_configs)
    server_manager.start()
    time.sleep(15)

    env_cfg1 = {
            'port':2000,
            'map_id':1,
        }
    env_cfg2 = {
        'port':2005,
        'map_id':2,
        }
    env_cfg = [env_cfg1, env_cfg2]
    if len(env_cfg) == 1:
        env = DummyVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
    else:
        env = SubprocVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
    policy_args = {
            'features_extractor_entry_point': 'SAC.torch_layers:diaynCNN', 
            'features_extractor_kwargs': {'states_neurons': [256, 256],
                                            'number_z': args.number_z}, 
            'use_sde': False,
        }
    sac_args = {
            'learning_rate': 1e-5,
            'buffer_size': 10000,
            'train_freq': (10,'step'),
            'gradient_steps': 10,
            'target_update_interval': 1,
            'learning_starts': 100,
            'batch_size': 128,
            'ent_coef': "auto",                    # defualt "auto"
            'add_action': True,
            'policy_kwargs': policy_args,
            'number_z': 5,
            'life_time': 1e5,
        }
    robot = Robot(env, 
            number_z=args.number_z, 
            sac_args=sac_args, 
            base_ckpt_path=args.base_ckpt_path, 
            base_buffer_path=args.base_buffer_path)

    tmp_z = 0
    video_path = 'SAC/video/' + 'eval_{}_{}.mp4'.format(robot._start_timesteps, tmp_z) 
    Robot.evaluate_policy(env=env, policy=robot.diayn.policy, video_path=video_path, min_eval_steps_per_z=100, number_z=robot._number_z)

if __name__ == '__main__':
    #main()
    eval()
