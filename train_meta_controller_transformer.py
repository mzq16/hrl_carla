from random import random
import re
from site import venv
from tokenize import single_quoted
import diayn_robot_transformer
from diayn_robot_transformer import Robot
from meta_control.option_critic import my_meta_controller
from meta_control.option_critic import critic_loss as critic_loss_fn
from meta_control.option_critic import actor_loss as actor_loss_fn
import server_utils
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
import torch
from gym_carla.wandb.wandb_callback_meta import WandbCallback
from gym_carla.rlbev_wrapper import RLdiaynWrapper, RLdiayn_egov_Wrapper, RL_meta_Wrapper
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse
import gym
import os
import time
import pathlib
import io
import pickle
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from collections import deque
from train_ppo import get_avg_ep_stat

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def env_make(port, map_id=0, number_z=5, device=torch.device(0)):
    map_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town10HD_Opt']
    map_name = map_list[map_id]
    h5file_path = map_name + '.h5'
    host = 'localhost'
    env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022, carla_map=map_name, 
            file_path=h5file_path, num_vehicles=0, num_walkers=0, use_bev_fusion=False, device=device)
    # env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022)
    #env = RLdiaynWrapper(env=env_base, number_z=number_z)
    env = RL_meta_Wrapper(env=env_base, number_z=number_z, number_timestep=5)
    return env

class meta_control(object):
    def __init__(
        self, 
        env: VecEnv, 
        
        skill_robot: Robot, 
        number_z: int = 10, 
        meta_ckpt_path: str = None, 
        meta_buffer_path: str = None,
        device: torch.device = torch.device(0),
        ):
        
        self.env = env
        
        self.observation_space = env.observation_space
        self.device = device
        # the meta controller's action_space need to be overide
        #self.action_space = env.action_space
        self.action_space = gym.spaces.Box(low=0, high=number_z+1, shape=(1,), dtype=np.int32)
        self.n_env = env.num_envs
        # add a brake skill and a None-action(0,0) skill
        self.number_z = number_z
        self.total_z = number_z + 2

        self.skill_robot = skill_robot
        self.skill_robot.diayn.policy.set_training_mode(False)

        self.gradient_step = 10
        self.batch_size = 128
        self.gamma = 0.99
        self.rollout_step = 10
        self.max_grad_norm: float = 10
        self.reg_ter: float = 0.01
        self.reg_entropy: float = 0.01
        self.tau = 0.005
        
        self.target_update_interval = 10
        self.save_timestep = 2e3
        self.eval_timestep = 2e4 
        self.verbose = 0

        self.check_ckpt(meta_ckpt_path)
        self.check_buffer(meta_buffer_path)
        self.num_timesteps = self._start_timesteps
        self.logger = None
        number_timestep = 5
        self._number_timestep = number_timestep
        self._z_deque = [deque(maxlen=number_timestep) for i in range(env.num_envs)]
        for i in range(env.num_envs):
            self._z_deque[i].append(0)
        self._z_deque_tmp = deepcopy(self._z_deque)

        # build
        lr = 1e-5
        self._build(lr)
                
    def check_ckpt(self, dir_path='meta_control/meta_ckpt'):
        ckpt_list = os.listdir(dir_path)
        if ckpt_list:
            ckpt_list[0].split('_')[1].split('.')[0]
            max_ckpt = max(ckpt_list, key = lambda x: int(x.split('_')[1].split('.')[0]))
            self._dir_ckpt = os.path.join(dir_path, max_ckpt)
            self._start_timesteps = int(max_ckpt.split('_')[1].split('.')[0])
            print('resume meta checkpoint latest ' + max_ckpt)
        else:
            self._start_timesteps = 0
            self._dir_ckpt = None
            print('no exist meta ckpt, start a new train')

    def check_buffer(self, dir_path='meta_control/meta_buffer'):
        buffer_list = os.listdir(dir_path)
        if len(buffer_list) == 0:
            self._dir_buffer = None
            print('no exist meta buffer')
        else:
            print('load meta buffer')
            self._dir_buffer = os.path.join(dir_path, buffer_list[0])
            
    def _build(self, lr):
        self.option_critic = my_meta_controller(
            observation_space=self.observation_space, 
            num_options=self.total_z,
            eps_start=1.0,
            eps_min=0.1,
            eps_decay=int(1e6),
            eps_test=0.05,
            device=self.device,
            testing=False,
        )
        if self._dir_ckpt is not None:
            ckpt = torch.load(self._dir_ckpt, map_location=self.device)
            self.option_critic.load_state_dict(ckpt['state_dict'])

        if self._dir_buffer is None:
            self.replay_buffer = DictReplayBuffer(
                buffer_size=10000,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                n_envs=self.env.num_envs
            )
        else:
            self.load_replay_buffer(self._dir_buffer)

        self.optimizer = torch.optim.Adam(self.option_critic.parameters(), lr=lr)
        self.optimizer_ter = torch.optim.Adam(self.option_critic.parameters(), lr=lr)

    def collect_rollout(
        self, 
        rollout_step: int, 
        current_option: np.ndarray, 
        curr_teminal: List[bool], 
        epsilon: float, 
        n_env: int,
        callback: BaseCallback,
        ):
        # switch training mode
        self.option_critic.train(False)
        self.skill_robot.diayn.policy.set_training_mode(False)

        total_reward = []
        total_ter = []
        
        if isinstance(curr_teminal, bool):
            ter_list = []
            for i in range(n_env):
                ter_list.append(curr_teminal)
            option_termination = ter_list
        elif isinstance(curr_teminal, List):
            option_termination = curr_teminal
        else:
            raise TypeError(f'the termination is not a list, bool')

        if isinstance(current_option, int):
            opt_array = np.zeros((n_env, 1), dtype=np.int32) + current_option
            greedy_option = opt_array
        elif isinstance(current_option, np.ndarray):
            greedy_option = current_option
        else:
            raise TypeError(f'the option is not a np.ndarray, int')
        
        for _ in range(rollout_step):
            last_obs_list = self._split_obs(self._last_obs, self.n_env)
            dones=[False for i in range(n_env)]
            action_list = []
            for i in range(n_env):
                
                single_obs = last_obs_list[i]
                single_ter = option_termination[i]
                single_opt = greedy_option[i]
                single_a, self._z_deque[i], self._z_deque_tmp[i] = self._sample_single_action(single_ter, single_opt, single_obs, epsilon,
                 z_deque=self._z_deque[i], z_deque_tmp=self._z_deque_tmp[i], done=dones[i])
                action_list.append(single_a)
            action = np.concatenate(action_list, axis=0)
            obs, reward, dones, info = self.env.step(action)
            total_reward.append(reward)
            # Store data in replay buffer (normalized action and unnormalized observation)
            # so the action name is buffer action, 
            # but my action range from -1 to 1, is already normalized
            self._store_transition(
                replay_buffer=self.replay_buffer, buffer_action=greedy_option, new_obs=obs, reward=reward, dones=dones, infos=info)
            features = self.option_critic.get_feature(WandbCallback.dict_to_tensor(self._last_obs, self.device))
            with torch.no_grad():
                option_termination, greedy_option, Q_values = self.option_critic.predict_option_termination(features, current_option)
            total_ter.append(option_termination)
        # wandb_callback local
        local_reward = np.mean(total_reward)
        local_ter = np.mean(total_ter)

        callback.update_locals(locals())

        return option_termination, greedy_option
        
    def train(self, callback: BaseCallback, gradient_step: int = 20, batch_size: int = 128):
        self.option_critic.train(True)

        td_losses = []
        ter_losses = []
        critic_losses = []
        q_values = []
        for _ in range(gradient_step):

            ''' 
            the following code is AC, but is used for continuous actor, 
            对离散动作, 需要actor分布接一个softmax头, 或者是选择离散的分布而不是一般高斯

            replay_data = self.replay_buffer.sample(self.batch_size, env=None)
            td_error = critic_loss(
                model=self.option_critic, 
                batch_size=self.batch_size, 
                replay_data=replay_data,
                device=self.device,
                gamma=0.99,
            )
            '''

            ''' 
            the following code is DQN, but do not consider termination loss 
            I need to add a termination loss
            '''
            replay_data = self.replay_buffer.sample(batch_size, env=None) # None env means no vecenv
            with torch.no_grad():
                assert isinstance(replay_data.next_observations, Dict), 'the type of obs is wrong'
                next_features = self.option_critic.get_feature(WandbCallback.dict_to_tensor(replay_data.next_observations, self.device)).detach()
                next_target_q = self.option_critic.q_target(next_features)
                next_target_q, _ = next_target_q.max(dim=1)
                next_target_q = next_target_q.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_target_q
            
            current_features = self.option_critic.get_feature(WandbCallback.dict_to_tensor(replay_data.observations, self.device))
            current_q_values_all_action = self.option_critic.q_value(current_features)
            current_q_values = torch.gather(current_q_values_all_action, dim=1, index=replay_data.actions.long())
            
            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)  # (batch_size, 1)
            #print(td_loss)
            self.optimizer.zero_grad()
            td_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.option_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            option_ter_prob = self.option_critic.get_terminations(current_features.detach())
            option_ter_prob = torch.gather(option_ter_prob, dim=1, index=replay_data.actions.long())
            with torch.no_grad():
                tmp = self.option_critic.q_value(current_features)
                tmp_test = tmp.max(dim=-1)[0].view(-1,1).detach()
            #print(replay_data.dones)
            termination_loss = option_ter_prob * (current_q_values.detach() + 1e-6 - tmp_test) \
                 * (1 - replay_data.dones.detach() + 1e-6)
            termination_loss = termination_loss.mean()
            self.optimizer_ter.zero_grad()
            termination_loss.backward()
            self.optimizer_ter.step()
            # td_loss and termination loss cannot be cal together 
            # loss = td_loss + termination_loss

            td_losses.append(td_loss.item())
            ter_losses.append(termination_loss.item())
            q_values.append(current_q_values.mean().detach().item())
            
        local_td_loss = np.mean(td_losses)
        local_ter_loss = np.mean(ter_losses)
        local_q_value = np.mean(q_values)
        callback.update_locals(locals())
              
    def setup_learn(self, callback: BaseCallback):
        self.start_time = time.time()
        self.callback = callback
        self.callback.init_callback(self)
        # self._last_obs = self.env.reset()

    def learn(self, total_timesteps=1000):
        timestep = self._start_timesteps
        current_teminal = True
        current_option = 0
        epsilon = 0.9

        obs = self.env.reset()
        self._last_obs = obs

        pbar = tqdm(initial=self._start_timesteps, total=total_timesteps)
        while timestep < total_timesteps:
            # rollout
            t0 = time.time()
            current_teminal, current_option = self.collect_rollout(
                rollout_step=self.rollout_step, 
                current_option=current_option,
                curr_teminal=current_teminal,
                epsilon=epsilon,
                n_env=self.n_env,
                callback=self.callback
            )
            self.rollout_t = time.time() - t0
            
            self.callback.on_rollout_end()
            # train
            t1 = time.time()
            self.train(callback=self.callback, gradient_step=self.gradient_step, batch_size=self.batch_size)
            #self.skill_robot.learn_from_meta(replay_buffer=self.replay_buffer, callback=self.callback, gredient_step=10, batch_size=self.batch_size)
            self.train_t = time.time() - t1
            self.callback.on_training_end()

            self._on_step(self.rollout_step*self.n_env)
            pbar.update(self.rollout_step*self.n_env)
            timestep += self.rollout_step*self.n_env

        pbar.close()  

    def learn_finetune(self):
        self.skill_robot.learn_from_meta(self.replay_buffer, self.callback)

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        copy from diayn from sb3.sac
        """
        
        # Store only the unnormalized version
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        '''
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
        '''         
        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs

    def _update_z_deque_tmp(self, z_deque, z_deque_tmp, done = False):
        z_array = np.zeros((1,self._number_timestep))
            
        if done is True:
            z_deque.clear()
            z_deque.append(0)
            z_deque_tmp = deepcopy(z_deque)
        assert len(z_deque) != 0, 'z_deque has nothing' 
        # assert len(z_deque_tmp) != 0, 'z_deque_tmp has nothing' 
        tmp_z = z_deque[-1]
        if tmp_z == z_deque_tmp[-1]:
            z_deque_tmp.append(tmp_z)
        else:
            z_deque_tmp.clear()
            z_deque_tmp.append(tmp_z)
        tmp_z_array = np.array(z_deque_tmp)

            # padding
        if len(tmp_z_array) != self._number_timestep:
            pad_len = self._number_timestep - len(tmp_z_array)
            tmp_z_array = np.pad(tmp_z_array, (0,pad_len), 'constant', constant_values=(0,-1))
                
        assert len(tmp_z_array) == self._number_timestep, 'the length of tmp_z_array is wrong'
        z_array[0] = tmp_z_array
        return z_array, z_deque, z_deque_tmp

    def _sample_single_action(
        self, 
        option_termination: bool, 
        greedy_option: int, 
        obs: dict, 
        epsilon: float,
        z_deque: list,
        z_deque_tmp: list, 
        done: bool = False
        ) -> np.ndarray:
        if option_termination:
            #total_options[str(current_option)] = curr_op_length
            if np.random.rand() < epsilon:
                # features = self.option_critic.get_feature(self._last_obs)
                current_option = greedy_option
            else:
                current_option = np.random.choice(self.total_z) 
        else:
            current_option = greedy_option

        z_deque.append(int(current_option))
        # option_termination
        if current_option == self.number_z:
            action = np.array([[-1.0,0.0]])
        elif current_option == self.number_z + 1:
            action = np.array([[0.1,0.0]])
        else:
            tmp_obs = self._transfer_env(obs=obs)
            z_array, z_deque, z_deque_tmp = self._update_z_deque_tmp(z_deque=z_deque, z_deque_tmp=z_deque_tmp, done=done)
            #action = self.skill_robot.forward(observation=tmp_obs, skill=current_option, deterministic=True)
            action = self.skill_robot.forward(observation=tmp_obs, skill=None, z_array=z_array, deterministic=True, device=self.device)
            action = action.cpu().detach().numpy()
            
        return action, z_deque, z_deque_tmp

    def _split_obs(self, obs: dict, n_env: int):
        single_obs = {}
        obs_list = []
        # check n_env dim
        for key in obs.keys():
                assert obs[key].shape[0] == n_env, '{} dim is not consistent with n_env'.format(key) 
        for i in range(n_env):
            for key in obs.keys():
                single_obs[key] = np.expand_dims(obs[key][i], axis=0)
            obs_list.append(single_obs)

        return obs_list

    def _transfer_env(self, obs: dict):
        tmp_obs = {}
        target_keys = ['birdview','state','z'] # &transformer 'z', origin 'z_onehot'
        for key in obs.keys():
            if key in target_keys:
                tmp_obs[key] = obs[key]
        return tmp_obs

    def _on_step(self, time_step) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self.num_timesteps += time_step
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.option_critic.q_value.parameters(), self.option_critic.q_target.parameters(), self.tau) 

        if self.num_timesteps % self.save_timestep == 0:
            meta_path = 'meta_control/meta_ckpt'
            diayn_path = 'meta_control/diayn_ckpt'
            file_name = 'ckpt_{}.pth'.format(self.num_timesteps)
            out_meta_path = os.path.join(meta_path, file_name)
            out_diayn_path = os.path.join(diayn_path, file_name)
            self.save(out_meta_path)
            #self.save_diayn(out_diayn_path)

        if self.num_timesteps % self.eval_timestep == 0:
            pass
            #evaluate_policy(controller=self, video_path='meta_control/meta_video/video_{}.mp4'.format(self.num_timesteps), min_eval_steps=1000)

    def save(self, path):
        torch.save({"state_dict": self.option_critic.state_dict()}, path)   

    def save_diayn(self, path):
        self.skill_robot.diayn.save(path=path)   

    def get_env(self):
        return self.env

def get_robot_args(args):
    policy_args = {
            'features_extractor_entry_point': 'diayn_pack.torch_layers:diayn_kit', 
            'features_extractor_kwargs': {'features_dim': 256, 'embedding_dim': 64, 'pos_dim': 5, 'number_z': args.number_z}, 
            'use_sde': False,
            }
    sac_args = {
        'learning_rate': 5e-5,
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
    return sac_args, policy_args

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
        '--skill_ckpt_path',
        default='meta_control/diayn_ckpt',
        help='path of skill robot ckpt dir')
    argparser.add_argument(
        '--skill_buffer_path',
        default='meta_control/diayn_buffer',
        help='path of skill robot buffer dir')
    argparser.add_argument(
        '--meta_buffer_path',
        default='meta_control/meta_buffer',
        help='path of skill robot buffer dir')
    argparser.add_argument(
        '--meta_ckpt_path',
        default='meta_control/meta_ckpt',
        help='path of skill robot buffer dir')
    argparser.add_argument(
        '--number_z',
        default=5,
        help='number of skill')
    args = argparser.parse_args()
    
    
    try:
        server_env_configs = diayn_robot_transformer.propre_envconfig(num_env=2)
        server_manager = server_utils.CarlaServerManager(args.carla_sh_path, configs=server_env_configs)
        server_manager.start()
        device = torch.device(1)
        env_cfg1 = {
            'port':2000,
            'map_id':0,
            'number_z':args.number_z,
            'device':device,
        }
        env_cfg2 = {
            'port':2005,
            'map_id':1,
            'number_z':args.number_z,
            'device':device,
        }
        env_cfg = [env_cfg1, env_cfg2]
        # env_cfg = [env_cfg1]
        if len(env_cfg) == 1:
            env = DummyVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        else:
            env = SubprocVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        # test two env wrapper 
        # diayn env create 
        '''
        if len(env_cfg) == 1:
            env_d = DummyVecEnv([lambda port=port: diayn_robot_transformer.env_make(**port) for port in env_cfg])
        else:
            env_d = SubprocVecEnv([lambda port=port: diayn_robot_transformer.env_make(**port) for port in env_cfg])
        '''
        # prepare wandb
        wandb_cfg = {'wb_project': 'meta_5skill_road&ego_transform_school', 'wb_name': None, 'wb_notes': None, 'wb_tags': None}
        wb_callback = WandbCallback(wandb_cfg, env, 'meta_control')
        callback = CallbackList([wb_callback])
        # prepare diayn robot
        sac_args, policy_args = get_robot_args(args)
        robot = Robot(
            env, 
            number_z=args.number_z, 
            sac_args=sac_args, 
            base_ckpt_path=args.skill_ckpt_path, 
            base_buffer_path=args.skill_buffer_path,
            train_bool=False,
            device=device,
        )
        controller = meta_control(
            env=env,
            skill_robot=robot,
            number_z=robot._number_z,
            meta_ckpt_path=args.meta_ckpt_path, 
            meta_buffer_path=args.meta_buffer_path,
            device=device,
        )
        controller.setup_learn(callback=callback)
        controller.learn(total_timesteps=1000000)
        # controller.skill_robot.learn_from_meta(replay_buffer=controller.replay_buffer, callback=callback)

    # except:
    #    pid = os.getpid()
    #    subprocess.Popen('kill {}'.format(pid), shell=True)
    
    finally:
        env.close()
        server_manager.stop()

def evaluate_policy(controller: meta_control, video_folder_path: str, info_folder_path: str, min_eval_steps: int = 1000):
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


    controller.option_critic.train(False)
    controller.skill_robot.diayn.policy.set_training_mode(False)
    option_termination = []
    greedy_option = np.zeros((controller.env.num_envs,1), dtype=np.int32)
    epsilon = 1
    
    total_reward = []
    list_render = []
    obs = controller.env.reset()
    ep_stat_buffer = []
    for i in range(controller.env.num_envs):
        controller.env.set_attr('eval_mode', True, indices=i)
        option_termination.append(True)

    dones=[False for i in range(controller.env.num_envs)]
    pbar = tqdm(initial=0)

    while len(ep_stat_buffer) < 2:
        pbar.update(1)
        last_obs_list = controller._split_obs(obs, controller.env.num_envs)
        
        action_list = []
        for i in range(controller.env.num_envs):
            single_obs = last_obs_list[i]
            single_ter = option_termination[i]
            single_opt = greedy_option[i]
            # single_a = controller._sample_single_action(single_ter, single_opt, single_obs, 2, skill_env=controller.skill_env)
            single_a, controller._z_deque[i], controller._z_deque_tmp[i] = controller._sample_single_action(single_ter, single_opt, single_obs, epsilon,
                 z_deque=controller._z_deque[i], z_deque_tmp=controller._z_deque_tmp[i], done=dones[i])
            action_list.append(single_a)
        action = np.concatenate(action_list, axis=0)
        obs, reward, dones, info = controller.env.step(action)
        # total_reward.append(reward)
        # local_reward = np.mean(total_reward)

        with torch.no_grad():
            features = controller.option_critic.get_feature(WandbCallback.dict_to_tensor(obs, controller.device))
            current_q_values_all_action = controller.option_critic.q_value(features.detach()).cpu()
            ind = torch.tensor(greedy_option).long()
            # print(current_q_values_all_action.shape, action.shape, ind.shape)
            #current_q_values = torch.gather(current_q_values_all_action, dim=-1, index=ind)
            tmp_ind = torch.arange(controller.env.num_envs)
            current_q_values = current_q_values_all_action[tmp_ind, ind.squeeze()].numpy()
            #print(current_q_values.shape, greedy_option.shape, ind.shape)
            option_termination, greedy_option, Q_values = controller.option_critic.predict_option_termination(features.detach(), greedy_option)

        for i in range(controller.env.num_envs):
                controller.env.set_attr('action', action[i], indices=i)
                controller.env.set_attr('action_value', current_q_values[i], indices=i)
                controller.env.set_attr('z', greedy_option[i], indices=i)
                controller.env.set_attr('ter', option_termination[i], indices=i)
                controller.env.set_attr('q_values', Q_values[i], indices=i)

        list_render.append(controller.env.render('rgb_array'))
        for i in np.where(dones)[0]:
            ep_stat_buffer.append(info[i]['eval_info'])

    pbar.close()
    eval_info = get_avg_ep_stat(ep_stat_buffer)
    #video_path = os.path.join(video_folder_path,'1.mp4')
    #info_path = os.path.join(info_folder_path, '1.pkl')
    with open(info_path,'wb') as f:
        pickle.dump(eval_info, f)
    encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

def eval():
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
        '--skill_ckpt_path',
        default='meta_control/diayn_ckpt',
        help='path of skill robot ckpt dir')
    argparser.add_argument(
        '--skill_buffer_path',
        default='meta_control/diayn_buffer',
        help='path of skill robot buffer dir')
    argparser.add_argument(
        '--meta_buffer_path',
        default='meta_control/meta_buffer',
        help='path of skill robot buffer dir')
    argparser.add_argument(
        '--meta_ckpt_path',
        default='meta_control/meta_ckpt',
        help='path of skill robot buffer dir')
    argparser.add_argument(
        '--number_z',
        default=5,
        help='number of skill')
    args = argparser.parse_args()
    
    
    try:
        server_env_configs = diayn_robot_transformer.propre_envconfig(num_env=2)
        server_manager = server_utils.CarlaServerManager(args.carla_sh_path, configs=server_env_configs)
        server_manager.start()
        device = torch.device(0)
        env_cfg1 = {
            'port':2000,
            'map_id':0,
            'number_z':args.number_z,
            'device':device,
        }
        env_cfg2 = {
            'port':2005,
            'map_id':1,
            'number_z':args.number_z,
            'device':device,
        }
        env_cfg = [env_cfg1, env_cfg2]
        #env_cfg = [env_cfg1]
        if len(env_cfg) == 1:
            env = DummyVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        else:
            env = SubprocVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        # test two env wrapper 
        # diayn env create 
        '''
        if len(env_cfg) == 1:
            env_d = DummyVecEnv([lambda port=port: diayn_robot_transformer.env_make(**port) for port in env_cfg])
        else:
            env_d = SubprocVecEnv([lambda port=port: diayn_robot_transformer.env_make(**port) for port in env_cfg])
        '''
       
        callback = None
        # prepare diayn robot
        sac_args, policy_args = get_robot_args(args)
        # diayn robot, get skill network robot
        robot = Robot(
            env, 
            number_z=args.number_z, 
            sac_args=sac_args, 
            base_ckpt_path=args.skill_ckpt_path, 
            base_buffer_path=args.skill_buffer_path,
            train_bool=False,
            device=device
        )

        # get meta network to control skill robot
        controller = meta_control(
            env=env,
            skill_robot=robot,
            number_z=robot._number_z,
            meta_ckpt_path=args.meta_ckpt_path, 
            meta_buffer_path=args.meta_buffer_path,
            device=device,
        )

        evaluate_policy(controller=controller, 
            video_folder_path='meta_control/eval/video', 
            info_folder_path='meta_control/eval/info', 
            min_eval_steps=1000)

    finally:
        env.close()
        server_manager.stop()
    

if __name__=="__main__":
    main()
    #eval()
