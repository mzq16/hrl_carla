from logging import critical
from operator import mod
from random import random
import numpy as np
import time
from pathlib import Path
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import torch
from multi_actor.diayn_policy_multi import my_diaynPolicy

class WandbCallback(BaseCallback):
    def __init__(self, cfg, vec_env, base_path):
        super(WandbCallback, self).__init__(verbose=1)

        save_dir = Path.cwd()
        print(save_dir)
        # self._save_dir = save_dir
        self._video_path = Path(os.path.join(base_path, 'diayn_video'))
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path(os.path.join(base_path, 'diayn_ckpt'))
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_dir = Path(os.path.join(base_path, 'diayn_buffer'))
        self._buffer_dir.mkdir(parents=True, exist_ok=True)

        # wandb.init(project=cfg.wb_project, dir=save_dir, name=cfg.wb_runname)
        wandb.init(project=cfg['wb_project'], name=cfg['wb_name'], notes=cfg['wb_notes'], tags=cfg['wb_tags'])
        # wandb.config.update(OmegaConf.to_container(cfg))

        # wandb.save('./config_agent.yaml')
        # wandb.save('.hydra/*')

        self.vec_env = vec_env

        self._eval_step = int(4e4)
        #self._buffer_step = int(1e4)
        self._save_step = int(4e3)
        self._save_buffer_step = int(4e3)

    def _init_callback(self):
        self.n_epoch = 0
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps
        self._last_time_save = self.model.num_timesteps
        self._last_time_save_buffer = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        #print('train start')
        pass
        

    def _on_rollout_start(self):
        # self.model._last_obs = self.model.env.reset()
        #print('rollout start')
        pass
        

    def _on_training_end(self) -> None:
        #print(f'train_end: n_epoch: {self.n_epoch}, num_timesteps: {self.model.num_timesteps}, train_time: {self.model.t_train}')
        # save time
        time_elapsed = time.time() - self.model.start_time
        # train_debug   wandb.log(self.model.train_debug, step=self.model.num_timesteps)
        wandb.log({
            'train/learning_rate': self.model.learning_rate,
            'train/ent_coef': self.locals['local_ent_coef'],
            'train/actor_loss': self.locals['local_actor_loss'],
            'train/critic_loss': self.locals['local_critic_loss'],
            'train/ent_coef_loss': self.locals['local_ent_coef_loss'],
            'train/log_prob': self.locals['local_log_prob'],
            'train/dis_loss': self.locals['dis_loss'],
            'train/log_q_fi_zs': self.locals['local_log_q_zs'],
            'train/reward_p': self.locals['local_reward_ps'],

        }, step=self.model.num_timesteps)
        
        wandb.log({
            'time/fps': (self.model.num_timesteps-self.model.start_timestamp) / time_elapsed,
            'time/train': self.model.t_train,
            'time/rollout': self.model.t_rollout
        }, step=self.model.num_timesteps)
        
        if (self.model.num_timesteps - self._last_time_save) >= self._save_step:
            self._last_time_save = self.model.num_timesteps
            ckpt_path = (self._ckpt_dir / f'ckpt_{self.model.num_timesteps}.pth').as_posix()
            self.model.save(ckpt_path)
            #wandb.save(f'./{ckpt_path}')

        # evaluate and save checkpoint
        if (self.model.num_timesteps - self._last_time_eval) >= self._eval_step:
            print('start eval and save')
            self._last_time_eval = self.model.num_timesteps
            # evaluate
            
            tmp_z = np.random.randint(0,self.model._number_z)
            eval_video_path = (self._video_path / f'eval_{self.model.num_timesteps}_{tmp_z}.mp4').as_posix()
            avg_ep_stat, ep_events = self.evaluate_policy(
                env=self.vec_env, 
                policy=self.model.policy, 
                video_path=eval_video_path, 
                min_eval_steps_per_z=50,
                number_z=self.model._number_z)
            # log to wandb
            #wandb.log({f'video/{self.model.num_timesteps}': wandb.Video(eval_video_path)},
            #         step=self.model.num_timesteps)
            wandb.log(avg_ep_stat, step=self.model.num_timesteps)
        
        self.n_epoch += 1

    def _on_rollout_end(self):
        #print(f'rollout_end: n_epoch: {self.n_epoch}, num_timesteps: {self.model.num_timesteps}, rollout_time: {self.model.t_rollout}')
        #wandb.log({'time/rollout': self.model.t_rollout}, step=self.model.num_timesteps)

        # save rollout statistics
        #avg_ep_stat = self.get_avg_ep_stat(self.model.ep_stat_buffer, prefix='rollout/')
        #wandb.log(avg_ep_stat, step=self.model.num_timesteps)
        
        t0 = time.time()
        if (self.model.num_timesteps - self._last_time_save_buffer) >= self._save_buffer_step:
            self._last_time_save_buffer = self.model.num_timesteps
            buffer_dir = (self._buffer_dir / f'buffer.pkl').as_posix()
            self.model.save_replay_buffer(buffer_dir)
            save_buffer_time = time.time() - t0 
        else:
            save_buffer_time = 0
        wandb.log({
            'rollout/tmp_z': self.locals['tmp_z'],
            'rollout/life_step': self.model._life_step,
        }, step=self.model.num_timesteps)

    @staticmethod
    def evaluate_policy(env, policy: my_diaynPolicy, video_path: str, min_eval_steps_per_z: int = 100, number_z: int = 50):
        device = torch.device('cuda:1')
        policy = policy.eval()
        
        t0 = time.time()
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()
        all_onehot = np.eye(number_z)
        
        list_render = []
        ep_stat_buffer = []
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        nz_step = 0  
        # min_nz_step = 100  # change aciton z per nz_step 
        #env_done = np.array([False]*env.num_envs)
        pbar = tqdm(initial=0, total=min_eval_steps_per_z*number_z)
        for i in range(number_z):
            nz_step = 0
            tmp_z = i
            obs['z_onehot'][:] = all_onehot[tmp_z]
            o = WandbCallback.dict_to_tensor(obs, device)
            actor = policy.actor_list[i]
            critic = policy.critic_list[i]
            while nz_step < min_eval_steps_per_z:
                mean_actions, log_std, kwargs= actor.get_action_dist_params(o)
                actions, log_probs = actor.action_dist.log_prob_from_params(mean_actions, log_std)
                value, _ = torch.cat(critic(o, actions), dim=1).min(dim=1)
                #value, _ = torch.min(value, dim=1)
                actions = np.array(actions.detach().cpu())
                log_probs = np.array(log_probs.detach().cpu())
                mean_actions = np.array(mean_actions.detach().cpu())
                log_std = np.array(log_std.detach().cpu().exp())
                values = np.array(value.detach().cpu()).reshape(-1,)
                obs, reward, done, info = env.step(actions)
                
                
                for i in range(env.num_envs):
                    env.set_attr('action', actions[i], indices=i)
                    env.set_attr('action_value', values[i], indices=i)
                    env.set_attr('action_log_probs', log_probs[i], indices=i)
                    env.set_attr('action_mu', mean_actions[i], indices=i)
                    env.set_attr('action_sigma', log_std[i], indices=i)
                    env.set_attr('z', tmp_z, indices=i)

                obs['z_onehot'][:] = all_onehot[tmp_z]
                o = WandbCallback.dict_to_tensor(obs, device)
                nz_step += 1
                n_step += 1
                pbar.update(1)
                list_render.append(env.render('rgb_array'))
            #env_done |= done
            
            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['episode_stat'])
                #ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                #n_timeout += int(info[i]['timeout'])

            
        pbar.close()
        
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()
        
        avg_ep_stat = WandbCallback.get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
        #avg_ep_stat['eval/eval_timeout'] = n_timeout

        duration = time.time() - t0
        avg_ep_stat['time/t_eval'] = duration
        avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

        for i in range(env.num_envs):
            env.set_attr('eval_mode', False, indices=i)
        obs = env.reset()
        return avg_ep_stat, ep_events

    @staticmethod
    def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        avg_ep_stat = {}
        if len(ep_stat_buffer) > 0:
            for ep_info in ep_stat_buffer:
                for k, v in ep_info.items():
                    k_avg = f'{prefix}{k}'
                    if k_avg in avg_ep_stat:
                        avg_ep_stat[k_avg] += v
                    else:
                        avg_ep_stat[k_avg] = v

            n_episodes = float(len(ep_stat_buffer))
            for k in avg_ep_stat.keys():
                avg_ep_stat[k] /= n_episodes
            avg_ep_stat[f'{prefix}n_episodes'] = n_episodes

        return avg_ep_stat

    @staticmethod
    def dict_to_tensor(obs:dict, device):
        return {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in obs.items()}
