from operator import mod
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

class WandbCallback(BaseCallback):
    def __init__(self, cfg, vec_env):
        super(WandbCallback, self).__init__(verbose=1)

        save_dir = Path.cwd()
        print(save_dir)
        # self._save_dir = save_dir
        self._video_path = Path('SAC/pen_video')
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path('SAC/pen_ckpt')
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_dir = Path('SAC/pen_buffer')
        self._buffer_dir.mkdir(parents=True, exist_ok=True)

        # wandb.init(project=cfg.wb_project, dir=save_dir, name=cfg.wb_runname)
        wandb.init(project=cfg['wb_project'], name=cfg['wb_name'], notes=cfg['wb_notes'], tags=cfg['wb_tags'])
        # wandb.config.update(OmegaConf.to_container(cfg))

        # wandb.save('./config_agent.yaml')
        # wandb.save('.hydra/*')

        self.vec_env = vec_env

        self._eval_step = int(1e10)
        self._buffer_step = int(1e10)
        self._save_step = int(1e5)
        self._save_buffer_step = int(1e10)

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
       
        
        wandb.log({
            'train/learning_rate': self.model.learning_rate,
            'train/ent_coef': self.locals['local_ent_coef'],
            'train/actor_loss': self.locals['local_actor_loss'],
            'train/critic_loss': self.locals['local_critic_loss'],
            'train/ent_coef_loss': self.locals['local_ent_coef_loss'],
            'train/log_prob': self.locals['local_log_prob']

        }, step=self.model.num_timesteps)
        
        
        
        if (self.model.num_timesteps - self._last_time_save) >= self._save_step:
            self._last_time_save = self.model.num_timesteps
            ckpt_path = (self._ckpt_dir / f'ckpt_{self.model.num_timesteps}.pth').as_posix()
            self.model.save(ckpt_path)
            wandb.save(f'./{ckpt_path}')

        self.n_epoch += 1

    def _on_rollout_end(self):
        wandb.log({
            'rollout/reward': self.locals['rewards'],
            'rollout/value': self.locals['total_value'],
        }, step=self.model.num_timesteps)

    @staticmethod
    def evaluate_policy(env, policy, video_path, min_eval_steps=1000):
        device = torch.device('cuda:1')
        policy = policy.eval()
        t0 = time.time()
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()
        o = WandbCallback.dict_to_tensor(obs, device)
        list_render = []
        ep_stat_buffer = []
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        
        env_done = np.array([False]*env.num_envs)
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
            o = WandbCallback.dict_to_tensor(obs, device)

            for i in range(env.num_envs):
                env.set_attr('action', actions[i], indices=i)
                env.set_attr('action_value', values[i], indices=i)
                env.set_attr('action_log_probs', log_probs[i], indices=i)
                env.set_attr('action_mu', mean_actions[i], indices=i)
                env.set_attr('action_sigma', log_std[i], indices=i)
                
            list_render.append(env.render('rgb_array'))

            n_step += 1
            env_done |= done
            
            #print(n_step)
            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['episode_stat'])
                #ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                #n_timeout += int(info[i]['timeout'])

            pbar.update(1)
        pbar.close()
        
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()
        #print('close')
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
        return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
