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



class WandbCallback(BaseCallback):
    def __init__(self, cfg, vec_env, base_path):
        super(WandbCallback, self).__init__(verbose=1)

        self._video_path = Path(os.path.join(base_path, 'meta_video'))
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path(os.path.join(base_path, 'meta_ckpt'))
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_dir = Path(os.path.join(base_path, 'meta_buffer'))
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project=cfg['wb_project'], name=cfg['wb_name'], notes=cfg['wb_notes'], tags=cfg['wb_tags'])
        self.vec_env = vec_env

        self._eval_step = int(4e4)
        #self._buffer_step = int(1e4)
        self._save_step = int(2e3)
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
        pass
        
    def _on_rollout_start(self):
        pass
        
    def _on_training_end(self) -> None:
        time_elapsed = time.time() - self.model.start_time
        wandb.log({
            'time/fps': (self.model.num_timesteps-self.model._start_timesteps) / time_elapsed,
            'time/rollout_time': self.model.rollout_t,
            'time/train_time': self.model.train_t,
        }, step=self.model.num_timesteps, commit=False)

        wandb.log({
            'train/td_loss': self.locals['local_td_loss'],
            'train/ter_loss': self.locals['local_ter_loss'],
            'train/q_value': self.locals['local_q_value'],
            #'train/skill_ent_coef': self.locals['local_ent_coef'],
            #'train/skill_actor_loss': self.locals['local_actor_loss'],
            #'train/skill_critic_loss': self.locals['local_critic_loss'],
            #'train/log_q_zs': self.locals['local_log_q_zs'],
        }, step=self.model.num_timesteps)
        self.n_epoch += 1

    def _on_rollout_end(self):
        
        if (self.model.num_timesteps - self._last_time_save_buffer) >= self._save_buffer_step:
            self._last_time_save_buffer = self.model.num_timesteps
            buffer_dir = (self._buffer_dir / f'buffer.pkl').as_posix()
            self.model.save_replay_buffer(buffer_dir)
            
        wandb.log({
            'rollout/reward': self.locals['local_reward'],
            'rollout/termination': self.locals['local_ter'],
        }, step=self.model.num_timesteps, commit=False)

    @staticmethod
    def dict_to_tensor(obs:dict, device):
        return {key: torch.as_tensor(_obs).float().to(device) for (key, _obs) in obs.items()}

    def evaluate_policy(controller, video_path: str, min_eval_steps: int = 1000):
        option_termination = []
        greedy_option = np.zeros((controller.env.num_envs,1), dtype=np.int32)
        list_render = []
        obs = controller.env.reset()
        for i in range(controller.env.num_envs):
            controller.env.set_attr('eval_mode', True, indices=i)
        for i in range(controller.env.num_envs):
            option_termination.append(True)
        pbar = tqdm(initial=0, total=min_eval_steps)
        for _ in range(min_eval_steps):
            pbar.update(1)
            last_obs_list = controller._split_obs(obs, controller.env.num_envs)
            
            action_list = []
            for i in range(controller.env.num_envs):
                single_obs = last_obs_list[i]
                single_ter = option_termination[i]
                single_opt = greedy_option[i]
                single_a = controller._sample_single_action(single_ter, single_opt, single_obs, 2, skill_env=controller.skill_env)
                action_list.append(single_a)
            action = np.concatenate(action_list, axis=0)
            obs, reward, done, info = controller.env.step(action)
            with torch.no_grad():
                features = controller.option_critic.get_feature(WandbCallback.dict_to_tensor(obs, controller.device))
                current_q_values_all_action = controller.option_critic.q_value(features.detach()).cpu()
                ind = torch.tensor(greedy_option).long()
                # print(current_q_values_all_action.shape, action.shape, ind.shape)
                #current_q_values = torch.gather(current_q_values_all_action, dim=-1, index=ind)
                tmp_ind = torch.arange(controller.env.num_envs)
                current_q_values = current_q_values_all_action[tmp_ind, ind.squeeze()].numpy()
                #print(current_q_values.shape, greedy_option.shape, ind.shape)
                option_termination, greedy_option = controller.option_critic.predict_option_termination(features.detach(), greedy_option)

            for i in range(controller.env.num_envs):
                    controller.env.set_attr('action', action[i], indices=i)
                    controller.env.set_attr('action_value', current_q_values[i], indices=i)
                    controller.env.set_attr('z', greedy_option[i], indices=i)
                    controller.env.set_attr('ter', option_termination[i], indices=i)

            list_render.append(controller.env.render('rgb_array'))

        pbar.close()
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()