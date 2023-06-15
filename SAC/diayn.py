import io
import pathlib
from select import select
import time
from turtle import speed
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from collections import deque
from stable_baselines3.common.logger import Logger, configure, make_output_format
from torch.nn import functional as F
from tqdm import tqdm

import gym
import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TrainFrequencyUnit, RolloutReturn, TrainFreq
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
)
import torch
from SAC.diayn_policy import my_diaynPolicy
from gym_carla.wandb.wandb_callback_diayn import WandbCallback

class my_diayn(BaseAlgorithm):
    '''
    copy from sb3 off policy algorithm
    used for SAC/TD3
    '''
    def __init__(
        self,
        policy: Type[my_diaynPolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = True,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        _init_setup_model: bool = True,
        add_action: bool = True,
        number_z: int = 50,
        life_time: int = 1000,
        buffer_type: str = 'dict',
        
    ):
        super(my_diayn, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )


        self.env = env  
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.target_update_interval = target_update_interval
        self._add_action = add_action
        self.remove_time_limit_termination = remove_time_limit_termination

      
        self.train_freq = train_freq

        #self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup
        self._init_setup_model = _init_setup_model
        self._number_z = number_z
        self._life_time = life_time
        self._life_step = 0
        self._buffer_type = buffer_type


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        if self._buffer_type == 'dict':
            self.replay_buffer_class = DictReplayBuffer
        else:
            self.replay_buffer_class = ReplayBuffer
        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )
        else:
            print('load buffer from disk')
        # policy = self.policy
        # in super base_class init has the following code
        # self.policy_class = SACPolicy
        self.policy_class = my_diaynPolicy
        self.policy = self.policy_class(  
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            add_action=self._add_action,
            number_z=self._number_z,
            **self.policy_kwargs,  
        )
        self.policy = self.policy.to(self.device)
        
        #self._create_aliases()
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.discriminator = self.policy.discriminator

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_optimizer = None
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)
            self.log_ent_coef = th.log(self.ent_coef_tensor).requires_grad_(True).to(self.device)
            #print('haha{},{}'.format(self.log_ent_coef, self.ent_coef))

        self._convert_train_freq()
    
    def _setup_model_for_meta(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self._logger = configure('diayn',['stdout'])
        print('only load ckpt, do not load buffer')
        self.policy_class = my_diaynPolicy
        self.policy = self.policy_class(  
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            add_action=self._add_action,
            number_z=self._number_z,
            **self.policy_kwargs,  
        )
        self.policy = self.policy.to(self.device)
        
        #self._create_aliases()
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.discriminator = self.policy.discriminator

        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        tmp_z: int = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        #assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        #pbar = tqdm(total=train_freq.frequency)
        total_reward = 0
        total_value = 0
        # random z and  store z with info 
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            if self._life_step >= self._life_time:
                dones[:] = True
                self._life_step = 0
            # revise behavior z 
            tmp_z_onehot = np.zeros((self._number_z,))
            tmp_z_onehot[tmp_z] = 1
            new_obs['z_onehot'][:] = tmp_z_onehot
            
            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            self._life_step += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            #if callback.on_step() is False:
            #    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

            #pbar.update(1)
        #pbar.close()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, callback: BaseCallback, batch_size: int = 64):
        
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        dis_optimizers = [self.discriminator.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        dis_losses = []
        reward_ps = []

        # 1.train discrimnator
        for gradient_step in range(2): # 10*gradient_steps
            # Sample replay buffer
            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            data_loader = self.sample_data(self.replay_buffer, 256)
            for data in data_loader:
                observations = data['observations']
                actions = data['actions']
                with th.no_grad():
                    # mid extract
                    tmp_obs = WandbCallback.dict_to_tensor(observations, self.device) # replay_data.observations
                    mid_features = self.policy.features_extractor.half_forward(tmp_obs)
                    
                # compute discriminator loss
                if self._add_action:
                    logits = self.policy.discriminator(mid_features, actions) # replay_data.actions
                else:
                    logits = self.policy.discriminator(mid_features)
                # dis_loss = F.mse_loss(logits, replay_data.observations['z_onehot'])
                dis_loss = F.cross_entropy(logits, tmp_obs['z_onehot'].float()) # replay_data.observations
                self.discriminator.optimizer.zero_grad()
                dis_loss.backward()
                self.discriminator.optimizer.step()

        # 2.train rl with disc 
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
                #print('hh{},{}'.format(ent_coef.item(),self.ent_coef_tensor))
            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            
            with th.no_grad():
                # mid extract
                tmp_obs = WandbCallback.dict_to_tensor(replay_data.observations, self.device)
                mid_features = self.policy.features_extractor.half_forward(tmp_obs)
                
            # compute discriminator loss but not train
            if self._add_action:
                logits = self.policy.discriminator(mid_features, replay_data.actions)
            else:
                logits = self.policy.discriminator(mid_features)

            thr = 0.2
            speed_r = 0.2
            tmp_v = torch.abs(tmp_obs['state'][:,-2:].detach())
            reward_r = (tmp_v.sum(dim=1).view(-1,1) - thr) * 10
            reward_r = torch.clip(reward_r, -2.0, 1)
            # p_z is random, p_z = 1.0/number_z 
            log_p_z = torch.log(torch.zeros((batch_size,1)) + 1.0/self._number_z ).to(self.device)
            log_q_fi_zs = torch.sum(logits * replay_data.observations['z_onehot'], dim=1).view(-1,1).detach()
            reward_p = log_q_fi_zs - log_p_z/10.0 + reward_r
            #reward_p = -0.5 - log_p_z/10.0 + reward_r
            reward_ps.append(torch.mean(reward_p).item())
            mean_log_q_zs = torch.mean(log_q_fi_zs) # local_log_q_zs
            dis_losses.append(mean_log_q_zs.item())

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                #target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                target_q_values = reward_p + (1 - replay_data.dones) * self.gamma * next_q_values

        
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        
        self._n_updates += gradient_steps
        local_ent_coef = np.mean(ent_coefs)
        local_actor_loss = np.mean(actor_losses)
        local_critic_loss = np.mean(critic_losses)
        local_log_prob = log_prob.detach().mean()
        local_log_q_zs = np.mean(dis_losses)
        local_reward_ps = np.mean(reward_ps)

        #self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        #self.logger.record("train/ent_coef", np.mean(ent_coefs))
        #self.logger.record("train/actor_loss", np.mean(actor_losses))
        #self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            #self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            local_ent_coef_loss = np.mean(ent_coef_losses)
        else:
            local_ent_coef_loss = 0
        callback.update_locals(locals())

    def finetune(self, gradient_steps: int, callback: BaseCallback, replay_data: DictReplayBufferSamples = None):
        
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        dis_optimizers = [self.discriminator.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        dis_losses = []
        for gradient_step in range(gradient_steps):

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(self._transfer_env(replay_data.observations))
            action = self.actor.forward(self._transfer_env(replay_data.observations), deterministic=True).detach()
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
                #print('hh{},{}'.format(ent_coef.item(),self.ent_coef_tensor))
            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            
            with th.no_grad():
                # mid extract
                tmp_obs = WandbCallback.dict_to_tensor(self._transfer_env(replay_data.observations), self.device)
                mid_features = self.policy.features_extractor.half_forward(tmp_obs)
                #aug_mid_features = torch.cat([mid_features, replay_data.actions], dim=1)
            # compute discriminator loss
            if self._add_action:
                logits = self.policy.discriminator(mid_features, action)
            else:
                logits = self.policy.discriminator(mid_features)
            #dis_loss = F.mse_loss(logits, replay_data.observations['z_onehot'])
            dis_loss = F.cross_entropy(logits, replay_data.observations['z_onehot'].float())
            self.discriminator.optimizer.zero_grad()
            dis_loss.backward()
            self.discriminator.optimizer.step()
       
            log_q_fi_zs = torch.sum(logits * replay_data.observations['z_onehot'], dim=1).view(-1,1).detach()
            mean_log_q_zs = torch.mean(log_q_fi_zs) # local_log_q_zs
            dis_losses.append(mean_log_q_zs.item())

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(self._transfer_env(replay_data.next_observations))
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(self._transfer_env(replay_data.next_observations), next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # target_q_values = reward_p + (1 - replay_data.dones) * self.gamma * next_q_values

        
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(self._transfer_env(replay_data.observations), action)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(self._transfer_env(replay_data.observations), actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        
        self._n_updates += gradient_steps
        local_ent_coef = np.mean(ent_coefs)
        local_actor_loss = np.mean(actor_losses)
        local_critic_loss = np.mean(critic_losses)
        local_log_prob = log_prob.detach().mean()
        local_log_q_zs = np.mean(dis_losses)
        
        if len(ent_coef_losses) > 0:
            #self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            local_ent_coef_loss = np.mean(ent_coef_losses)
        else:
            local_ent_coef_loss = 0
        callback.update_locals(locals())

    def _transfer_env(self, obs: dict, env: VecEnv = None):
        tmp_obs = {}
        target_keys = ['birdview','state','z_onehot']
        for key in obs.keys():
            if key in target_keys:
                tmp_obs[key] = obs[key]
        return tmp_obs

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        number_z: int = 50,
    ) -> "BaseAlgorithm":
        self._total_timesteps = total_timesteps
        self._number_z = number_z
        pbar = tqdm(initial=self.num_timesteps, total=total_timesteps)
        frq = self.train_freq.frequency
        count_step = 0
        select_z = np.random.randint(0, self._number_z)
        #old_z = select_z
        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            tt0 = time.time()
            if count_step < 1000:
                count_step += frq * self.env.num_envs
            else:
                count_step = 0
                new_z = np.random.randint(0, self._number_z)

                #while new_z==select_z or new_z==old_z:
                #    new_z = np.random.randint(0, self._number_z)
                #old_z = select_z
                select_z = new_z
            
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                tmp_z=select_z,
            )
            self.t_rollout = time.time() - tt0
            callback.on_rollout_end()
            if rollout.continue_training is False:
                break
            
            if self.num_timesteps > 0 and self.num_timesteps - self.start_timestamp > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                t0 = time.time()
                if gradient_steps > 0:
                    self.train(
                        batch_size=self.batch_size, 
                        gradient_steps=gradient_steps, 
                        callback=callback
                    )
                    self.t_train = time.time() - t0
                    callback.on_training_end()
            #new_k = int(self.num_timesteps / 10000)
            #if old_k < new_k:
            #    old_k = new_k
            #if self.num_timesteps % 100 == 0:
            #    tmp_name = 'ckpt_{}.pth'.format(self.num_timesteps)
            #    self.policy.save('SAC/ckpt/'+ tmp_name)
            pbar.update(frq*self.env.num_envs)
        pbar.close()
        #print('finished train')

    def sample_data(self, replay_buffer:DictReplayBuffer, batch_size: int = 256): # return n_envs * batch_size
        pos = replay_buffer.pos
        size = replay_buffer.buffer_size
        full = replay_buffer.full
        start_idx = 0
        
        if full:
            end = size
        else:
            end = pos
        indices = np.random.permutation(size)
        while start_idx + batch_size < end:
            yield self.get_datasamples(replay_buffer, indices[start_idx : start_idx + batch_size], batch_size)
            start_idx += batch_size

    def get_datasamples(self, replay_buffer:DictReplayBuffer, batch_inds: np.ndarray, batch_size: int):
        observations = {}
        for key, obs in replay_buffer.observations.items():
            shape = (batch_size * self.n_envs,) + obs[0,0].shape
            obs = obs[batch_inds,:,:].reshape(shape)
            observations[key] = obs
        # observations = {key: obs[batch_inds, :, :].reshape for key, obs in replay_buffer.observations.items()}
        # observations ={key: replay_buffer.to_torch(obs[batch_inds]) for (key, obs) in replay_buffer.observations.items()}
        actions=replay_buffer.to_torch(replay_buffer.actions[batch_inds,:].reshape(batch_size*self.n_envs,-1))
        
        data = {
            'observations': observations,
            'actions': actions,
        }
        return data

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
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
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        pass

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)
        
    def _setup_learn(
        self,
        #total_timesteps: int,
        start_timestamp: int, 
        eval_env: Optional[GymEnv] = None,
        callback: WandbCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> None:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        #if isinstance(self.replay_buffer, HerReplayBuffer):
         #   replay_buffer = self.replay_buffer.replay_buffer
        #else:
        #replay_buffer = self.replay_buffer
        self._last_obs = self.env.reset()
        self.start_timestamp = start_timestamp
        self.start_time = time.time()
        self.num_timesteps = start_timestamp
        self._logger = configure('diayn',['stdout'])
        callback.init_callback(self)
        
    def _setup_eval(
        self,
        start_timestamp: int, 
        
    ) -> None:
        self._last_obs = self.env.reset()
        self.start_timestamp = start_timestamp
        self.start_time = time.time()
        self.num_timesteps = start_timestamp

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

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "discriminator.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
        
    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv],
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        buffer_path: str = None,
        train_bool: bool = True,
        **kwargs,
    ) -> "BaseAlgorithm":
       
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects, print_system_info=print_system_info
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # check_for_correct_spaces(env, data["observation_space"], data["action_space"])

        # noinspection PyArgumentList
        model = cls(  
            policy="MultiInputPolicy",
            env=env,
            policy_base= SACPolicy,
            # learning_rate=1e-5,
            device=device,
            _init_setup_model=False,
            **kwargs,
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        if train_bool:
            if buffer_path:
                model.load_replay_buffer(buffer_path)
            model._setup_model()
        else:
            model._setup_model_for_meta()
        

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                if pytorch_variables[name] is None:
                    continue
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error

        # print(kwargs['ent_coef'])
        if not isinstance(kwargs['ent_coef'], str):
            model.log_ent_coef = th.log(torch.tensor(kwargs['ent_coef'])).requires_grad_(True).to(model.device)
            model.ent_coef = kwargs['ent_coef']
            model.ent_coef_tensor = th.tensor(float(model.ent_coef)).to(model.device)
        # print(model.log_ent_coef)
        return model
    
    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
       
        data = self.__dict__.copy()
        tmp_data = {
            'observation_space': data['observation_space'],
            'action_space': data['action_space'],
            'policy_kwargs': data['policy_kwargs'],
            'log_ent_coef': data['log_ent_coef'], 
            'ent_coef_optimizer': data['ent_coef_optimizer'],
        }

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        #all_pytorch_variables = state_dicts_names + torch_variable_names
    
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        #params_to_save = self.get_parameters()
        params_to_save = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params_to_save[name] = attr.state_dict()

        save_to_zip_file(path, data=tmp_data, params=params_to_save, pytorch_variables=pytorch_variables)