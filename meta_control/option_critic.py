from operator import ne
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from typing import Any, Dict, List, Optional, Tuple, Union
from math import exp
import numpy as np
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
import pathlib
import io


def dict_to_tensor(obs:dict, device):
        return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

class my_meta_controller(nn.Module):
    def __init__(self,
            observation_space,
            num_options,
            eps_start=1.0,
            eps_min=0.1,
            eps_decay=int(1e6),
            eps_test=0.05,
            device='auto',
            testing=False,
            features_dim=256,
        ) -> None:
        super(my_meta_controller, self).__init__()
        states_neurons=[256]
        self.in_channels = observation_space['total_birdview'].shape[0]
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        if device == 'auto':
            self.device = torch.device('cuda:2')
        else:
            self.device = device
        self.testing = testing

        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        #self.test_flag = True
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['total_birdview'].sample()[None]).float()).shape[1]
        
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear_args = []
        for i in range(len(states_neurons)-1):
            self.state_linear_args.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear_args.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear_args)
        
        self.feature_head = nn.Sequential(
            nn.Linear(n_flatten+states_neurons[-1], 512), 
            nn.ReLU(),
            nn.Linear(512, features_dim), 
            #nn.ReLU()
        )

        self.q_value = nn.Sequential(
            nn.Linear(features_dim, 64),                 # Policy-Over-Options
            nn.ReLU(),
            nn.Linear(64, num_options)
        )
        self.q_target = nn.Sequential(
            nn.Linear(features_dim, 64),                 # Policy-Over-Options
            nn.ReLU(),
            nn.Linear(64, num_options)
        )
        self.terminations = nn.Sequential(
            nn.Linear(features_dim, num_options),                 # Option-Termination
            nn.Sigmoid(),
        )

        self.to(self.device)
        self.train(not testing)
        self.q_target.load_state_dict(self.q_value.state_dict())
        self.q_target.train(False)

    def get_feature(self, obs):  # Dict[torch.Tensor]
        birdview = obs['total_birdview']
        state = obs['state']
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)
        x = torch.cat((x, latent_state), dim=1)
        x = self.feature_head(x)
        return x
    
    def get_Q(self, state):
        return self.q_value(state)
    
    def get_Q_target(self, state):
        return self.q_target(state)
    
    def predict_option_termination(self, state: torch.Tensor, current_option) -> Tuple[List[bool], np.ndarray]: 
        # state is a tensor in cuda
        termination = self.terminations(state).detach()
        n_env = termination.shape[0]
        # there are two methods to code
        # 1.gather
        # 2.index, option can be list or tensor, need to shape
        ind = torch.arange(n_env)
        termination = termination[ind, current_option]
        
        #if termination.shape != torch.Size([1]):
        #    print(current_option)
        #    print(termination)
        option_termination = Bernoulli(termination).sample()
        
        Q_ = self.get_Q(state).cpu().detach()
        next_option = Q_.argmax(dim=-1).numpy().reshape(-1,1)

        ter_list = option_termination.cpu().tolist()
        ter_bool = [bool(t) for t in ter_list]
        return ter_bool, next_option, Q_
    
    def get_terminations(self, state):
        return self.terminations(state)
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps

    def load(cls, path: str, device: Union[torch.device, str] = "auto"):
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        if isinstance(device, str):
            if device == 'auto':
                device = torch.device("cuda:1")
        saved_variables = torch.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

def critic_loss(
    model: my_meta_controller, 
    batch_size: int, 
    replay_data: DictReplayBufferSamples, 
    device, 
    gamma: float = 0.9
):
    
    batch_idx = torch.arange(batch_size).long()
    options   = torch.LongTensor(replay_data.actions).to(model.device)
    rewards   = torch.FloatTensor(replay_data.rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(replay_data.dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_feature(dict_to_tensor(replay_data.observations, device))
    Q      = model.get_Q(states)
    
    # the update target contains Q_next, but for stable learning we use target network for this
    with torch.no_grad():
        next_states_target = model.get_feature(dict_to_tensor(replay_data.next_observations, device))
        next_Q_target      = model.get_Q_target(next_states_target) # detach?

        # Additionally, we need the beta probabilities of the next state
        next_states            = model.get_feature(dict_to_tensor(replay_data.next_observations, device))
        next_termination_probs = model.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * gamma * \
        ((1 - next_options_term_prob) * next_Q_target[batch_idx, options] + next_options_term_prob  * next_Q_target.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    state = model.get_feature(to_tensor(obs))
    next_state = model.get_feature(to_tensor(next_obs))
    next_state_prime = model_prime.get_feature(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss
