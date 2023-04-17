"""Policies: abstract base class and concrete implementations."""

from turtle import forward, pos
import torch as th
import torch.nn as nn
import numpy as np
import torch
from torch.nn.functional import log_softmax
from typing import Any, Dict, List, Optional
import gym

from agent import torch_util as tu


class XtMaCNN(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        n_input_channels = observation_space['birdview'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['birdview'].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+states_neurons[-1], 512), nn.ReLU(),
                                    nn.Linear(512, features_dim), nn.ReLU())

        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    #def forward(self, birdview, state):
    def forward(self, obs):
        birdview = obs['birdview']
        state = obs['state']
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)

        # latent_state = state.repeat(1, state.shape[1]*256)

        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, observation_space, chans=(16, 32, 32, 64, 64), states_neurons=[256],
                 features_dim=256, nblock=2, batch_norm=False, final_relu=True):
        # (16, 32, 32)
        super().__init__()
        self.features_dim = features_dim
        self.final_relu = final_relu

        # image encoder
        curshape = observation_space['birdview'].shape
        s = 1 / np.sqrt(len(chans))  # per stack scale
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = tu.CnnDownStack(curshape[0], nblock=nblock, outchan=outchan, scale=s, batch_norm=batch_norm)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        # dense after concatenate
        n_image_latent = tu.intprod(curshape)
        self.dense = tu.NormedLinear(n_image_latent+states_neurons[-1], features_dim, scale=1.4)

        # state encoder
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(tu.NormedLinear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

    def forward(self, birdview, state):
        # birdview: [b, c, h, w]
        # x = x.to(dtype=th.float32) / self.scale_ob

        for layer in self.stacks:
            birdview = layer(birdview)

        x = th.flatten(birdview, 1)
        x = th.relu(x)

        latent_state = self.state_linear(state)

        x = th.cat((x, latent_state), dim=1)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x


class diayn_kit(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, features_dim=256, embedding_dim=64, pos_dim=5, number_z=50):
        super().__init__()
        self.features_dim = features_dim
        self.number_z = number_z
        self.embedding_dim = embedding_dim
        self.sqe_len = observation_space['birdview'].shape[0]
        self.h = observation_space['birdview'].shape[1]
        self.w = observation_space['birdview'].shape[2]
        # embedding net
        self.bev_embedding  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.h * self.w, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.ReLU(),
        )
        self.pos_embedding = nn.Sequential(
            nn.Linear(self.sqe_len, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.Flatten(),
        )
        self.z_embedding = nn.Sequential(
            nn.Linear(number_z, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.Flatten(),
        )
        state_dim = observation_space['state'].shape[0]
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )
        # Compute shape by doing one forward pass 
        # encoder
        with torch.no_grad():
            tmp_sample = th.as_tensor(observation_space['birdview'].sample()).float()
            # tmp_bev = tmp_sample.unsqueeze(1) already unsqueeze(o_space['bev'].sample()[None])
            tmp_sample.squeeze_(1)
            tmp_out = self.bev_embedding(tmp_sample)
            d_model = tmp_out.shape[-1]
            self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        
        # cnn 
        '''
        self.cnn = nn.Sequential(
            nn.Conv2d(self.sqe_len, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['birdview'].sample()[None]).float()).shape[1]
        '''
        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.sqe_len * self.d_model + 2 * embedding_dim, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, features_dim)
        )
        self.z_linear = nn.Sequential(
            nn.Linear(features_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def pre_obs(self, obs):
        z_array = obs['z'] # (batch_size, sqe_len)
        device = z_array.device
        z_onehot = torch.zeros((z_array.shape[0], self.number_z)) # (batch_size, number_z)
        pos_onehot = torch.zeros((z_array.shape[0], self.sqe_len)) # (batch_size, sqe_len)
        tmp_z = z_array[:, 0].reshape(-1, 1)
        tmp_len = torch.sum(z_array==tmp_z, dim=1).reshape(-1, 1)
        for i in range(z_array.shape[0]):
            z_onehot[i][int(tmp_z[i])] = 1
            pos_onehot[i][int(tmp_len[i]) - 1] = 1
        return z_onehot.to(device), pos_onehot.to(device)

        
    def forward(self, obs): # already in cuda Dict[torch.Tensor]
        birdview = obs['birdview']
        state = obs['state']
        z = obs['z']
        z_onehot, pos_onehot = self.pre_obs(obs)
        # embedding
        batch_size = birdview.shape[0]
        emb_bev_list = []
        for i in range(batch_size):
            tmp_bev = birdview[i]
            tmp_bev = tmp_bev.unsqueeze(1)
            tmp_out = self.bev_embedding(tmp_bev).unsqueeze(0) # 
            emb_bev_list.append(tmp_out)
        emb_bev = torch.cat(emb_bev_list, dim=0) # (batch_size, sqe_len, d_model) -> (batch_size, sqe_len, d_model)
        emb_bev_latent = self.encoder_layer(emb_bev)
        emb_bev_latent = emb_bev_latent.view(batch_size, -1)
        emb_pos = self.pos_embedding(pos_onehot)
        emb_z = self.z_embedding(z_onehot)
        latent_state = self.state_embedding(state)
        emb_x = torch.cat((emb_bev_latent, emb_pos, latent_state), dim=-1) # (batch_size, plus_dim)
        half_x = self.mlp_decoder(emb_x)
        x = torch.cat((half_x, emb_z), dim=-1)
        x = self.z_linear(x)
        return x
    
    def half_forward(self, obs):
        birdview = obs['birdview']
        state = obs['state']
        z_onehot, pos_onehot = self.pre_obs(obs)
        # embedding
        batch_size = birdview.shape[0]
        emb_bev_list = []
        for i in range(batch_size):
            tmp_bev = birdview[i]
            tmp_bev = tmp_bev.unsqueeze(1)
            tmp_out = self.bev_embedding(tmp_bev).unsqueeze(0) # 
            emb_bev_list.append(tmp_out)
        emb_bev = torch.cat(emb_bev_list, dim=0) # (batch_size, sqe_len, d_model) -> (batch_size, sqe_len, d_model)
        emb_bev_latent = self.encoder_layer(emb_bev)
        emb_bev_latent = emb_bev_latent.view(batch_size, -1)
        emb_pos = self.pos_embedding(pos_onehot)
        latent_state = self.state_embedding(state)
        emb_x = torch.cat((emb_bev_latent, emb_pos, latent_state), dim=-1) # (batch_size, plus_dim)
        x = self.mlp_decoder(emb_x)
        return x

class diaynFlatten(nn.Module):

    def __init__(self, observation_space: gym.Space, features_dim: int = 32, number_z=50):
        super().__init__()
        self.features_dim = features_dim
        self.observation_space = observation_space
        
        with th.no_grad():
            n_flatten = nn.Flatten(th.as_tensor(observation_space['raw'].sample()[None]).float())
        self.flatten = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.z_linear = nn.Sequential(
            nn.Linear(n_flatten + number_z, 32),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = self.flatten(observations['raw'])
        z_onehot = observations['z_onehot']
        x = th.cat((x, z_onehot), dim=1)
        x = self.z_linear(x)
        return x
    
    def half_forward(self, observations):
        x = self.flatten(observations)
        return x

class penFlatten(nn.Module):

    def __init__(self, observation_space: gym.Space, features_dim: int = 32, number_z=50):
        super().__init__()
        self.features_dim = features_dim
        self.observation_space = observation_space
        
        with th.no_grad():
            n_flatten = nn.Flatten(th.as_tensor(observation_space['raw'].sample()[None]).float())
        self.flatten = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.z_linear = nn.Sequential(
            nn.Linear(n_flatten + number_z, 32),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = self.flatten(observations['raw'])
        z_onehot = observations['z_onehot']
        x = th.cat((x, z_onehot), dim=1)
        x = self.z_linear(x)
        return x
    
    def half_forward(self, observations):
        x = self.flatten(observations)
        return x

class discriminator(nn.Module):
    def __init__(self, feat_latent_dim, action_dim, action_latent_dim=None, number_z=50, states_neurons=[100,100]) -> None:
        super().__init__()
        self.optimizer = None       # type: Optional[th.optim.Optimizer]
        self.action_dim = action_dim
        self.number_z = number_z
        input_dim = action_dim + feat_latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, states_neurons[0]),
            nn.ReLU(),
            nn.Linear(states_neurons[0], states_neurons[1]),
            nn.ReLU(),
            nn.Linear(states_neurons[1], number_z)
        )
        #self.action2latent = nn.Sequential(
        #    nn.Linear(action_dim, action_latent_dim),
        #    nn.ReLU(),
        #)

    def forward(self, x, action):
        #action_latent = self.action2latent(action)
        x = torch.cat([x, action], dim=1)
        x = self.mlp(x)
        #return log_softmax(x)
        return log_softmax(x,dim=1)

class discriminator_no_action(nn.Module):
    def __init__(self, feat_latent_dim, number_z=50, states_neurons=[100,100]) -> None:
        super().__init__()
        self.optimizer = None       # type: Optional[th.optim.Optimizer]
        self.number_z = number_z
        input_dim = feat_latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, states_neurons[0]),
            nn.ReLU(),
            nn.Linear(states_neurons[0], states_neurons[1]),
            nn.ReLU(),
            nn.Linear(states_neurons[1], number_z)
        )
        #self.action2latent = nn.Sequential(
        #    nn.Linear(action_dim, action_latent_dim),
        #    nn.ReLU(),
        #)

    def forward(self, x):
        #action_latent = self.action2latent(action)
        x = self.mlp(x)
        #return log_softmax(x)
        return log_softmax(x,dim=1)