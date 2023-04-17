import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.type_aliases import Schedule
from SAC.utils import load_entry_point
from SAC.torch_layers import discriminator, discriminator_no_action


class my_diaynPolicy(BasePolicy):
    """
    copy from sb3
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        #features_extractor_class: type[BaseFeaturesExtractor] = None,
        features_extractor_entry_point: str = 'agent.torch_layers:XtMaCNN',
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        add_action: bool = True,
        number_z: int = 50,
    ):
        self.features_extractor_entry_point = features_extractor_entry_point
        features_extractor_class = load_entry_point(features_extractor_entry_point)
        features_extractor = features_extractor_class(
            observation_space=observation_space, 
            #number_z=self._number_z,
            **features_extractor_kwargs)
        self._add_action = add_action
        self._number_z = number_z
        
        super(my_diaynPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor=features_extractor,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )
        self.actor_list: List[Actor] = []
        self.actor_target = None
        self.critic_list: List[ContinuousCritic] = []
        self.critic_target_list = []
        self.share_features_extractor = share_features_extractor

        if th.cuda.is_available():
            self._device = th.device('cuda:1')
        else:
            self._device = 'cpu'
        self._build(lr_schedule=lr_schedule, num_actor=self._number_z)

    def _build(self, lr_schedule: Schedule, num_actor=10) -> None:
        for i in range(num_actor):
            # actor
            actor = self.make_actor()
            actor.optimizer = self.optimizer_class(actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
            self.actor_list.append(actor)
            
            # critic
            if self.share_features_extractor:
                critic = self.make_critic(features_extractor=actor.features_extractor)
                # Do not optimize the shared features extractor with the critic loss
                # otherwise, there are gradient computation issues
                critic_parameters = [param for name, param in critic.named_parameters() if "features_extractor" not in name]
            else:
                # Create a separate features extractor for the critic
                # this requires more memory and computation
                critic = self.make_critic(features_extractor=None)
                critic_parameters = critic.parameters()
            critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
            self.critic_list.append(critic)

            # Critic target should not share the features extractor with critic
            critic_target = self.make_critic(features_extractor=None)
            critic_target.load_state_dict(critic.state_dict())
            # Target networks should always be in eval mode
            critic_target.set_training_mode(False)
            self.critic_target_list.append(critic_target)
            
            # TODO add modules to the class from the Lists

            self.__setattr__('actor_{}'.format(i), actor)
            self.__setattr__('critic_{}'.format(i), critic)
            self.__setattr__('critic_target_{}'.format(i), critic_target)

        
        self.discriminator = self.make_discriminator(number_z=self._number_z)
        self.discriminator.optimizer = self.optimizer_class(self.discriminator.parameters(), lr=3e-5, **self.optimizer_kwargs)

    def to_cuda(self, device):
        for i in range(self._number_z):
            actor = self.actor_list[i]
            critic = self.critic_list[i]
            critic_target = self.critic_target_list[i]
            actor.to(device=device)
            critic.to(device=device)
            critic_target.to(device=device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                #features_extractor_class=self.features_extractor_class,
                features_extractor_entry_point = self.features_extractor_entry_point,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        number_a = len(self.actor_list)
        assert number_a == self._number_z, "the number of actor must be consistent"
        for i in range(number_a):
            actor = self.actor_list[i]
            actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        #return Actor(**actor_kwargs).to(self.device)
        return Actor(**actor_kwargs)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        #return ContinuousCritic(**critic_kwargs).to(self.device)
        return ContinuousCritic(**critic_kwargs)

    def make_discriminator(self, feat_latent_dim=256, number_z=50, action_dim=2, action_latent_dim=128):
        if self._add_action:
            discriminator_kwargs = {
                'feat_latent_dim':feat_latent_dim,
                'number_z':number_z,
                'action_dim':action_dim,
                }
            discrim = discriminator(**discriminator_kwargs)
        else:
            discriminator_kwargs = {
                'feat_latent_dim':feat_latent_dim,
                'number_z':number_z,
                }
            discrim = discriminator_no_action(**discriminator_kwargs)
        return discrim

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def predict_multi(
        self, 
        observation: np.ndarray, 
        deterministic: bool = False,
        index_actor: int = 0,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, index_actor=index_actor)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            # defualt squash: False
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions

    def _predict(self, observation: th.Tensor, deterministic: bool = False, index_actor: int = 0) -> th.Tensor:
        assert len(self.actor_list) > index_actor, "out of index"
        actor = self.actor_list[index_actor]
        return actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        number_a = len(self.actor_list)
        #print(number_a, self._number_z)
        assert number_a == self._number_z, "the number of actor must be consistent"
        for i in range(number_a):
            actor = self.actor_list[i]
            actor.set_training_mode(mode)
            critic = self.critic_list[i]
            critic.set_training_mode(mode)
        self.discriminator.train(mode)
        self.training = mode

