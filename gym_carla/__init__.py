from gym.envs.registration import register
from importlib_metadata import entry_points

register(
    id='carla_mzq-v0',
    entry_point='gym_carla.envs:CarlaEnv_mzq',
) 

register(
    id='task1-v0',
    entry_point='gym_carla.envs:CarlaEnv_task1'
)
