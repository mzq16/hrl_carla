import gym
import numpy as np
import torchaudio
from zmq import device
from SAC.sac import my_sac
from stable_baselines3.sac.policies import SACPolicy
import carla
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from gym_carla.rlbev_wrapper import RLBEVWrapper
import gym
from SAC.sac_policy import my_SACPolicy
from gym_carla.envs.carla_env_mzq import CarlaEnv_mzq
from SAC.sac import my_sac
import torch
import os
import time
import server_utils
import subprocess
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from tqdm import tqdm
from stable_baselines3.common.callbacks import CallbackList
from gym_carla.wandb.wandb_callback_sac import WandbCallback
import wandb
import warnings
import pickle
warnings.filterwarnings("ignore")


def env_make(port, map_id=0, device=torch.device(0)):
    map_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town10HD_Opt']
    map_name = map_list[map_id]
    h5file_path = map_name + '.h5'
    host = 'localhost'
    env_base = gym.make('carla_mzq-v0', host=host, port=port, seed=2022, carla_map=map_name, 
            file_path=h5file_path, num_vehicles=100, num_walkers=0, use_bev_fusion=False, device=device)
                        
    env = RLBEVWrapper(env_base)
    return env

class Robot(object):
    def __init__(self, env, policy_kwargs, device:torch.device=torch.device(0)):
        self.env = env
        #self.args = args
        self._dir_ckpt = None
        sac_args = {
            'learning_rate': 2e-5,
            'buffer_size': 10000,
            'train_freq': (10,'step'),
            'gradient_steps': 10,
            'target_update_interval': 1,
            'learning_starts': 10,
            'batch_size': 512,
            'policy_kwargs': policy_kwargs,
            'device': device,
        }
        self.sac = my_sac(
                policy="MultiInputPolicy", 
                env=env, 
                policy_base=SACPolicy, 
                **sac_args
        )
        self.check_ckpt()
        self.check_buffer()  

        if self._dir_ckpt is not None:
            self.sac = self.sac.load(path=self._dir_ckpt, env=env, buffer_path=self._dir_buffer, **sac_args)
        
        if self.sac._init_setup_model:    
            if self._dir_buffer:
                self.sac.load_replay_buffer(self._dir_buffer)     
            self.sac._setup_model()

    def learn(self, callback):
        self.sac._setup_learn(callback=callback, start_timestamp=self._start_timesteps)
        self.sac.learn(
            total_timesteps=1500000,
            callback=callback
        )

    def evluate_sac(self):
        video_path = 'sac_data/video/' + 'eval_{}.mp4'.format(self._start_timesteps)
        self.evaluate_policy(self.env, self.sac.policy, video_path)

    def check_ckpt(self, dir_path='sac_data/ckpt'):
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

    def check_buffer(self, dir_path='sac_data/buffer'):
        buffer_list = os.listdir(dir_path)
        if len(buffer_list) == 0:
            self._dir_buffer = None
            print('no exist buffer')
        else:
            self._dir_buffer = os.path.join(dir_path, buffer_list[0])
            #self.sac.load_replay_buffer(self._dir_buffer)

    @staticmethod
    def dict_to_tensor(obs:dict, device):
        return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}

    @staticmethod
    def evaluate_policy(env:CarlaEnv_mzq, policy: my_SACPolicy, video_path, min_eval_steps=100, is_use_wandb=False):
        device = torch.device('cuda:2')
        policy = policy.eval()
        actor = policy.actor
        critic = policy.critic
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()
        o = Robot.dict_to_tensor(obs, device)
        list_render = []

        n_step = 0
        env_done = np.array([False]*env.num_envs)
        if is_use_wandb:
            pbar = tqdm(initial=0)
        else:
            pbar = tqdm(initial=0, total=min_eval_steps)
        
        ep_stat_buffer = []
        while len(ep_stat_buffer) <10:
            mean_actions, log_std, kwargs= actor.get_action_dist_params(o)
            actions, log_probs = actor.action_dist.log_prob_from_params(mean_actions, log_std)
            value = min(critic(o, actions))
            actions = np.array(actions.detach().cpu())
            log_probs = np.array(log_probs.detach().cpu())
            mean_actions = np.array(mean_actions.detach().cpu())
            log_std = np.array(log_std.detach().cpu().exp())
            values = np.array(value.detach().cpu()).reshape(-1,)
            actions = np.array([[0.5, 0.0]])
            obs, reward, done, info = env.step(actions)
            o = Robot.dict_to_tensor(obs, device)

            for i in range(env.num_envs):
                env.set_attr('action', actions[i], indices=i)
                env.set_attr('action_value', values[i], indices=i)
                env.set_attr('action_log_probs', log_probs[i], indices=i)
                env.set_attr('action_mu', mean_actions[i], indices=i)
                env.set_attr('action_sigma', log_std[i], indices=i)
                
            list_render.append(env.render('rgb_array'))

            n_step += 1
            env_done |= done
            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['eval_info'])
            pbar.update(1)
        pbar.close()
        encoder = ImageEncoder(video_path, list_render[0].shape, 20, 20)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()

        eval_info = Robot.get_avg_ep_stat(ep_stat_buffer)
        with open('sac_data/info/test_eval_info.pkl','wb') as f:
            pickle.dump(eval_info, f)
        #if is_use_wandb:
        #    wandb.init(project='sac_eval', name=None, notes=None, tags=None)
            

    @staticmethod
    def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        '''
        n_collision = 0
        n_collision_layout = 0
        n_collisions_vehicle = 0
        n_collisions_walkers = 0
        n_collisions_other = 0
        n_run_red_light = 0
        route_compl = 0
        route_deviation = 0
        route_wrong = 0
        route_finish = 0
        '''
        avg_eval_stat = {}
        for eval_info in ep_stat_buffer:
            for k, v in eval_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_eval_stat:
                    avg_eval_stat[k_avg] += v
                else:
                    avg_eval_stat[k_avg] = v
        return avg_eval_stat
        
    
    
def propre_envconfig(num_env=2):
    env_configs = []
    for i in range(num_env):
        port = 2010 + i * 5
        env_cfg = {'gpu':0, 'port':port}
        env_configs.append(env_cfg)
    return env_configs
   
def main():
    cfg = {}
    cfg['carla_sh_path'] = '/home/ubuntu/carla-simulator/CarlaUE4.sh'
    env_configs = propre_envconfig(2)
    cfg['env_configs'] = env_configs
    device = torch.device(2)
    try:
        server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
        server_manager.start()
        time.sleep(15)

        #port_list = [2000]
        env_cfg1 = {
            'port':2000,
            'map_id':0,
            'device':device,
        }
        env_cfg2 = {
            'port':2005,
            'map_id':1,
            'device':device,
        }
        env_cfg = [env_cfg1, env_cfg2]
        if len(env_cfg) == 1:
            env = DummyVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        else:
            env = SubprocVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
        
        wandb_cfg = {
            'wb_project': 'sac_new',
            'wb_name': None,
            'wb_notes': None,
            'wb_tags': None
        }
        wb_callback = WandbCallback(wandb_cfg, env)
        callback = CallbackList([wb_callback])
        policy_args = {
            'features_extractor_entry_point': 'SAC.torch_layers:XtMaCNN', 
            'features_extractor_kwargs': {'states_neurons': [256, 256]}, 
            'use_sde': False,
            }
        robot = Robot(env, policy_args, device=device)
        robot.learn(callback=callback)

    #except:
    #    pid = os.getpid()
    #    subprocess.Popen('kill {}'.format(pid), shell=True)

    finally:
        env.close()
        server_manager.stop()
        
def eval():
    cfg = {}
    cfg['carla_sh_path'] = '/home/ubuntu/carla-simulator/CarlaUE4.sh'
    env_configs = propre_envconfig(num_env=1)
    cfg['env_configs'] = env_configs
    device = torch.device(2)
    server_manager = server_utils.CarlaServerManager(cfg['carla_sh_path'], configs=cfg['env_configs'])
    server_manager.start()
    time.sleep(15)
    env_cfg1 = {
            'port':2010,
            'map_id':1,
            'device':device,
        }
    env_cfg =  [env_cfg1]
    if len(env_cfg) == 1:
        env = DummyVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
    else:
        env = SubprocVecEnv([lambda port=port: env_make(**port) for port in env_cfg])
    args = {
            'features_extractor_entry_point': 'SAC.torch_layers:XtMaCNN', 
            'features_extractor_kwargs': {'states_neurons': [256, 256]}, 
            'use_sde': False,
        }
    robot = Robot(env=env, policy_kwargs=args, device=device)
    video_path = 'sac_data/eval_video/' + 'eval_{}.mp4'.format(robot._start_timesteps) 
    robot.evaluate_policy(env, robot.sac.policy, video_path, is_use_wandb=True)
    '''
    env 'mask' -> 'bev_mask' -> envwrapper parameter bev->'birdview' -> torch_layer 
    '''

if __name__ == '__main__':
    #main()
    eval()