from ctypes import util
from locale import currency
import os
from sqlite3 import Timestamp
from turtle import TurtleScreen, distance
import weakref
import carla
import collections
import math
from cv2 import sort, threshold
import shapely
import shapely.geometry
import numpy as np
from gym_carla.envs import utils
from gym_carla.envs.trafficlight import Trafficlight
from sklearn.metrics import top_k_accuracy_score
import os
import subprocess
'''
all_info(sample):

reward info(every tick):
    info['r_position'] = r_position
    info['r_speed'] = r_speed
    info['r_rotation'] = r_rotation
    info['r_action'] = r_action
    reward_info['r_terminal'] = r_terminal
    reward_info['r_arrived'] = r_arrived
    reward_info['r_redlight'] = r_redlight

collision info(only happend, None):
    self._collision_info = {
            'step': event.frame,
            'simulation_time': event.timestamp,
            'collision_type': collision_type,  
                0 is layout(except sidewalk), 1 is vehicle, 2 is walkers, -1 is others
            'other_actor_id': event.other_actor.id,
            'other_actor_type_id': event.other_actor.type_id,
            'intensity': intensity,
            'normal_impulse': [impulse.x, impulse.y, impulse.z],
            'event_loc': [event_loc.x, event_loc.y, event_loc.z],
            'event_rot': [event_rot.roll, event_rot.pitch, event_rot.yaw],
            'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z],
            'ev_rot': [ev_rot.roll, ev_rot.pitch, ev_rot.yaw],
            'ev_vel': [ev_vel.x, ev_vel.y, ev_vel.z],
            'oa_loc': [oa_loc.x, oa_loc.y, oa_loc.z],
            'oa_rot': [oa_rot.roll, oa_rot.pitch, oa_rot.yaw],
            'oa_vel': [oa_vel.x, oa_vel.y, oa_vel.z],
        }

runredlight info(only happend, None):
    self.info = {
                'step': timestamp['step'],
                'simulation_time': timestamp['relative_simulation_time'],
                'id': traffic_light.id,
                'tl_loc': [tl_loc.x, tl_loc.y, tl_loc.z],
                'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
                }

block info(only happend, None):
    self.info = {
                'step':timestamp['step'],
                'elapsed_time':elapsed_time,
                'loc': ev_loc
                }

route info(every tick):
    info['route_completed_in_wp_percensage'] = self._close_idx / (len(self._route_trace_wp) - 1)
    info['route_completed_in_m'] = self._completed_route_length
    info['route_completed_in_m_percentage'] = self._completed_route_length / self._full_route_length
    info['is_route_finished'] = self.is_route_finished()

'''

class Reward(object):
    def __init__(self, ego_vehicle, route_trace, trafficlight_manager: Trafficlight):
        pass

    def tick(self, timestamp):
        info = {'text':['haha']}
        return 1, False, info

    def destroy(self):
        pass