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
        self._ego_vehicle = ego_vehicle
        self._route_trace = route_trace
        # self._collision_sensor = None
        # self._block_sensor = None
        # self._route_manager = None
        self._all_info = None
        self._collision_sensor = CollisionSensor(self._ego_vehicle)
        self._block_sensor = Block(self._ego_vehicle)
        self._route_manager = Route(self._route_trace, self._ego_vehicle)
        self._runredlight_sensor = RunRedlightSensor(self._ego_vehicle, trafficlight_manager)
        self._last_steer = 0.0
        self._last_lateral = 0.0
        self._min_threshold_lat = 3.5

    def tick(self, timestamp):
        self._all_info = None
        # reward and terminal reward and arrived reward
        #self._route_trace = route_trace
        route_finished, route_info = self._route_manager.tick()
        if route_info['go_next']:
            r_route = 0.2
        else:
            r_route = 0
        route_deviation = route_info['is_deviation']
        route_wrong = route_info['is_wrong']
        if route_finished:
            r_arrived = 5.0
        else:
            r_arrived = 0.0

        r, r_text, reward_info = self.get_reward(timestamp=timestamp)
        runredlight_info = reward_info['runredlight_info']
        d, r_terminal, terminal_info = self.get_terminal(timestamp=timestamp)
        block_info = terminal_info['block_info']
        collision_info = terminal_info['collision_info']
        block_time = terminal_info['block_time']
        
        
        reward = r + r_terminal + r_arrived + r_route
        reward_info['r_terminal'] = r_terminal
        reward_info['r_arrived'] = r_arrived
        
        reward_info['r_total'] = reward
        done = d or route_finished or route_deviation or route_wrong
        text = [f'r_total:{reward:5.2f}, r_route:{r_route:5.2f}']
        # r_terminal:{r_terminal:5.2f}',  f'r_arrived:{r_arrived:5.2f},
        route_text = route_info['text']
        text = r_text+ text + route_text
        text.append(f'block_time:{block_time:5.2f}')
        self._all_info = {
            'route_info': route_info,
            'reward_info': reward_info,
            'runredlight_info': runredlight_info,
            'block_info': block_info,
            'collision_info': collision_info,
            'text':text,
        }
        return reward, done, self._all_info

    def reset(self):
        # useless. fuck it up!
        self.destroy()

    def destroy(self):
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
            self._collision_sensor = None
        if self._block_sensor is not None:
            self._block_sensor = None
        if self._route_manager is not None:
            self._route_manager = None
        if self._runredlight_sensor is not None:
            self._runredlight_sensor = None

    def get_reward(self, timestamp):
        info = {}
        ev_vel = self._ego_vehicle.get_velocity()
        ev_speed = self.cal_speed(ev_vel)
        ev_transform = self._ego_vehicle.get_transform()

        #set desired speed
        basic_speed = 6.0
        
        desired_speed_walker = basic_speed
        desired_speed = basic_speed
        # get nearby vehicles and walkers
        nearby_vehicle, nearby_walker = self.get_nearby(10,10)
        v_list = []
        for i in range(len(nearby_vehicle)):
            tmp_dist, tmp_vehicle = nearby_vehicle[i]
            flag = self._route_manager.is_ov_ow_blocked(tmp_vehicle)
            if flag:
                v_list.append(nearby_vehicle[i])
        
        if v_list:
            tmp_distance_list = [v[0] for v in v_list]
            tmp_distance = max(0.0, min(tmp_distance_list) - 8.0)
            desired_speed_vehicle = basic_speed * np.clip(tmp_distance, 0.0, 5.0) / 5.0
        else:
            desired_speed_vehicle = basic_speed 
        runredlight_info, desired_speed_tl_list = self._runredlight_sensor.tick(timestamp=timestamp)

        if desired_speed_tl_list:
            desired_speed_tl = min(desired_speed_tl_list)
        else:
            desired_speed_tl = basic_speed

        if runredlight_info:
            r_redlight = - 2.0
        else:
            r_redlight = 0.0
        '''
        if len(nearby_vehicle) != 0:
            tmp_distance = max(0.0, sorted(nearby_vehicle)[0][0] - 8.0)
            desired_speed_vehicle = basic_speed * np.clip(tmp_distance, 0.0, 5.0) / 5.0
        if len(nearby_walker) != 0:
            tmp_distance = max(0.0, sorted(nearby_walker)[0][0] - 6.0)
            desired_speed_walker = basic_speed * np.clip(tmp_distance, 0.0, 5.0) / 5.0
        # desired_speed = min(basic_speed, desired_speed_vehicle, desired_speed_walker)
        desired_speed = min(desired_speed_vehicle, desired_speed_walker)
        '''
        # reward speed
        
        desired_speed = min(desired_speed_tl, basic_speed, desired_speed_vehicle)
        r_speed = 1 - np.abs(ev_speed - desired_speed) / basic_speed

        

        '''
        if ev_speed > basic_speed:
            r_speed = 1.0 - np.abs(ev_speed - desired_speed) / basic_speed
        else:
            r_speed = (1.0 - np.abs(ev_speed - desired_speed) / basic_speed) / 2
        '''
        # r_position
        wp_transform = self._route_manager.get_current_wp_transform()
        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_left = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)
        lateral_distance = np.abs(np.dot(np_wp_unit_left, np_d_vec))
        self.lateral_dist = lateral_distance
        r_position = -1.0 * (lateral_distance / 4.0)

        #reward rotation
        angle_difference = np.deg2rad(np.abs(utils.cast_angle(
            ev_transform.rotation.yaw - wp_transform.rotation.yaw)))
        r_rotation = -1.0 * angle_difference / 2.0
        
        # reward action
        ev_control = self._ego_vehicle.get_control()
        if abs(ev_control.steer - self._last_steer) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        self._last_steer = ev_control.steer

        r = r_position + r_speed + r_rotation + r_action + r_redlight

        # reward info
        info['r_position'] = r_position
        info['r_speed'] = r_speed
        info['r_rotation'] = r_rotation
        info['r_action'] = r_action
        info['r_redlight'] = r_redlight
        info['runredlight_info'] = runredlight_info  
        reward_text = [ f'r_redlight:{r_redlight:5.2f}, r_pos:{r_position:5.2f}', 
                        f'des_sped:{desired_speed:5.2f}, r_rot:{r_rotation:5.2f}',
                        f'r_speed:{r_speed:5.2f}, r_act:{r_action:5.2f}']
        return r, reward_text, info

    def get_terminal(self, timestamp):
        info = None
        terminal = None
        # terminal condition 1: collision
        if self._collision_sensor is None:
            print('lost collision sensor')
            return True
        collision_info = self._collision_sensor.tick(self._ego_vehicle, timestamp)
        collision_condition = collision_info is not None

        # terminal condition 2: block
        if self._block_sensor is None:
            print('lost block sensor')
            return True
        block_info = self._block_sensor.tick(timestamp)
        blocktime = self._block_sensor.get_blocktime()
        block_condition = block_info is not None

        # terminal condition 2: lateral so much
        if self.lateral_dist - self._last_lateral > 0.8:
            threshold_lat = self.lateral_dist + 0.5
        else:
            threshold_lat = max(self._min_threshold_lat, self._last_lateral)
        lateral_condition = (self.lateral_dist > threshold_lat + 1e-2)
        self._last_lateral = self.lateral_dist

        terminal = collision_condition or block_condition or lateral_condition
        reward = 0.0
        if terminal:
            reward = - 1.0
            if collision_condition:
                ev_velocity = self._ego_vehicle.get_velocity()
                ev_speed = self.cal_speed(ev_velocity)
                reward += - ev_speed/2.0

        info = {
            'collision_info': collision_info,
            'block_info': block_info,
            'block_time':blocktime,
            'lateral_condition':None,

        }
        return terminal, reward, info

    def encounter_traffic_light(self, timestamp):
        pass

    def get_nearby(self, vehicle_distance=20, walker_distance=10):
        # get vehicle
        nearby_vehicle = []
        nearby_walker = []
        world = self._ego_vehicle.get_world()
        all_vehicles = world.get_actors().filter('vehicle.*')
        all_walkers = world.get_actors().filter('walker.pedestrian.*')
        ev_loc = self._ego_vehicle.get_location()
        for vehicle in all_vehicles:
            vehicle_loc = vehicle.get_location()
            if abs(ev_loc.x - vehicle_loc.x) > 10 or abs(ev_loc.y - vehicle_loc.y) > 10 or abs(ev_loc.z - vehicle_loc.z) > 5:
                continue 
            tmp_dist = self.cal_dist(vehicle_loc)
            if tmp_dist < vehicle_distance:
                nearby_vehicle.append((tmp_dist, vehicle)) 
        for walker in all_walkers:
            walker_loc = walker.get_location()
            if abs(ev_loc.x - walker_loc.x) > 10 or abs(ev_loc.y - walker_loc.y) > 10 or abs(ev_loc.z - walker_loc.z) > 5:
                continue 
            tmp_dist = self.cal_dist(walker_loc)
            if tmp_dist < walker_distance:
                nearby_walker.append((tmp_dist, walker))
        # get the actors that block the route
        
        return nearby_vehicle, nearby_walker
    
    def cal_dist(self, vehicle_loc):
        ev_loc = self._ego_vehicle.get_location()
        return np.sqrt((ev_loc.x - vehicle_loc.x) ** 2 + (ev_loc.y - vehicle_loc.y) ** 2 + (ev_loc.z - vehicle_loc.z) ** 2)

    @staticmethod
    def cal_speed(velocity):
        return np.linalg.norm([velocity.x, velocity.y])

class CollisionSensor(object):
    def __init__(self, parent_actor, intensity_threshold=0.0):
        self.sensor = None
        self.last_id = None
        self.history = []
        self.registered_collisions = []
        self._intensity_threshold = intensity_threshold
        self._collision_info = None
        self._parent = parent_actor
        self._max_area_of_collision = 5.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)

        # each time collision against something, the sensor will get event
        self.sensor.listen(lambda event: self._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()
        self.sensor = None

    def tick(self, ego_vehicle, timestamp):
        ev_loc = ego_vehicle.get_location()
        new_registered_collisions = []
        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:
            distance = ev_loc.distance(collision_location)
            # If far away from a previous collision, forget it
            if distance <= self._max_area_of_collision:
                new_registered_collisions.append(collision_location)

        self.registered_collisions = new_registered_collisions

        if self.last_id and timestamp['relative_simulation_time'] - self.collision_time > self._max_id_time:
            self.last_id = None

        info = self._collision_info
        self._collision_info = None
        if info is not None:
            info['step'] -= timestamp['start_frame']
            info['simulation_time'] -= timestamp['start_simulation_time']
        return info

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # generate impulse for vision
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        if intensity < self._intensity_threshold:
            return
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        
        # Ignore some situations, such as too close
        if self.last_id == event.other_actor.id:
            return
        egovehicle_loc = event.actor.get_location()
        for collision_location in self.registered_collisions:
            if egovehicle_loc.distance(collision_location) <= self._min_area_of_collision:
                return

        # type of collision
        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
            and 'sidewalk' not in event.other_actor.type_id:
            collision_type = 0    # collision static
        elif 'vehicle' in event.other_actor.type_id:
            collision_type = 1    # collision vehicle
        elif 'walker' in event.other_actor.type_id:
            collision_type = 2    # collision walker
        else:
            collision_type = -1   # other collision
        
        # write to info, all quantities in in world coordinate
        event_loc = event.transform.location
        event_rot = event.transform.rotation
        oa_loc = event.other_actor.get_transform().location
        oa_rot = event.other_actor.get_transform().rotation
        oa_vel = event.other_actor.get_velocity()
        ev_loc = event.actor.get_transform().location
        ev_rot = event.actor.get_transform().rotation
        ev_vel = event.actor.get_velocity()
        
        self._collision_info = {
            'step': event.frame,
            'simulation_time': event.timestamp,
            'collision_type': collision_type,
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

class RunRedlightSensor(object):

    def __init__(self, ego_vehicle, trafficlight_manager, distance_light=30):
        self._ego_vehicle = ego_vehicle
        self._world = self._ego_vehicle.get_world()
        self._map = self._world.get_map()
        self._last_red_light_id = None
        self._trafficlight_manager = trafficlight_manager
        self._distance_light = distance_light
        self.info = None
        tl_num, tl_actor, tv_loc, swp, sv = self._trafficlight_manager.get_parameter()

    def tick(self, timestamp):
        '''
        1. nearby traffic light
        2. red light
        3. is affect us
        4. is run red light
        5. desired speed tell the ego to stop
        '''
        self.info = None
        ev_tranform = self._ego_vehicle.get_transform()
        ev_loc = ev_tranform.location
        ev_dir = ev_tranform.get_forward_vector()
        ev_extent = self._ego_vehicle.bounding_box.extent.x

        # tail
        
        #tail_close_pt = ev_tranform.transform(carla.Location(x=-0.8 * ev_extent))
        #tail_far_pt = ev_tranform.transform(carla.Location(x=-ev_extent - 1.0))
        #tail_wp = self._map.get_waypoint(tail_far_pt)
        
        # top 
        #top_close_pt = ev_tranform.transform(carla.Location(x=0.8 * ev_extent))
        #top_far_pt = ev_tranform.transform(carla.Location(x=ev_extent + 1.0))
        #top_wp = self._map.get_waypoint(top_far_pt)

        # top & tail
        top_pt = ev_tranform.transform(carla.Location(x=ev_extent + 1))
        tail_pt = ev_tranform.transform(carla.Location(x=-ev_extent - 1.0))
        tail_wp = self._map.get_waypoint(tail_pt)

        top_close_pt = top_pt
        top_far_pt = tail_pt
        top_wp = tail_wp


        # prepare for desired speed
        basic_speed = 6.0
        desired_speed = []

        # find nearby red traffic light
        ev_loc = self._ego_vehicle.get_location()
        for i in range(self._trafficlight_manager.tl_num):
            traffic_light = self._trafficlight_manager.tl_actor[i]
            tv_loc = self._trafficlight_manager.tv_loc[i]
            # trigger_volume is more precise than base_transform
            if tv_loc.distance(ev_loc) > self._distance_light:
                continue
            # red
            if traffic_light.state != carla.TrafficLightState.Red and traffic_light.state != carla.TrafficLightState.Yellow:
                continue
            # ignore repeated red light
            #if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
             #   continue
            
            # swp = stop line way points
            # sv = stop vertices
            for j in range(len(self._trafficlight_manager.swp[i])):
                waypoint = self._trafficlight_manager.swp[i][j]
                waypoint_dir = waypoint.transform.get_forward_vector()
                dot_dir = ev_dir.x * waypoint_dir.x + ev_dir.y * waypoint_dir.y

                if top_wp.road_id == waypoint.road_id and top_wp.lane_id == waypoint.lane_id and dot_dir > 0:
                    # affect us
                    left_point, right_point = self._trafficlight_manager.sv[i][j]
                    
                    tmp_distance = max(0.0, tv_loc.distance(ev_loc) - 5)
                    tmp_desired_speed = basic_speed * np.clip(tmp_distance, 0.0, 5.0) / 5.0
                    desired_speed.append(tmp_desired_speed)
                    
                    if self._is_vehicle_crossing_line((top_close_pt, top_far_pt), (left_point, right_point)):
                        tl_loc = traffic_light.get_location()
                        # loc_in_ev = trans_utils.loc_global_to_ref(tl_loc, ev_tra)
                        self._last_red_light_id = traffic_light.id
                        self.info = {
                            'step': timestamp['step'],
                            'simulation_time': timestamp['relative_simulation_time'],
                            'id': traffic_light.id,
                            'tl_loc': [tl_loc.x, tl_loc.y, tl_loc.z],
                            'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
                        }
        return self.info, desired_speed

    def get_info(self):
        return self.info
    
    @staticmethod
    def _is_vehicle_crossing_line(seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)
        return not inter.is_empty



        

class Block(object):
    def __init__(self, ego_vehicle, speed_threshold=0.1, max_time=90.0):
        self._ego_vehicle = ego_vehicle
        self.speed_threshold = speed_threshold
        self.max_time = max_time
        self.last_valid_time = 0
        self.info = None
        self.blocktime = 0

    def tick(self, timestamp):
        self.info = None
        self.blocktime = 0
        ego_vel = self._ego_vehicle.get_velocity()
        ego_speed = self.cal_speed(ego_vel)
        elapsed_time = timestamp['relative_simulation_time']
        if ego_speed < self.speed_threshold:
            self.blocktime = elapsed_time - self.last_valid_time
            if elapsed_time - self.last_valid_time > self.max_time:
                # block
                ev_loc = self._ego_vehicle.get_location()
                self.info = {
                    'step':timestamp['step'],
                    'elapsed_time':elapsed_time,
                }
        else:
            self.last_valid_time = elapsed_time
        return self.info

    def get_blocktime(self):
        return self.blocktime

    @staticmethod
    def cal_speed(velocity):
        return np.linalg.norm([velocity.x, velocity.y])

class Route(object):
    def __init__(self, route_trace, ego_vehicle):
        self._route_trace = route_trace
        self._route_trace_wp = self.get_route_trace_wp()
        
        self._ego_vehicle = ego_vehicle
        self._close_idx = 0
        self._current_route_trace_wp = self._route_trace_wp
        self._full_route_length = self.cal_route_length(self._route_trace_wp)
        self._completed_route_length = 0
        self._out_route_length = 0
        self._route_wrong = False
        if len(self._route_trace_wp) < 2 or self._full_route_length==0:
            f = open('2.txt','a')
            ev_loc = self._ego_vehicle.get_location()
            f.write('start: ({},{}) \n'.format(round(ev_loc.x), round(ev_loc.y)))
            f.close()
            world = ego_vehicle.get_world()
            carla_map = world.get_map()
            self._route_trace_wp = [carla_map.get_waypoint(ego_vehicle.get_location())]
            self._full_route_length = 1
            self._route_wrong = True
            #pid = os.getpid()
            #subprocess.Popen('kill {}'.format(pid), shell=True)
            #exit(2)
        
    def tick(self):
        # output(bool): is_completed_route 
        flag = self.go_next_way_point()  # close idx ++ and update current route
        info = self.get_info()
        info['is_deviation'] = self.is_deviation()
        info['is_wrong'] = self._route_wrong
        info['go_next'] = flag
        return self.is_route_finished(), info
        
    def is_route_finished(self, dist_threshold=5.0):
        idx_finished = (self._close_idx >= len(self._route_trace_wp) - 2)
        ev_loc = self._ego_vehicle.get_location()
        destination_loc = self._route_trace_wp[-1].transform.location
        destination_arrived = destination_loc.distance(ev_loc) < dist_threshold
        return destination_arrived
    
    def get_info(self):
        info = {}
        info['route_completed_in_wp_percensage'] = route_comp_wpper = self._close_idx / len(self._route_trace_wp)
        info['route_completed_in_m'] = r_comp_m = self._completed_route_length
        info['route_completed_in_m_percentage'] = self._completed_route_length / self._full_route_length
        info['full_route_length_in_m'] = self._full_route_length
        info['is_route_finished'] = self.is_route_finished()
        next_wp_loc = self._current_route_trace_wp[0].transform.location
        origin_wp_loc = self._route_trace_wp[0].transform.location
        info['text'] = [f'route_m:{r_comp_m:5.2f}', f'route_wpper:{route_comp_wpper:5.2f}',
                         f'next_wp_loc: {next_wp_loc.x:5.2f}, {next_wp_loc.y:5.2f}', 
                         f'origin_wp_loc: {origin_wp_loc.x:5.2f}, {origin_wp_loc.y:5.2f}']
        return info
        
    def get_close_idx(self):
        return self._close_idx

    def get_current_wp(self):
        # basic route
        return self._route_trace_wp[self._close_idx]

    def get_current_wp_transform(self): 
        # current route
        return self._current_route_trace_wp[0].transform

    def get_route_trace_wp(self):
        return [way_point for way_point, _ in self._route_trace]

    def go_next_way_point(self, windows_size=5):
        ev_loc = self._ego_vehicle.get_location()
        # to next way point
        for i in range(windows_size):
            if i + self._close_idx + 1 >= len(self._route_trace_wp):
                break
            current_way_point = self._route_trace_wp[self._close_idx + i]
            next_way_point = self._route_trace_wp[self._close_idx + i + 1]
            wp_direct = next_way_point.transform.location - current_way_point.transform.location 
            wp_vehicle = ev_loc - current_way_point.transform.location
            dot_vec = wp_direct.x * wp_vehicle.x + wp_direct.y * wp_vehicle.y + wp_direct.z * wp_vehicle.z
            if dot_vec > 0:
                self._close_idx += 1+i
                # complete route
                complete_length = self.cal_route_length(self._route_trace_wp[self._close_idx-1: self._close_idx+1])
                self._completed_route_length += complete_length 
                self._current_route_trace_wp = self._route_trace_wp[self._close_idx:]
                return True
        return False

    def is_deviation(self, max_dist=30, min_dist=15):
        ev_loc = self._ego_vehicle.get_location()
        current_way_point = self._route_trace_wp[self._close_idx]
        wp_transform = current_way_point.transform
        dist = ev_loc.distance(wp_transform.location)
        off_route = dist > max_dist

        return off_route

    def is_ov_ow_blocked(self, actor, windows_size=10, thr=10):
        a_loc = actor.get_location()
        ev_loc = self._ego_vehicle.get_location()
        current_dist = thr + 1
        for i in range(windows_size):
            if i + self._close_idx + 1 >= len(self._route_trace_wp):
                break
            current_way_point = self._route_trace_wp[self._close_idx + i]
            next_way_point = self._route_trace_wp[self._close_idx + i + 1]
            current_dist = a_loc.distance(current_way_point.transform.location)
            next_dist = a_loc.distance(next_way_point.transform.location)

            if next_dist < current_dist: 
                continue
            else:
                break
        
        if current_dist > thr:
            return False
        forward_dir = next_way_point.transform.location - current_way_point.transform.location
        a_transform = actor.get_transform()
        a_dir = a_transform.get_forward_vector()
        pos_dir = a_loc - ev_loc
        return (forward_dir.x * a_dir.x + forward_dir.y * a_dir.y > 0) and (forward_dir.x * pos_dir.x + forward_dir.y * pos_dir.y > 0)
            
        





    @staticmethod
    def cal_route_length(route_wp):
        length_in_m = 0.0
        for i in range(len(route_wp) - 1):
            d = route_wp[i].transform.location.distance(route_wp[i+1].transform.location)
            length_in_m += d
        return length_in_m