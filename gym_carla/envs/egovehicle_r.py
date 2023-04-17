import random
import sys
from collections import deque
from distutils.log import error
from queue import Queue
from telnetlib import WONT

import cv2 as cv
import h5py
import numpy as np
from gym_carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from gym_carla.envs.trafficlight import Trafficlight
from gym_carla.envs.task import Task
from matplotlib.transforms import Bbox
from gym_carla.envs import utils
import time

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (187, 187, 186)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return r, g, b


class EgoVehicle(object):
    def __init__(self, client, port, multi_cam=False, num_des=10, is_meta=False):
        self._actor_list = []
        self._other_camera_list = []
        self._client = client
        self._world = client.get_world()
        self._map = self._world.get_map()
        self._multi_cam = multi_cam
        self._num_des = num_des
        self._sensor_info = {}
        # test tm
        self._tm_port = port + 6000
        self._traffic_manager = client.get_trafficmanager(self._tm_port)

        # vehichle
        self._spawn_points = list(self._map.get_spawn_points())
        self._blueprint_library = self._world.get_blueprint_library()
        self._vehicle_bp = self._create_vehicle_blueprint()
        self._ego_vehicle = self._spawn_ego_vehicle()
        self._auto = False

        # Sensors
        self._camera = self._create_camera()
        self._lidar = self._create_lidar()
        self._GPS = self._create_GPS()
        #self._IMU = self._create_IMU()
        self.sensor_queue = None

        # BEV
        self._width = 192
        self._road = None
        self._lane_marking_all = None
        self._lane_marking_white_broken = None
        self._history_queue = None
        self._pixels_per_meter = 0
        self._world_offset = 0
        self._pixels_ev_to_bottom = 0
        self._history_idx = None
        self._start_wp_ind = 0

        # route 
        self._route_planner = GlobalRoutePlanner(self._map, 1)
        self._is_des = None

        #mode
        self.is_meta = is_meta
 
    def reset(self):
        # destroy
        if not self._actor_list:
            pass
        else:
            self.destroy()
            self._actor_list = []
        self._ego_vehicle = None
        self._camera = None
        self._other_camera_list = []
        self._lidar = None
        #self.sensor_queue = None
        self.sensor_queue = Queue(maxsize=10)
        self._start_wp_ind = 0
        self._sensor_info = {}

        # create vehicle and sensors(camera, lidar)
        self._ego_vehicle = self._spawn_ego_vehicle()
        self._task_handle = Task(self._ego_vehicle, self._client, num_des=1)
        self._trafficlight_manager = Trafficlight(self._client)
        self._vehicle_bp = self._create_vehicle_blueprint()
        self._camera = self._create_camera()
        self.get_route()
        self._lidar = self._create_lidar()
        self._GPS = self._create_GPS()
        #self._IMU = self._create_IMU()
        if self._multi_cam:
            self._crerate_other_camera()
        #self.gnss_array = np.zeros(shape=(1000, 3), dtype='float64')
        #self.imu_array = np.zeros(shape=(1000, 7), dtype='float32')

        # keep listening
        self._start_listening()

    def destroy(self):
        # destroy all
        if self._actor_list:
            for x in self._actor_list:
                try:
                    x.destroy()
                finally:
                    pass
            self._actor_list = []
            #print('destroy ego vehicle and sensors')
        else:
            pass
            #print('no actors to destroy')
        if self._history_queue is not None:
            self._history_queue.clear()
        if self.sensor_queue is not None:
            self.sensor_queue.queue.clear()
        self._other_camera_list = []
        self._sensor_info = {}

    def set_vehicle_bp(self, bp_name='vehicle.mercedes.coupe_2020'):
        self._vehicle_bp = self._blueprint_library.find(bp_name)
        
    # ==============================================================================
    # -- spawn, route and control -----------------------------------------------------------
    # ==============================================================================
           
    def apply_control(self, control):
        self._ego_vehicle.apply_control(control)
    
    def get_control(self):
        return self._ego_vehicle.get_control()

    def get_spawn_point(self):
        return self._spawn_points
        
    def get_blueprint_library(self):
        return self._blueprint_library

    def get_vehicle_blueprint(self):
        return self._vehicle_bp
    
    def get_ego_vehicle(self):
        return self._ego_vehicle
    
    def get_trafficlight_manager(self):
        return self._trafficlight_manager

    def _create_vehicle_blueprint(self, ev_filter='vehicle'):
        vehicle_bp = self._blueprint_library.filter(ev_filter)
        is_bike = True
        bp = random.choice(vehicle_bp)
        while is_bike:
            bp = random.choice(vehicle_bp)
            is_bike = bp.get_attribute('number_of_wheels').as_int() == 2
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _spawn_ego_vehicle(self):
        self.set_vehicle_bp()
        spawn_transform = random.choice(self._spawn_points)
        ego_vehicle = self._world.try_spawn_actor(self._vehicle_bp, spawn_transform)
        while ego_vehicle is None:
            spawn_transform = random.choice(self._spawn_points)
            ego_vehicle = self._world.try_spawn_actor(self._vehicle_bp, spawn_transform)
        self._world.tick()
        self._actor_list.append(ego_vehicle)
        # print('created %s' % ego_vehicle.type_id)
        return ego_vehicle

    def auto_control(self):
        self._ego_vehicle.set_autopilot(True)
        self._auto = True
    
    def get_auto_state(self):
        return self._auto
    
    def get_route_trace(self):
        return self._route_trace

    def get_route(self):
        start_locaiton = self._ego_vehicle.get_location()
        while start_locaiton.x==0 and start_locaiton.y==0 and start_locaiton.z==0:
            start_locaiton = self._ego_vehicle.get_location()
            self._world.tick()
        #time.sleep(0.5)     wait for ev loc, it is important for trace route
      
        self._des = self._task_handle.gen_task_des()
        
        self._is_des = True
        self._route_trace = self._route_planner.trace_route(start_locaiton, self._des)
        '''
        the following code is used to get correct des, only calculate route once may get some errors
        but we get calculate route every step to avoid those errors

        # 1、des loc too close
        while start_locaiton.distance(des_location) < 10:
            #print('dist less than 10')
            des_list = self._task_handle.gen_task_des()
            des_location = des_list[0]
        #des_location.x = round(des_location.x)
        #des_location.y = round(des_location.y)
        self._route_trace = self._route_planner.trace_route(start_locaiton, des_location)
        start_wp, _ = self._route_trace[0]
        
        # 2、route deviation
        if start_locaiton.distance(start_wp.transform.location) > 10:
            f = open('1.txt', 'a')
            f.write('start: ({},{})  '.format(start_locaiton.x, start_locaiton.y))
            f.write('des: ({},{})\n'.format(des_location.x, des_location.y))
            f.write('   start_wp: ({},{})\n'.format(start_wp.transform.location.x, start_wp.transform.location.y))
            des_list = self._task_handle.gen_task_des()
            des_location = des_list[0]
            self._route_trace = self._route_planner.trace_route(start_locaiton, des_location)
            f.close()
            #des_location.x = round(des_location.x)
            #des_location.y = round(des_location.y)
            #print('change des')
        
        # route wrong
        if len(self._route_trace) < 2:
            f = open('3.txt', 'a')
            f.write('start: ({},{})  '.format(start_locaiton.x, start_locaiton.y))
            f.write('des: ({},{})\n'.format(des_location.x, des_location.y))
            des_list = self._task_handle.gen_task_des()
            des_location = des_list[0]
            self._route_trace = self._route_planner.trace_route(start_locaiton, des_location)
            f.close()
        '''
        
    def get_state(self):
        velocity = self._ego_vehicle.get_velocity()
        control = self._ego_vehicle.get_control()
        return velocity, control

    # ==============================================================================
    # -- Sensors_observation -----------------------------------------------------------
    # ==============================================================================

    def set_transform(self, bbox):
        transform_front_mid = carla.Transform(carla.Location(x=1.1*bbox.x, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=0,roll=0.0))
        transform_front_right = carla.Transform(carla.Location(x=1.1*bbox.x, y=1.1*bbox.y, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=45,roll=0.0))
        transform_front_left = carla.Transform(carla.Location(x=1.1*bbox.x, y=-1.1*bbox.y, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=-45,roll=0.0))
        transform_back = carla.Transform(carla.Location(x=-1.1*bbox.x, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=180,roll=0.0))
        transform_right = carla.Transform(carla.Location(y=1.1*bbox.y, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=90,roll=0.0))
        transform_left = carla.Transform(carla.Location(y=-1.1*bbox.y, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=-90,roll=0.0))
        transform_list = [transform_front_mid, transform_front_right, transform_front_left, transform_back, transform_right, transform_left]
        return transform_list

    def _crerate_other_camera(self, c_type='rgb'):
        bbox = self._ego_vehicle.bounding_box.extent
        attachment = carla.AttachmentType
        camera_bp = self._blueprint_library.find('sensor.camera.{}'.format(c_type))
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('image_size_x', '1600')
        camera_bp.set_attribute('image_size_y', '900')
        transform_list = self.set_transform(bbox)
        for i in range(1,6):
            transform = transform_list[i]
            camera = self._world.spawn_actor(camera_bp, transform, attach_to=self._ego_vehicle,\
                attachment_type=attachment.Rigid)
            self._other_camera_list.append(camera)
            #camera_info = utils.get_camera_info(camera)
            #self._sensor_info['camera_{}'.format(i)] = camera_info
        self._actor_list.extend(self._other_camera_list)
                

    def _create_camera(self, c_type='rgb'):
        bbox = self._ego_vehicle.bounding_box.extent
        attachment = carla.AttachmentType
        camera_bp = self._blueprint_library.find('sensor.camera.{}'.format(c_type))
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('image_size_x', '1600')
        camera_bp.set_attribute('image_size_y', '900')
        #camera_transform = carla.Transform(carla.Location(x=-8, z=2.5), carla.Rotation(pitch=8.0))
        #camera_transform = carla.Transform(carla.Location(x=0.9*bbox.x, z=1.3*bbox.z), carla.Rotation(pitch=-10.0,yaw=230,roll=-25.0))
        camera_transform = carla.Transform(carla.Location(x=1.1*bbox.x, z=1.3*bbox.z), carla.Rotation(pitch=0.0,yaw=0.0,roll=0.0))
        camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=self._ego_vehicle,
                                         attachment_type=attachment.Rigid)
        self._actor_list.append(camera)
        #camera_info = utils.get_camera_info(camera)
        #self._sensor_info['camera_{}'.format(c_type)] = camera_info
        return camera

    def _create_lidar(self):
        lidar_bp = self._blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', str(10*56000))
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.1))
        lidar = self._world.spawn_actor(lidar_bp, lidar_transform, attach_to=self._ego_vehicle)
        self._actor_list.append(lidar)
        #lidar_info = utils.get_lidar_info(lidar)
        #self._sensor_info['lidar'] = lidar_info
        return lidar

    def _create_GPS(self):
        GPS_bp = self._blueprint_library.find('sensor.other.gnss')
        GPS_transform = carla.Transform(carla.Location(x=0.0, z=2.8))
        GPS = self._world.spawn_actor(GPS_bp, GPS_transform, attach_to=self._ego_vehicle)
        self._actor_list.append(GPS)
        return GPS

    def _create_IMU(self):
        IMU_bp = self._blueprint_library.find('sensor.other.imu')
        IMU_transform = carla.Transform()
        IMU = self._world.spawn_actor(IMU_bp, IMU_transform, attach_to=self._ego_vehicle)
        self._actor_list.append(IMU)
        return IMU
    
    def _start_listening(self):
        self._camera.listen(lambda data: self.sensor_callback(data, self.sensor_queue, "camera"))
        self._lidar.listen(lambda data: self.sensor_callback(data, self.sensor_queue, "lidar"))
        self._GPS.listen(lambda data: self.sensor_callback(data, self.sensor_queue, "GPS"))
        #self._IMU.listen(lambda data: self.sensor_callback(data, self.sensor_queue, "IMU"))
        if self._multi_cam:
            self._other_camera_list[0].listen(lambda data: self.sensor_callback(data, self.sensor_queue, "camera_1"))
            self._other_camera_list[1].listen(lambda data: self.sensor_callback(data, self.sensor_queue, "camera_2"))
            self._other_camera_list[2].listen(lambda data: self.sensor_callback(data, self.sensor_queue, "camera_3"))
            self._other_camera_list[3].listen(lambda data: self.sensor_callback(data, self.sensor_queue, "camera_4"))
            self._other_camera_list[4].listen(lambda data: self.sensor_callback(data, self.sensor_queue, "camera_5"))


    @staticmethod
    def sensor_callback(sensor_data, sensor_queue, sensor_name):
        # Do stuff with the sensor_data data like save it to disk
        # Then you just need to add to the queue
        sensor_queue.put((sensor_data, sensor_name))
        '''
        if sensor_name == 'camera':
            array = utils.preprocess_img(sensor_data)
            sensor_queue.put((array, sensor_name))
        elif sensor_name == 'lidar':
            array = utils.preprocess_lidar(sensor_data)
            sensor_queue.put((array, sensor_name))
        else:
            sensor_queue.put((sensor_data, sensor_name))
        '''
    # ==============================================================================
    # -- BEV_observation -----------------------------------------------------------
    # ==============================================================================
    def init_bev_observation(self, path):
        with h5py.File(path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
            self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)

            self._sidewalk = np.array(hf['sidewalk'], dtype=np.uint8)
            self._shoulder = np.array(hf['shoulder'], dtype=np.uint8)
            self._parking = np.array(hf['parking'], dtype=np.uint8)

            self._history_queue = deque(maxlen=20)
            self._ev_history_queue = deque(maxlen=20)
            self._pixels_per_meter = 5.0
            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            self._pixels_ev_to_bottom = 40.0
            self._history_idx = [-16, -11, -6, -1]
            if self.is_meta:
                self._ev_history_idx = [-5,-4,-3,-2,-1]
            else:
                self._ev_history_idx = [-1]
            self._scale_mask_col = 1.0

    def get_bev_observation(self, distance_threshold, scale_bbox):

        ev_transform = self._ego_vehicle.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self._ego_vehicle.bounding_box
        # snap_shot = self._world.get_snapshot()

        # judge if within distance and get the surrounding vehicles or walkers
        def is_within_distance(w):
            c_distance = abs(ev_loc.x - w.location.x) < distance_threshold \
                         and abs(ev_loc.y - w.location.y) < distance_threshold \
                         and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        # if scale the bbox
        if scale_bbox:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
        else:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)
        tl_num, tl_actor, tv_loc, swp, sv = self._trafficlight_manager.get_parameter()
        tl_green = self._trafficlight_manager.get_stop_line_vtx(ev_loc, 0, tl_num, tl_actor, tv_loc, swp, sv)
        tl_yellow = self._trafficlight_manager.get_stop_line_vtx(ev_loc, 1, tl_num, tl_actor, tv_loc, swp, sv)
        tl_red = self._trafficlight_manager.get_stop_line_vtx(ev_loc, 2, tl_num, tl_actor, tv_loc, swp, sv)
        # stops = self._get_stops(self._ego_vehicle.criteria_stop)

        # self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))
        ev_tuple = [(ev_transform, carla.Location(), carla.Vector3D(ev_bbox.extent))]
        self._history_queue.append((vehicles, walkers, ev_tuple, tl_green, tl_yellow, tl_red))
        # self._ev_history_queue.append(ev_bbox)

        # the upper 'history_queue' is list sequence and all location is global.
        # the latter 'masks' are history local location, using 'M_warp' to project
        # t time sampled 'mask's  
        M_warp = self._get_warp_transform(ev_loc, ev_rot)
        self.M_warp = M_warp

        # objects with history
        vehicle_masks, walker_masks, ev_histroy_masks, tl_green_masks, tl_yellow_masks, tl_red_masks = \
            self._get_history_masks(M_warp)

        
        # road_mask, lane_mask, projection from global mask to local mask
        road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
                                         (self._width, self._width)).astype(np.bool)
        sidewalk_mask = cv.warpAffine(self._sidewalk, M_warp, (self._width, self._width)).astype(np.bool)
        shoulder_mask = cv.warpAffine(self._shoulder, M_warp, (self._width, self._width)).astype(np.bool)
        parking_mask = cv.warpAffine(self._parking, M_warp, (self._width, self._width)).astype(np.bool)

        # route_mask
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        def prepro_route_trace(route_trace, ev_loc, curr_wp_ind, windows_size):
            for i in range(windows_size):
                if i + curr_wp_ind + 1 >= len(route_trace):
                    break
                curr_wp, _ = route_trace[curr_wp_ind]
                next_wp, _ = route_trace[curr_wp_ind + 1]
                wp_direct = next_wp.transform.location - curr_wp.transform.location
                wp_vehicle = ev_loc - curr_wp.transform.location
                dot_vec = wp_direct.x * wp_vehicle.x + wp_direct.y * wp_vehicle.y
                if dot_vec > 0:
                    curr_wp_ind += 1+i
                    break
            return curr_wp_ind, len(route_trace) - curr_wp_ind
        self._start_wp_ind, rest_wp_number = prepro_route_trace(self._route_trace, ev_loc, self._start_wp_ind, 10)
        
        rest_wp_number = min(30, rest_wp_number)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)] for wp, _ in self._route_trace[self._start_wp_ind :]])
        #print('hh',curr_wp_ind)
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(np.bool)
        
        # ev_mask
        ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)
        # ev_mask_col = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent *
        # self._scale_mask_col)], M_warp)

        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image_map = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image_map[road_mask] = COLOR_ALUMINIUM_5
        
        image[route_mask] = COLOR_ALUMINIUM_3
        image_map[route_mask] = COLOR_ALUMINIUM_3

        image[lane_mask_all] = COLOR_MAGENTA
        image_map[lane_mask_all] = COLOR_MAGENTA

        image[lane_mask_broken] = COLOR_MAGENTA_2
        image_map[lane_mask_broken] = COLOR_MAGENTA_2

        image[sidewalk_mask] = COLOR_ALUMINIUM_1
        image[shoulder_mask] = COLOR_YELLOW_2
        image[parking_mask] = COLOR_ALUMINIUM_0

        h_len = len(self._history_idx) - 1
        # tint is used for history trajectory. the time is longer, the trajectory is lighter
        # mask traffic light
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len - i) * 0.2)
            image_map[mask] = tint(COLOR_GREEN, (h_len - i) * 0.2)

        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len - i) * 0.2)
            image_map[mask] = tint(COLOR_YELLOW, (h_len - i) * 0.2)

        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len - i) * 0.2)
            image_map[mask] = tint(COLOR_RED, (h_len - i) * 0.2)

        # mask vehicles and walkers
        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len - i) * 0.2)

        #for i, mask in enumerate(ev_histroy_masks):
            #image[mask] = tint(COLOR_WHITE, (h_len - i) * 0.2)
            #image_map[mask] = tint(COLOR_WHITE, (h_len - i) * 0.2)

        image[ev_mask] = COLOR_WHITE
        image_map[ev_mask] = COLOR_WHITE

        # image[obstacle_mask] = COLOR_BLUE

        # masks
        c_road = road_mask * 255
        c_ev = ev_mask * 255
        c_route = route_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 120

        # masks with history
        c_tl_history = []
        for i in range(len(self._history_idx)):
            c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
            c_tl[tl_green_masks[i]] = 80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl_history.append(c_tl)

        c_vehicle_history = [m * 255 for m in vehicle_masks]
        c_walker_history = [m * 255 for m in walker_masks]


        # the following axis transpose may be confusing, we can use axis=0 instead of axis=2 & transpose[2,0,1] 
        masks = np.stack((c_road, c_route, c_lane, c_ev, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
        ev_histroy_masks = np.stack(ev_histroy_masks)
        #else:
            #masks = np.stack((c_road, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
        masks = np.transpose(masks, [2, 0, 1])

        obs_dict = {'rendered': image, 'masks': masks, 'ev_history_masks': ev_histroy_masks, 'image_map':image_map}

        # self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])

        return obs_dict

    def get_full_obs(self):
        w, h = self._road.shape
        # init img
        route_img = np.zeros([w,h], dtype=np.uint8)
        full_img = np.zeros([w,h,3], dtype=np.uint8)
        # draw ego_v and route and road
        ev_transform = self._ego_vehicle.get_transform()
        ev_loc = ev_transform.location
        ev_in_pixel = np.array(self._world_to_pixel(ev_loc))
        
        road_bool = np.array(self._road, dtype=bool)
        full_img[road_bool] = COLOR_ALUMINIUM_5
        if self._is_des: 
            route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                    for wp, _ in self._route_trace])
            cv.polylines(route_img, [np.round(route_in_pixel).astype(np.int32)], False, 1, thickness=8)
            route_bool = np.array(route_img, dtype=bool)
            full_img[route_bool] = COLOR_GREEN
            des_in_pixel = np.array(self._world_to_pixel(self._des))
            cv.circle(full_img, np.round(des_in_pixel).astype(np.int32), 1, COLOR_BLUE, 8)
        cv.circle(full_img, np.round(ev_in_pixel).astype(np.int32), 1, COLOR_WHITE, 8)
        
        return full_img


    @staticmethod
    def _get_surrounding_actors(bbox_list, criterium, scale=None):
        actors = []
        for bbox in bbox_list:
            is_within_distance = criterium(bbox)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors

    def _get_history_masks(self, M_warp):
        q_size = len(self._history_queue)
        vehicle_masks, walker_masks, ev_masks, tl_green_masks, tl_yellow_masks, tl_red_masks = [], [], [], [], [], []
        for idx in self._history_idx:
            idx = max(idx, -1 * q_size)
            vehicles, walkers, evs, tl_green, tl_yellow, tl_red = self._history_queue[idx]
            vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))
            tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))
            tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))
            tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))

        for idx in self._ev_history_idx:
            idx = max(idx, -1 * q_size)
            _, _, evs, _, _, _ = self._history_queue[idx]
            ev_masks.append(self._get_mask_from_actor_list(evs, M_warp))

        return vehicle_masks, walker_masks, ev_masks, tl_green_masks, tl_yellow_masks, tl_red_masks

    def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for sp_locs in stopline_vtx:
            stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
            stopline_warped = cv.transform(stopline_in_pixel, M_warp).astype(int)
            cv.line(mask, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
                    color=1, thickness=6)
        return mask.astype(np.bool)

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5 * self._width) * right_vec
        top_left = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec - (
                    0.5 * self._width) * right_vec
        top_right = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec + (
                    0.5 * self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width - 1],
                            [0, 0],
                            [self._width - 1, 0]], dtype=np.float32)
        return cv.getAffineTransform(src_pts, dst_pts)

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            # corners under the ev local coordinates
            corners = [bb_loc + corner for corner in corners]

            # converts the ev coodinates to world coordinates
            corners = [actor_transform.transform(corner) for corner in corners]
            # converts the world coodinate to pixel coordinates
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])

            # porject the global pixel coordinates to ev local pixel coordinates
            corners_warped = cv.transform(corners_in_pixel, M_warp)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(np.bool)
