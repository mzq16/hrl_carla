import numpy as np
import random
import sys
from tqdm import tqdm
import time

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla


class OtherWalker(object):
    def __init__(self, client, num_walkers=10):
        self._walkers_list = []
        self._controller_list = []
        self._walker_speed = []
        self.percentagePedestriansRunning = 0.0  # how many pedestrians will run
        self.percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        self._client = client
        self._world = self._client.get_world()
        self._map = self._world.get_map()
        self._num_walkers = num_walkers
        self._spawn_points = self._get_spawn_point()

        self._walker_bp = self._get_walker_blueprint_library()
        self._walker_controller_bp = self._get_walker_controller_blueprint_library()

    def reset(self):
        self.destroy()
        self._spawn_walkers()
        self._world.tick()
        self._start()

    def destroy(self):
        self._stop()
        #for i in range(len(self._controller_list)):
        #    self._controller_list[i].stop()
        #self._client.apply_batch([carla.command.DestroyActor(x) for x in self._walkers_list])
        for x in self._walkers_list:
            x.destroy()
        #self._client.apply_batch([carla.command.DestroyActor(x) for x in self._controller_list])
        for x in self._controller_list:
            x.destroy()
        self._walkers_list = []
        self._controller_list = []
        self._walker_speed = []
        #print("destroy all background walkers and controllers")

    def _get_walker_blueprint_library(self):
        # used for spawning
        return self._world.get_blueprint_library().filter("walker.pedestrian.*")

    def _get_walker_controller_blueprint_library(self):
        # used for spawning
        return self._world.get_blueprint_library().find("controller.ai.walker")

    def set_num_walkers(self, num_walkers):
        self._num_walkers = num_walkers

    def _get_spawn_point(self):
        # find all random location to spawn walkers
        spawn_points = []
        for i in range(self._num_walkers):
            spawn_point = carla.Transform()
            loc = self._world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        return spawn_points

    def _get_spawn_point_m(self):
        '''include roads, crosswalks, grass. the most is sidwalk'''
        spawn_points = []
        # grass

        # crosswalks(return list of carla locaiton(x,y,z))
        map = self._world.get_map()
        list_crosswalks = map.get_crosswalks()

        # sidewalk

        # all_spawn_points = self._map.get_spawn_points()
        for i in range(self._num_walkers):
            spawn_point = carla.Transform()
            loc = list_crosswalks[i]
            if loc is not None:
                loc.z = loc.z + 1
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        return spawn_points

    def get_spawn_point(self):
        return self._spawn_points

    def get_walker_bp(self):
        return self._walker_bp
    
    def get_walker_controller_bp(self):
        return self._walker_controller_bp

    def _spawn_walkers(self):
        random.seed(int(time.time()))
        self._world.set_pedestrians_seed(int(time.time()))
    
        #pbar = tqdm(total=self._num_walkers)
        while(len(self._walkers_list) < self._num_walkers):
            transform = carla.Transform(location=self._world.get_random_location_from_navigation())
            walker_bp = random.choice(self._walker_bp)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            tmp_walker = self._world.try_spawn_actor(walker_bp, transform)
            if tmp_walker != None:
                self._walkers_list.append(tmp_walker)
                if walker_bp.has_attribute('speed'):
                    self._walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                #pbar.update(1)
        #pbar.close()

        controller_bp = self._walker_controller_bp
        for i in range(len(self._walkers_list)):
            tmp_controller = self._world.spawn_actor(controller_bp, carla.Transform(), self._walkers_list[i])
            self._controller_list.append(tmp_controller)
      

    def _start(self):
        for i in range(len(self._controller_list)):
            self._controller_list[i].start()
            # this function will cause error because of the wrong version between carlaserver and its api
            self._controller_list[i].go_to_location(self._world.get_random_location_from_navigation())
            self._controller_list[i].set_max_speed(float(self._walker_speed[i]))

    def _stop(self):
        for i in range(len(self._controller_list)):
            self._controller_list[i].stop()