import numpy as np
import random
import sys

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla


class OtherWalker(object):
    def __init__(self, client, num_walkers=50):
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
        self._start()

    def destroy(self):
        self._stop()
        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._controller_list])
        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._walkers_list])
        #for i in range(len(self._controller_list)):
           #self.controller[i].stop()
        
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
        # import module
        SpawnActor = carla.command.SpawnActor

        # spawn points
        spawn_points = self._spawn_points
        number_of_spawn_points = len(spawn_points)
        if self._num_walkers <= number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self._num_walkers > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            print(msg % (number_of_spawn_points, number_of_spawn_points))
            self._num_walkers = number_of_spawn_points
            random.shuffle(spawn_points)

        # batch n command
        batch = []
        self._walker_speed = []
        for n, transform in enumerate(spawn_points):
            if n == self._num_walkers:
                # if sp number > walker number
                #print("spawn %d background walker" % n)
                break
            bp = random.choice(self._walker_bp)
            # is_invincible
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            # set max speed
            if bp.has_attribute('speed'):
                if random.random() > self.percentagePedestriansRunning:
                    # running
                    self._walker_speed.append(bp.get_attribute('speed').recommended_values[1])
                else:
                    # walking
                    self._walker_speed.append(bp.get_attribute('speed').recommended_values[2])
                # bp.set_attribute('speed', walker_speed[-1])
            else:
                print('walker has no speed')
                self._walker_speed.append(0.0)
            batch.append(SpawnActor(bp, transform))
        
        # execute n commands to spawn
        results = self._client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                pass
                #print(response.error)
            else:
                self._walkers_list.append(results[i].actor_id)
                

        # batch n commands for controller
        batch_c = []
        controller_bp = self._walker_controller_bp
        for i in range(len(self._walkers_list)):
            batch_c.append(SpawnActor(controller_bp, carla.Transform(), self._walkers_list[i]))
            
        results = self._client.apply_batch_sync(batch_c, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self._controller_list.append(results[i].actor_id)

        #self._world.wait_for_tick()
        # get all actors by actor ids
        self.walkers = self._world.get_actors(self._walkers_list)
        self.controller = self._world.get_actors(self._controller_list)

    def _start(self):
        for i in range(len(self._controller_list)):
            self.controller[i].start()
            # this function will cause error
            self.controller[i].go_to_location(self._world.get_random_location_from_navigation())
            self.controller[i].set_max_speed(float(self._walker_speed[i]))

    def _stop(self):
        for i in range(len(self._controller_list)):
            self.controller[i].stop()

    