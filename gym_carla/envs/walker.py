import numpy as np
import random
import sys

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla


class Otherwalker(object):
    def __init__(self, client, num_walkers=10):
        self._walkers_list = []
        self.percentagePedestriansRunning = 0.0  # how many pedestrians will run
        self.percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        self._client = client
        self._world = self._client.get_world()
        self._map = self._world.get_map()
        self._num_walkers = num_walkers

    def reset(self):
        self.destroy()
        self._spawn_vehicles()

    def destroy(self):
        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._walkers_list])
        self._walkers_list = []
        print("destroy all background vehicles")

    def get_vehicle_blueprint_library(self):
        return self._world.get_blueprint_library().filter("vehicle.*")

    def set_num_vehicles(self, num_vehicles):
        self._num_walkers = num_vehicles

    def _get_spawn_point(self):
        all_spawn_points = self._map.get_spawn_points()
        return all_spawn_points

    def _spawn_vehicles(self):
        # import module
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # spawn points
        spawn_points = self._spawn_points
        number_of_spawn_points = len(spawn_points)
        if self._num_vehicles <= number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self._num_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            print(msg % (number_of_spawn_points, number_of_spawn_points))
            self._num_vehicles = number_of_spawn_points
            random.shuffle(spawn_points)

        # batch n command
        batch = []
        for n, transform in enumerate(spawn_points):
            if n == self._num_vehicles:
                print("spawn %d background vehicles" % n)
                break
            bp = random.choice(self.bps)
            # vehicle = self._world.try_spawn_actor(bp, transform)
            # self._actor_list.append(vehicle)

            batch.append(SpawnActor(bp, transform)
                         .then(SetAutopilot(FutureActor, True, self._tm_port)))

        # execute n commands to spawn
        for response in self._client.apply_batch_sync(batch):
            if not response.error:
                self._walkers_list.append(response.actor_id)
            else:
                print(response.error)
