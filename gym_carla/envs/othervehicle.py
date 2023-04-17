import numpy as np
import random
import sys

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla
from carla import VehicleLightState as vls

DISCARD_VEHICLE = [
    'vehicle.micro.microlino', 
    'vehicle.dodge.charger_police', 
    'vehicle.mercedes.coupe', 
    'vehicle.carlamotors.carlacola', 
    'vehicle.dodge.charger_police_2020', 
    'vehicle.mercedes.sprinter', 
    'vehicle.ford.crown', 
    'vehicle.ford.ambulance', 
    'vehicle.nissan.patrol_2021', 
    'vehicle.lincoln.mkz_2020', 
    'vehicle.lincoln.mkz_2017', 
    'vehicle.tesla.cybertruck', 
    'vehicle.chevrolet.impala', 
    'vehicle.carlamotors.firetruck', 
    'vehicle.audi.etron', 
    'vehicle.dodge.charger_2020',
    ]

class OtherVehicle(object):
    def __init__(self, client, port, num_vehicles=100, car_lights_on=False):
        self._vehicles_list = []
        self._car_lights_on = car_lights_on
        self._client = client
        self._world = self._client.get_world()
        self._map = self._world.get_map()
        self._tm_port = port + 6000
        self._traffic_manager = client.get_trafficmanager(self._tm_port)
        self._num_vehicles = num_vehicles
        self._spawn_points = list(self._get_spawn_point())
        self.bps = self._get_vehicle_blueprint_library()


    def reset(self, num_vehicles=None):
        self.destroy()
        if num_vehicles is not None:
            self._num_vehicles = num_vehicles
        self._spawn_vehicles()
        self._world.tick()

    def reset_once(self):
        self.destroy()
        self._spawn_vehicles()

    def destroy(self):
        #self._client.apply_batch([carla.command.DestroyActor(x) for x in self._vehicles_list])
        for x in self._vehicles_list:
            x.destroy()
        self._vehicles_list = []
        #print("destroy all background vehicles")

    def _get_vehicle_blueprint_library(self, discard=True):
        blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        if discard == True:
            blueprints = [x for x in blueprints if not x.id in DISCARD_VEHICLE]
        return blueprints

    def _set_num_vehicles(self, num_vehicles):
        self._num_vehicles = num_vehicles

    def _get_spawn_point(self):
        all_spawn_points = self._map.get_spawn_points()
        return all_spawn_points

    # do not need to set traffic synchronous in here
    """
    def _set_synchronous(self, fps=20):
        delta_seconds = 1.0 / fps
        settings = self._world.get_settings()
        self._traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = delta_seconds
        else:
            synchronous_master = False
    """

    def _set_hybrid(self):
        self._traffic_manager.set_hybrid_physics_mode(True)
        self._traffic_manager.set_hybrid_physics_radius(50.0)
        #print("set other vehicles in hybrid mode")

    def _spawn_vehicles(self):
        # import module
        #SpawnActor = carla.command.SpawnActor
        #SetAutopilot = carla.command.SetAutopilot
        #SetVehicleLightState = carla.command.SetVehicleLightState
        #FutureActor = carla.command.FutureActor

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

        
        while(len(self._vehicles_list) < self._num_vehicles):
            bp = random.choice(self.bps)
            transform = random.choice(spawn_points)
            vehicle = self._world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                self._vehicles_list.append(vehicle)

        # set autopilot
        for v in self._vehicles_list:
            v.set_autopilot(True, self._tm_port)

        # safe distance
        self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # hybrid
        self._set_hybrid()
        # max speed
        self._traffic_manager.global_percentage_speed_difference(-20)
        '''
        # batch n command
        batch = []
        for n, transform in enumerate(spawn_points):
            if n == self._num_vehicles:
                #print("spawn %d background vehicles" % n)
                break
            bp = random.choice(self.bps)
            # vehicle = self._world.try_spawn_actor(bp, transform)
            # self._actor_list.append(vehicle)

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if self._car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            batch.append(SpawnActor(bp, transform)
                         .then(SetAutopilot(FutureActor, True, self._tm_port))
                         .then(SetVehicleLightState(FutureActor, light_state)))

        # execute n commands to spawn
        for response in self._client.apply_batch_sync(batch):
            if not response.error:
                self._vehicles_list.append(response.actor_id)
            else:
                pass
                #print(response.error)
        '''




