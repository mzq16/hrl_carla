import random

class Task(object):
    def __init__(self, ego_vehicle, client, num_des=1) -> None:
        '''
        Args:
            ev_loc(vector3(x,y,z)): ego vehicle's locaion
            num_des(int): number of destination. after arrived all destinations, shut down the game loop
            client(carla.Client): client
        '''
        self._ev = ego_vehicle
        self._world = ego_vehicle.get_world()
        self._map = self._world.get_map()
        self._spawn_points = list(self._map.get_spawn_points())
        self._ev_loc = self._ev.get_transform().location
        if not self.check_ego_loc:
            print('need to wait')
        self._client = client
        self._num_des = num_des
        #self.gen_task_des()
        
    def select_des(self, distance=20):
        world = self._client.get_world()
        map = world.get_map()
        ego_waypoint = map.get_waypoint(self._ev_loc)
        next_waypoint =  ego_waypoint.next(distance)
        index = 0
        while len(next_waypoint) == 0:
            world.tick()
            ego_waypoint = map.get_waypoint(self._ev_loc)
            next_waypoint =  ego_waypoint.next(distance)
        if len(next_waypoint) != 1:
            #print('more than 1 waypoint')
            index = random.randint(0,len(next_waypoint) - 1)
        
        return next_waypoint[index].transform.location, distance

    def gen_task_des(self, min_dist=100, max_dist=1000):
        for i in range(self._num_des):
            dist = random.randint(min_dist, max_dist)
            des, _ = self.select_des(distance=dist)
            #des_transform = random.choice(self._spawn_points)
            #des = des_transform.location
        return des

    def check_ego_loc(self):
        if self._ev_loc.x or self._ev_loc.y or self._ev_loc.z:
            return True
        else:
            return False

    def change_des(self):
        pass