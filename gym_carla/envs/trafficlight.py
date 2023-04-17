import sys
import numpy as np

sys.path.append('/mnt/ubuntu/Lab_Files/04_MZQ/carla/PythonAPI/carla/dist/carla-0.9.12-py3.9-linux-x86_64.egg')

import carla

class Trafficlight(object):
    def __init__(self, client):
        self._client = client
        self._world = client.get_world()
        # get global traffic light info
        tl_num, tl_actor, tv_loc, swp, sv = self.get_all_traffic_light_waypoints(self._world)
        self.tl_num = tl_num
        self.tl_actor = tl_actor
        self.tv_loc = tv_loc
        self.swp = swp
        self.sv = sv

    def get_traffic_light_waypoints(self, traffic_light, carla_map):
        '''
        base_transform is the traffic light's transform
        tv is trigger volume, which is a parameter to describe traffic sign. Inside the trigger volume, the carla.Actor can know about it.
            for traffic light, the trigger volume means the most far locaiton for vehicles to konw the traffic light. The following code 
            use the structure to find the stop line. 
            1.trigger volume    2.tv extent.x   3.trasfor local-x to global-x   4.go forward util arrived at traffic intersection
            5.the point may be the center of the lane and arrived at the traffic intersection. got stopline wp
            6.get the stopline vertices, from center go left and go right
        stopline wp: center of the lane(x-axis) 
        stopline vertices: left or right vertices of the lanes
        '''
        base_transform = traffic_light.get_transform()
        tv_loc = traffic_light.trigger_volume.location # center
        tv_ext = traffic_light.trigger_volume.extent # do not include inverse forward, cover all the right foward traffic intersection

        # Discretize the trigger box into points
        x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes
        area = []
        for x in x_values:
            point_location = base_transform.transform(tv_loc + carla.Location(x=x))
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = carla_map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Leaderboard: Advance them until the intersection
        stopline_wps = []
        stopline_vertices = []
        junction_wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            junction_wps.append(wpx)

            stopline_wps.append(wpx)
            vec_forward = wpx.transform.get_forward_vector()
            vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

            loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
            loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
            stopline_vertices.append([loc_left, loc_right])

        return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices


    def get_all_traffic_light_waypoints(self, world):
        '''
        output: put all traffic info in list, all outputs are list. for single traffic info, use get_traffic_light_waypoints().
        num_traffic_light(list), 
        traffic_light_actor(list),
        trigger_volume_loc(list): for more details, go to get_traffic_light_waypoints()
        stop_line_way_points(list), 
        stop_line_vertices(list)
        '''
        carla_map = world.get_map()
        all_actors = world.get_actors()
        traffic_light_actor = []
        trigger_volume_loc = []
        stop_line_way_points = []
        stop_line_vertices = []
        num_traffic_light = 0

        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                tv_loc, stopline_wps, stopline_vtx = self.get_traffic_light_waypoints(_actor, carla_map)
                traffic_light_actor.append(_actor)
                trigger_volume_loc.append(tv_loc)
                stop_line_way_points.append(stopline_wps)
                stop_line_vertices.append(stopline_vtx)
                num_traffic_light += 1

        return num_traffic_light, traffic_light_actor, trigger_volume_loc, stop_line_way_points, stop_line_vertices


    def get_stop_line_vtx(self, veh_loc, color, num, tl_actor, tv_loc, swp, sv, dist_threshold=50.0):
        if color == 0:
            tl_state = carla.TrafficLightState.Green
        elif color == 1:
            tl_state = carla.TrafficLightState.Yellow
        elif color == 2:
            tl_state = carla.TrafficLightState.Red

        stopline_vtx = []

        for i in range(num):
            traffic_light = tl_actor[i]
            tv_loc1 = tv_loc[i]
            if tv_loc1.distance(veh_loc) > dist_threshold:
                continue
            if traffic_light.state != tl_state:
                continue
            stopline_vtx += sv[i]

        return stopline_vtx

    def get_parameter(self):
        return self.tl_num, self.tl_actor, self.tv_loc, self.swp, self.sv

    def get_encounter_trafficlight(self, egovehicle, tl_actor):
        egovehicle_transform = egovehicle.get_transform()
        egovehicle_direction = egovehicle_transform.get_forward_vector()

    
    

