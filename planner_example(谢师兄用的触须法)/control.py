from __future__ import division, print_function

import os.path
import sys
import time

import numpy as np
import copy

import scipy.io

from highway_env import utils
from highway_env.vehicle.dynamics import Vehicle
import global_val
import uuid
sys.path.append(os.path.dirname(__file__))
import behavioural_planner
import local_planner
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from EM_Planner import Speed_planner, traj_index_trans2s_vec, virtual_dynamic_obs_planning, traj_index_trans2s
import EM_Planner
from vehicle_control import PID

class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.6  # [s]  纵向加速度时间常数
    TAU_DS = 0.2  # [s]  航向角时间常数
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A  # 纵向加速度PID 比例系数（P）
    KP_HEADING = 1 / TAU_DS  # 横向航向角PID比例系数
    KP_LATERAL = 1 / 0.5  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad] 最大转向角(60度)

    DELTA_VELOCITY = 5  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route
        self.actual_action = None
        self.longi_acc = 0
        self.id = str(uuid.uuid4())

        # Decomposed method
        self.weight = 1  # NDD probablity/Critical probability default is 1 for no manipulation
        self.criticality = 0  # Just used for BVs for those we controlled using decomposed method. For other vehicle such as CAV, this attribute has no meaning
        self.decomposed_controlled_flag = False  # Used for BV that has maximum criticality and selected to controll

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
            Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None, essential_flag=False):
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()  # 大概是判断有没有到道路的尽头，到了就自动转到下一个开始
        _from, _to, _id = self.lane_index
        if essential_flag == 0:
            if action:
                self.actual_action = action
                if action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                    self.actual_action = "IDLE"
                elif action == "LANE_LEFT" and self.lane_index == 0:
                    self.actual_action = "IDLE"
            # if action == "FASTER":
            #     self.longi_acc = self.DELTA_VELOCITY

            # elif action == "SLOWER":
            #     self.longi_acc = -self.DELTA_VELOCITY
            if action == "LANE_RIGHT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif action == "LANE_LEFT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif action:
                self.longi_acc = float(action)

        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.longi_acc}
        super(ControlledVehicle, self).act(action)

    def follow_road(self):
        """
           At the end of a lane, automatically switch to a next one.
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index, route=self.route, position=self.position, np_random=self.road.np_random)
            # print(self.target_lane_index)

    def steering_control(self, target_lane_index):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)   # 定到-pi到pi
        # Heading rate to steering angle
        steering_angle = self.LENGTH / utils.not_zero(self.velocity) * np.arctan(heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def set_route_at_intersection(self, _to):
        """
            Set the road to be followed at the next intersection.
            Erase current planned route.
        :param _to: index of the road to follow at next intersection, in the road network
        """

        if not self.route:
            return
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return
        next_destinations_from = list(next_destinations.keys())
        if _to == "random":
            _to = self.road.np_random.randint(0, len(next_destinations_from))
        next_index = _to % len(next_destinations_from)
        self.route = self.route[0:index + 1] + \
                     [(self.route[index][1], next_destinations_from[next_index], self.route[index][2])]

    def get_action_indicator(self, ndd_flag=False, safety_flag=True, CAV_flag=False):
        """
        Get AV action indicator.

        Args:
            ndd_flag: whether bound by naturalistic data.
            safety_flag: whether bound by safety.
            CAV_flag: whether is AV.

        Returns:
            np.array with the same size of the AV action.
        """
        if CAV_flag:
            action_shape = len(global_val.ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                pass
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                obs = self._get_veh_obs()
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(obs, lateral_action_indicator, CAV_flag=True)
                longi_result = self._check_longitudinal_safety(obs, np.ones(action_shape - 2), lateral_result=lateral_result, CAV_flag=True)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2]
                safety_action_indicator[2:] = longi_result
            action_indicator = ndd_action_indicator * safety_action_indicator
            action_indicator = (action_indicator > 0)
            return action_indicator
        else:
            raise ValueError("Get BV action Indicator in CAV function")

    def _check_longitudinal_safety(self, obs, pdf_array, lateral_result=None, CAV_flag=False):
        """
        Model-based safety guard for each longitudinal acceleration.

        Args:
            obs: observation.
            pdf_array: potential action array.
            lateral_result: lateral feasibility
            CAV_flag: whether current checking vehicle is AV.

        Returns:
            whether each maneuver is safe (>0 or True for safe, =0 or False for unsafe).
        """
        f_veh, _, _, _, _, _ = obs
        safety_buffer = global_val.longi_safety_buffer
        for i in range(len(pdf_array) - 1, -1, -1):
            if CAV_flag:
                acc = global_val.CAV_acc_to_idx_dic.inverse[i]
            else:
                acc = global_val.acc_to_idx_dic.inverse[i]
            if f_veh:
                rr = f_veh.velocity - self.velocity
                r = f_veh.position[0] - self.position[0] - self.LENGTH
                criterion_1 = rr * global_val.simulation_resolution + r + 0.5 * (global_val.acc_low - acc) * global_val.simulation_resolution ** 2
                self_v_2, f_v_2 = max(self.velocity + acc, global_val.v_low), max((f_veh.velocity + global_val.acc_low), global_val.v_low)
                dist_r = (self_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low))
                dist_f = (f_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low)) + global_val.v_low * (f_v_2 - self_v_2) / global_val.acc_low
                criterion_2 = criterion_1 - dist_r + dist_f
                if criterion_1 <= safety_buffer or criterion_2 <= safety_buffer:
                    pdf_array[i] = 0
                    # if CAV_flag:
                    #     print(pdf_array)
                else:
                    break

        # Only set the decelerate most when non of lateral is OK.
        if lateral_result is not None:
            lateral_feasible = lateral_result[0] or lateral_result[2]
        else:
            lateral_feasible = False
        if np.sum(pdf_array) == 0 and not lateral_feasible:
            pdf_array[0] = 1 if not CAV_flag else np.exp(-2)
            return pdf_array

        if CAV_flag:
            new_pdf_array = pdf_array
        else:
            new_pdf_array = pdf_array / np.sum(pdf_array)
        return new_pdf_array

    def _check_lateral_safety(self, obs, pdf_array, CAV_flag=False):
        """
        Model-based safety guard for each lateral maneuver.

        Args:
            obs: observation.
            pdf_array: potential action array.
            CAV_flag: whether current checking vehicle is AV.

        Returns:
            whether each maneuver is safe (>0 or True for safe, =0 or False for unsafe). [Left turn, go straight, right turn],

        """
        f1, r1, f0, r0, f2, r2 = obs
        lane_change_dir = [0, 2]
        nearby_vehs = [[f0, r0], [f2, r2]]
        safety_buffer = global_val.lateral_safety_buffer
        if self.lane_index[2] == 0:
            pdf_array[0] = 0
        elif self.lane_index[2] == 2:
            pdf_array[2] = 0
        for lane_index, nearby_veh in zip(lane_change_dir, nearby_vehs):
            if pdf_array[lane_index] != 0:
                f_veh, r_veh = nearby_veh[0], nearby_veh[1]
                if f_veh:
                    rr = f_veh.velocity - self.velocity
                    r = f_veh.position[0] - self.position[0] - self.LENGTH
                    dis_change = rr * global_val.simulation_resolution + 0.5 * global_val.acc_low * global_val.simulation_resolution ** 2
                    r_1 = r + dis_change
                    rr_1 = rr + global_val.acc_low * global_val.simulation_resolution

                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2, f_v_2 = max(self.velocity, global_val.v_low), max((f_veh.velocity + global_val.acc_low), global_val.v_low)
                        dist_r = (self_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low))
                        dist_f = (f_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low)) + global_val.v_low * (f_v_2 - self_v_2) / global_val.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0

                if r_veh:
                    rr = self.velocity - r_veh.velocity
                    r = self.position[0] - r_veh.position[0] - self.LENGTH
                    dis_change = rr * 1 - 0.5 * global_val.acc_high * 1 ** 2
                    r_1 = r + dis_change
                    rr_1 = rr - global_val.acc_high * 1
                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2, r_v_2 = min(self.velocity, global_val.v_high), min((r_veh.velocity + global_val.acc_high), global_val.v_high)
                        dist_r = (r_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low))
                        dist_f = (self_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low)) + global_val.v_low * (-r_v_2 + self_v_2) / global_val.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0
        if np.sum(pdf_array) == 0:
            return np.array([0, 1, 0])

        if CAV_flag:
            new_pdf_array = pdf_array
        else:
            new_pdf_array = pdf_array / np.sum(pdf_array)
        return new_pdf_array

    def _get_veh_obs(self):
        """
        Get vehicle surround observations. f0, f1, f2 denote closest other vehicles in front on the subject vehicle in the left adjacent lane, same lane, right adjacent lane.
        r0, r1, r2 denote the corresponding ones behind the subject vehicle.

        Returns:
            observations.
        """
        lane_id = self.lane_index[2]
        observation = []  # observation for this vehicle
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.cav_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.cav_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.cav_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.cav_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.cav_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.cav_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.cav_obs_range)
            f2, r2 = None, None
            observation = [f1, r1, f0, r0, f2, r2]
        return observation

    def _get_range_vehicle(self, f_range, r_range):
        return self.road.vehicles_in_range(self, f_range, r_range)

class MDPVehicle(ControlledVehicle):
    """
        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 3  # []
    SPEED_MIN = 20  # [m/s]
    SPEED_MAX = 30  # [m/s]

    # CAV surrogate model Longitudinal policy parameters
    COMFORT_ACC_MAX = global_val.SM_IDM_COMFORT_ACC_MAX  # [m/s2]  2
    COMFORT_ACC_MIN = global_val.SM_IDM_COMFORT_ACC_MIN  # [m/s2]  -4
    DISTANCE_WANTED = global_val.SM_IDM_DISTANCE_WANTED  # [m]  5
    TIME_WANTED = global_val.SM_IDM_TIME_WANTED  # [s]  1.5
    DESIRED_VELOCITY = global_val.SM_IDM_DESIRED_VELOCITY  # [m/s]
    DELTA = global_val.SM_IDM_DELTA  # []

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        # self.velocity_index = self.speed_to_index(self.target_velocity)
        # self.target_velocity = self.index_to_speed(self.velocity_index)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        # acceleration = self.COMFORT_ACC_MAX * (
        #         1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), self.DELTA))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / self.DESIRED_VELOCITY, self.DELTA))
        if front_vehicle:
            d = max(1e-5, ego_vehicle.lane_distance_to(front_vehicle) - self.LENGTH)
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + max(0, ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab)))
        return d_star

    def act(self, action=None, essential_flag=False):
        """
            Perform a high-level action.

            If the action is a velocity change, choose velocity from the allowed discrete range.
            Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        super(MDPVehicle, self).act(action, essential_flag=essential_flag)
        # if action == "FASTER":
        #     self.velocity_index = self.speed_to_index(self.velocity) + 1
        # elif action == "SLOWER":
        #     self.velocity_index = self.speed_to_index(self.velocity) - 1
        # else:
        #     super(MDPVehicle, self).act(action)
        #     return
        # self.velocity_index = np.clip(self.velocity_index, 0, self.SPEED_COUNT - 1)
        # self.target_velocity = self.index_to_speed(self.velocity_index)
        # super(MDPVehicle, self).act()

    @classmethod
    def index_to_speed(cls, index):
        """
            Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
            Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
            The index of current velocity
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
            Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


class Lattice_Planner():
    def __init__(self, dt):
        self.pre_local_waypoints = None
        self.pre_u0 = np.array([0.0, 0.0])
        self.NUM_PATHS = 9
        self.BP_LOOKAHEAD_BASE = 15.0  # m
        self.BP_LOOKAHEAD_TIME = 0.5  # s
        self.PATH_OFFSET = 1.0  # m
        self.CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
        self.CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
        self.TIME_GAP = 1.0  # s
        self.PATH_SELECT_WEIGHT = 10
        self.A_MAX = 2.0  # m/s^2
        self.SLOW_SPEED = 2.0  # m/s
        self.STOP_LINE_BUFFER = 3.5  # m
        self.LP_FREQUENCY_DIVISOR = 4  # Frequency divisor to make the

        self.waypoints = None

        # Path interpolation parameters
        self.INTERP_DISTANCE_RES = 0.01  # distance between interpolated points
        self.lp = local_planner.LocalPlanner(self.NUM_PATHS,
                                             self.PATH_OFFSET,
                                             self.CIRCLE_OFFSETS,
                                             self.CIRCLE_RADII,
                                             self.PATH_SELECT_WEIGHT,
                                             self.TIME_GAP,
                                             self.A_MAX,
                                             self.SLOW_SPEED,
                                             self.STOP_LINE_BUFFER)
        self.bp = behavioural_planner.BehaviouralPlanner(self.BP_LOOKAHEAD_BASE)   # 这里要改动BP，第二个参数不需要
        self.hyperparameters = {
            "max_accel": 2,
            "min_accel": -4,
            "max_steering": np.pi/4
        }
        self.time = 0
        self.dt = dt
        self.paths = None
        self.best_index = 0
        self.prev_timestamp = 0
        self.time = 0
        self.dt = dt
        self.paths = None
        self.best_index = 0
        self.prev_timestamp = 0
        self.max_lateral_accel = 100
        self.traj = None
        self.frame = 0
        self.relative_time = 1
        self.s = None
        self.s_dot = None
        self.relative_time_set = None
        self.theta = None
        self.trajectory_kappa_init = None
        self.s_pid = PID(KP=0.5, KI=0, KD=0)
        self.v_pid = PID(KP=2.5, KI=0.1, KD=0)
        self.planner_begin = [0, 0]
        self.index2s = None
        self.speed_planner = Speed_planner()
        self.first_run = True
        self.pre_traj = None
        self.traj_xy = None
        self.s_final = None
        self.s_dot_final = None
        self.traj_x_final = None
        self.traj_y_final = None
        self.theta_final = None

    def vehicle_path_reset(self):
        self.time = 0
        self.waypoints = None

    def set_global_points(self, global_points):
        self.waypoints = global_points

    def exec_waypoint_nav_sn(self, av, lane_ref, frame=2):
        """
        这个版本适用曲率不大的曲线
        """
        obs = av._get_veh_obs()
        f1, r1, f0, r0, f2, r2 = obs
        dyn_obs_x_list = []
        dyn_obs_y_list = []
        dyn_obs_vx_list = []
        dyn_obs_vy_list = []
        the_same_lane = []

        for vehi in [f1, r1, f0, r0, f2, r2]:
            if vehi is not None:
                longitudinal, lateral = lane_ref.local_coordinates(vehi.position)
                dyn_obs_x_list.append(longitudinal)
                dyn_obs_y_list.append(lateral)
                dyn_obs_vx_list.append(vehi.velocity * np.cos(vehi.heading - lane_ref.heading_at(longitudinal)))
                dyn_obs_vy_list.append(vehi.velocity * np.sin(vehi.heading - lane_ref.heading_at(longitudinal)))
                if vehi is f0:
                    the_same_lane.append(True)
                else:
                    the_same_lane.append(False)

        dyn_obs_x = np.array(dyn_obs_x_list)
        dyn_obs_y = np.array(dyn_obs_y_list)
        dyn_obs_vx = np.array(dyn_obs_vx_list)
        dyn_obs_vy = np.array(dyn_obs_vy_list)

        # 当前参数获取
        current_x, current_y = lane_ref.local_coordinates(av.position)
        current_yaw = av.heading - lane_ref.heading_at(current_x)
        current_speed = av.velocity
        ego_length = av.LENGTH
        current_timestamp = self.time

        # 控制和规划周期不一样，控制25ms规划100ms
        if self.frame % self.LP_FREQUENCY_DIVISOR == 0:
            begin = time.time()
            # 相对时间置0， 其作用是可用来索引期望的s
            self.relative_time = 0
            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - self.prev_timestamp,
                                                                            av.velocity)
            lookahead = 60  # self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed        #直接设置成60m
            self.bp.set_lookahead(lookahead)
            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints,
                                                        ego_state)

            # Calculate planned paths in the local frame.  在车辆坐标系下计算局部路径坐标
            paths, path_validity = self.lp.plan_paths(goal_state_set)

            # Transform those paths back to the global frame.  计算全局坐标系的路径坐标
            self.paths = np.array(local_planner.transform_paths(paths, ego_state))

            # self.paths的维度是Nx3xM
            index2s_set = traj_index_trans2s_vec(self.paths[:, 0, :], self.paths[:, 1, :])
            # 动态障碍物网格划分
            time_step = np.linspace(0, lookahead / av.velocity, 11)

            best_index = self.select_best_path_index_numpy_dyn(av, dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy,
                                                               index2s_set, time_step)
            self.best_index = best_index
            self.traj = self.paths[best_index, ...]
            self.traj_xy = lane_ref.position(self.traj[0, :].reshape(-1, 1), self.traj[1, :].reshape(-1, 1)).T


            # 计算生成的局部路径的航向角和曲率
            self.theta, self.trajectory_kappa_init = EM_Planner.cal_heading_kappa(self.traj[0, :], self.traj[1, :])

            # 寻找障碍物的匹配点
            proj_x_set, proj_y_set, proj_heading_set, match_path_kapp, proj_match_point_index_set = \
                EM_Planner.find_match_point(dyn_obs_x, dyn_obs_y, self.traj[0, :], self.traj[1, :], self.theta, self.trajectory_kappa_init)
            self.index2s = index2s_set[best_index, :]

            # 计算每个障碍物在SL坐标系下的位置
            s_set, l_set = EM_Planner.CalcSL(dyn_obs_x, dyn_obs_y, self.traj[0, :], self.traj[1, :], proj_x_set, proj_y_set,
                                             proj_heading_set, proj_match_point_index_set, self.index2s)
            # 计算每个障碍物在SL坐标系下的速度
            l_dot_set, s_dot_set = EM_Planner.CalcDot_SL(l_set, dyn_obs_vx, dyn_obs_vy, proj_heading_set,
                                                         match_path_kapp)
            # 根据障碍物的SL坐标系的速度和位置生成ST图
            obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set = \
                EM_Planner.generate_st_graph(s_set, l_set, s_dot_set, l_dot_set)

            # 计算av的初始规划条件
            plan_start_s_dot, plan_start_s_dot2 = \
                EM_Planner.calc_speed_planning_start_condition(current_speed * np.cos(current_yaw),
                                                               current_speed * np.sin(current_yaw),
                                                               av.longi_acc, 0.0, current_yaw)
            # 速度的动态规划模块
            self.speed_planner.set_s_list(self.index2s[-1])
            dp_speed_s, dp_speed_t = self.speed_planner.speed_dp(obs_st_s_in_set, obs_st_s_out_set,
                                                                 obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot)
            # self.speed_planner.plot_ST_graph(obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
            #                                  dp_speed_s, dp_speed_t)
            if len(dp_speed_s) > 1:
                # 不知道什么原因，规划的路径有可能小于30，那我直接赋值就可以了
                self.speed_planner.set_s_list(self.index2s[-1])
                # 根据动态规划结果生成凸空间并计算凸空间的上下界约束
                s_lb, s_ub, s_dot_lb, s_dot_ub = \
                    self.speed_planner.convex_space_gen(dp_speed_s, dp_speed_t, self.index2s, obs_st_s_in_set,
                                                        obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                        self.trajectory_kappa_init, self.max_lateral_accel)
                # 二次规划解决速度规划问题
                qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init, status = \
                    self.speed_planner.speed_planning_qp(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t, s_lb,
                                                         s_ub, s_dot_lb, s_dot_ub)
                if status != 'optimal':
                    print('二次规划出问题')
                # 为二次规划算的变量增密
                self.s, self.s_dot, s_dot2, self.relative_time_set = \
                    self.speed_planner.increase_points(qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init)

            print('time cost: ', time.time() - begin)
            self.prev_timestamp = self.time
            self.s_pid.clear_err()
            self.v_pid.clear_err()

        target_index = int((self.relative_time + 0.025) * 40)
        targets_index = int(self.relative_time * 40)
        target_s = self.s[targets_index]
        # 找到车辆在当前位置在s方向的投影
        real_s, _, _, _, _ = \
            EM_Planner.find_match_point(np.array([max(current_x - self.traj[0, 0], 0)]),
                                        np.array([max(current_y - self.traj[1, 0], 0)]),
                                        self.traj[0, :] - self.traj[0, 0], self.traj[1, :] - self.traj[1, 0],
                                        self.theta, self.trajectory_kappa_init)
        if targets_index == 0:
            err_s = 0
        else:
            err_s = target_s - (real_s[0] + 0.2)
        delta_pos = self.s_pid.output_cal(err_s)
        acc = self.v_pid.output_cal(self.s_dot[target_index] - current_speed) + delta_pos

        state = [av.position[0], av.position[1], av.heading]
        steer = self.pp_controller_xy(state, current_speed)

        self.relative_time += 0.025
        self.frame += 1
        # print('the target speed: ', self.s_dot[target_index], 'the real speed: ', current_speed)
        print('the cal action: ', [acc, steer])
        return [acc, steer]


    def exec_waypoint_nav_demo_em(self, av, frame=2):
        """
        这个版本只适合直线
        """
        obs = av._get_veh_obs()
        f1, r1, f0, r0, f2, r2 = obs
        dyn_obs_x_list = []
        dyn_obs_y_list = []
        dyn_obs_vx_list = []
        dyn_obs_vy_list = []
        the_same_lane = []

        for vehi in [f1, r1, f0, r0, f2, r2]:
            if vehi is not None:
                dyn_obs_x_list.append(vehi.position[0])
                dyn_obs_y_list.append(vehi.position[1])
                dyn_obs_vx_list.append(vehi.velocity * np.cos(vehi.heading))
                dyn_obs_vy_list.append(vehi.velocity * np.sin(vehi.heading))
                if vehi is f0:
                    the_same_lane.append(True)
                else:
                    the_same_lane.append(False)

        dyn_obs_x = np.array(dyn_obs_x_list)
        dyn_obs_y = np.array(dyn_obs_y_list)
        dyn_obs_vx = np.array(dyn_obs_vx_list)
        dyn_obs_vy = np.array(dyn_obs_vy_list)

        # 当前参数获取
        current_x = av.position[0]
        current_y = av.position[1]
        current_yaw = av.heading
        current_speed = av.velocity
        ego_length = av.LENGTH
        current_timestamp = self.time

        # 控制和规划周期不一样，控制25ms规划100ms
        if self.frame % self.LP_FREQUENCY_DIVISOR == 0:
            begin = time.time()
            # 相对时间置0， 其作用是可用来索引期望的s
            self.relative_time = 0
            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - self.prev_timestamp,
                                                                            av.velocity)
            lookahead = 60      #  self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed        #直接设置成60m
            self.bp.set_lookahead(lookahead)
            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints,
                                                        ego_state)

            # Calculate planned paths in the local frame.  在车辆坐标系下计算局部路径坐标
            paths, path_validity = self.lp.plan_paths(goal_state_set)

            # Transform those paths back to the global frame.  计算全局坐标系的路径坐标
            self.paths = np.array(local_planner.transform_paths(paths, ego_state))

            # self.paths的维度是Nx3xM
            index2s_set = traj_index_trans2s_vec(self.paths[:, 0, :], self.paths[:, 1, :])
            # 动态障碍物网格划分
            time_step = np.linspace(0, lookahead/av.velocity, 11)

            best_index = self.select_best_path_index_numpy_dyn(av, dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy,
                                                               index2s_set, time_step)
            self.best_index = best_index
            traj = self.paths[best_index, ...]

            self.traj = traj
            # 计算生成的局部路径的航向角和曲率
            theta, trajectory_kappa_init = EM_Planner.cal_heading_kappa(traj[0, :], traj[1, :])
            self.theta = theta
            self.trajectory_kappa_init = trajectory_kappa_init
            # 寻找障碍物的匹配点
            proj_x_set, proj_y_set, proj_heading_set, match_path_kapp, proj_match_point_index_set = \
                EM_Planner.find_match_point(dyn_obs_x, dyn_obs_y, traj[0, :], traj[1, :], theta, trajectory_kappa_init)
            # 计算轨迹每个点所在的长度s
            index2s = index2s_set[best_index, :]
            self.index2s = index2s
            # 计算每个障碍物在SL坐标系下的位置
            s_set, l_set = EM_Planner.CalcSL(dyn_obs_x, dyn_obs_y, traj[0, :], traj[1, :], proj_x_set, proj_y_set,
                                             proj_heading_set, proj_match_point_index_set, index2s)
            # 计算每个障碍物在SL坐标系下的速度
            l_dot_set, s_dot_set = EM_Planner.CalcDot_SL(l_set, dyn_obs_vx, dyn_obs_vy, proj_heading_set, match_path_kapp)
            # 根据障碍物的SL坐标系的速度和位置生成ST图
            obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set = \
                EM_Planner.generate_st_graph(s_set, l_set, s_dot_set, l_dot_set)

            # 计算av的初始规划条件
            plan_start_s_dot, plan_start_s_dot2 = \
                EM_Planner.calc_speed_planning_start_condition(  av.velocity * np.cos(av.heading),
                                                                 av.velocity * np.sin(av.heading),
                                                                 av.longi_acc, 0.0, av.heading)
            # 速度的动态规划模块
            self.speed_planner.set_s_list(index2s[-1])
            dp_speed_s, dp_speed_t = self.speed_planner.speed_dp(obs_st_s_in_set, obs_st_s_out_set,
                                                                 obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot)
            # self.speed_planner.plot_ST_graph(obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
            #                                  dp_speed_s, dp_speed_t)

            if len(dp_speed_s) > 1:
                # 不知道什么原因，规划的路径有可能小于30，那我直接赋值就可以了
                self.speed_planner.set_s_list(index2s[-1])
                # 根据动态规划结果生成凸空间并计算凸空间的上下界约束
                s_lb, s_ub, s_dot_lb, s_dot_ub = \
                    self.speed_planner.convex_space_gen(dp_speed_s, dp_speed_t, index2s, obs_st_s_in_set,
                                                        obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                        trajectory_kappa_init, self.max_lateral_accel)
                # 二次规划解决速度规划问题
                qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init, status = \
                    self.speed_planner.speed_planning_qp(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t, s_lb,
                                                         s_ub, s_dot_lb, s_dot_ub)
                if status != 'optimal':
                    print('二次规划出问题')
                # 为二次规划算的变量增密
                self.s, self.s_dot, s_dot2, self.relative_time_set = \
                    self.speed_planner.increase_points(qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init)
            print('time cost: ', time.time() - begin)
            self.prev_timestamp = self.time
            # self.s_pid.clear_err()
            # self.v_pid.clear_err()

        # .
        target_index = int((self.relative_time + 0.025) * 40)
        targets_index = int(self.relative_time * 40)
        target_s = self.s[targets_index]
        # 找到车辆在当前位置在s方向的投影
        real_s, _, _, _, _ = \
            EM_Planner.find_match_point(np.array([max(current_x - self.traj[0, 0], 0)]),
                                        np.array([max(current_y - self.traj[1, 0], 0)]),
                                        self.traj[0, :] - self.traj[0, 0], self.traj[1, :] - self.traj[1, 0],
                                        self.theta, self.trajectory_kappa_init)
        if targets_index == 0:
            err_s = 0
        else:
            err_s = target_s - (real_s[0])
        delta_pos = self.s_pid.output_cal(err_s)
        acc = self.v_pid.output_cal(self.s_dot[target_index] - current_speed) + delta_pos
        # print(acc)
        state = [current_x, current_y, current_yaw]
        steer = self.pp_controller_(state, current_speed)

        self.relative_time += 0.025
        self.frame += 1
        # print('the target speed: ', self.s_dot[target_index], 'the real speed: ', current_speed)
        print('the cal action: ', [acc, steer])
        return [acc, steer]

    def exec_waypoint_nav_demo_em_stitch(self, av, lane_ref, frame=2):
        """
        这个版涉及曲线拼接
        """
        obs = av._get_veh_obs()
        f1, r1, f0, r0, f2, r2 = obs
        dyn_obs_x_list = []
        dyn_obs_y_list = []
        dyn_obs_vx_list = []
        dyn_obs_vy_list = []
        the_same_lane = []

        for vehi in [f1, r1, f0, r0, f2, r2]:
            if vehi is not None:
                longitudinal, lateral = lane_ref.local_coordinates(vehi.position)
                dyn_obs_x_list.append(longitudinal)
                dyn_obs_y_list.append(lateral)
                dyn_obs_vx_list.append(vehi.velocity * np.cos(vehi.heading - lane_ref.heading_at(longitudinal)))
                dyn_obs_vy_list.append(vehi.velocity * np.sin(vehi.heading - lane_ref.heading_at(longitudinal)))
                if vehi is f0:
                    the_same_lane.append(True)
                else:
                    the_same_lane.append(False)

        dyn_obs_x = np.array(dyn_obs_x_list)
        dyn_obs_y = np.array(dyn_obs_y_list)
        dyn_obs_vx = np.array(dyn_obs_vx_list)
        dyn_obs_vy = np.array(dyn_obs_vy_list)

        # 当前参数获取
        current_x, current_y = lane_ref.local_coordinates(av.position)
        current_yaw = av.heading - lane_ref.heading_at(current_x)
        current_speed = av.velocity
        ego_length = av.LENGTH
        current_timestamp = self.time

        # 控制和规划周期不一样，控制25ms规划100ms
        if self.frame % self.LP_FREQUENCY_DIVISOR == 0:
            if self.first_run is True:
                plan_start_x = current_x
                plan_start_y = current_y
                plan_start_heading = current_yaw
                plan_start_vx = current_speed * np.cos(plan_start_heading)
                plan_start_vy = current_speed * np.sin(plan_start_heading)
                plan_start_ax = 0
                plan_start_ay = 0
                stitch_x = np.array([])
                stitch_y = np.array([])
                stitch_s = np.array([0])
                stitch_s_dot = np.array([])
                stitch_theta = np.array([])
                stitch_s_dot_2 = np.array([])
            else:
                traj_x, traj_y, traj_h, s, s_dot, s_dot2 = self.pre_traj
                time_index = self.LP_FREQUENCY_DIVISOR * 2
                plan_start_x = traj_x[time_index]
                plan_start_y = traj_y[time_index]
                plan_start_heading = traj_h[time_index]
                plan_start_vx = s_dot[time_index] * np.cos(plan_start_heading)
                plan_start_vy = s_dot[time_index] * np.sin(plan_start_heading)
                tor = np.array([np.cos(plan_start_heading), np.sin(plan_start_heading)])
                # nor = np.array([-np.sin(plan_start_heading), np.cos(plan_start_heading)])
                a_tor = s_dot2[time_index] * tor
                # a_nor = s_dot[time_index] ** 2 * traj_k[time_index] * nor
                plan_start_ax = a_tor[0]
                plan_start_ay = a_tor[1]
                stitch_x = traj_x[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                stitch_y = traj_y[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                # 这个地方主要是因为不想重复计算s了，程序偷懒，多取一位
                stitch_s = s[time_index - self.LP_FREQUENCY_DIVISOR:time_index+1] - s[time_index - self.LP_FREQUENCY_DIVISOR]
                stitch_s_dot = s_dot[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                stitch_s_dot_2 = s_dot2[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                stitch_theta = traj_h[time_index - self.LP_FREQUENCY_DIVISOR:time_index]

            begin = time.time()
            # 相对时间置0， 其作用是可用来索引期望的s
            self.relative_time = 0
            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - self.prev_timestamp,
                                                                            av.velocity)
            lookahead = 60  # self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed        #直接设置成60m
            self.bp.set_lookahead(lookahead)
            # car_state
            # ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            ego_state = [plan_start_x, plan_start_y, plan_start_heading, open_loop_speed, ego_length]
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints,
                                                        ego_state)

            # Calculate planned paths in the local frame.  在车辆坐标系下计算局部路径坐标
            paths, path_validity = self.lp.plan_paths(goal_state_set)

            # Transform those paths back to the global frame.  计算全局坐标系的路径坐标
            self.paths = np.array(local_planner.transform_paths(paths, ego_state))

            # self.paths的维度是Nx3xM
            index2s_set = traj_index_trans2s_vec(self.paths[:, 0, :], self.paths[:, 1, :])
            # 动态障碍物网格划分
            time_step = np.linspace(0, lookahead / av.velocity, 11)

            best_index = self.select_best_path_index_numpy_dyn(av, dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy,
                                                               index2s_set, time_step)
            self.best_index = best_index
            self.traj = self.paths[best_index, ...]
            self.traj_xy = lane_ref.position(self.traj[0, :].reshape(-1, 1), self.traj[1, :].reshape(-1, 1)).T

            # 计算生成的局部路径的航向角和曲率
            self.theta, self.trajectory_kappa_init = EM_Planner.cal_heading_kappa(self.traj[0, :], self.traj[1, :])

            # 寻找障碍物的匹配点
            proj_x_set, proj_y_set, proj_heading_set, match_path_kapp, proj_match_point_index_set = \
                EM_Planner.find_match_point(dyn_obs_x, dyn_obs_y, self.traj[0, :], self.traj[1, :], self.theta,
                                            self.trajectory_kappa_init)
            self.index2s = index2s_set[best_index, :]

            # 计算每个障碍物在SL坐标系下的位置
            s_set, l_set = EM_Planner.CalcSL(dyn_obs_x, dyn_obs_y, self.traj[0, :], self.traj[1, :], proj_x_set,
                                             proj_y_set, proj_heading_set, proj_match_point_index_set, self.index2s)
            # 计算每个障碍物在SL坐标系下的速度
            l_dot_set, s_dot_set = EM_Planner.CalcDot_SL(l_set, dyn_obs_vx, dyn_obs_vy, proj_heading_set,
                                                         match_path_kapp)
            # 根据障碍物的SL坐标系的速度和位置生成ST图
            obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set = \
                EM_Planner.generate_st_graph(s_set, l_set, s_dot_set, l_dot_set)

            # 计算av的初始规划条件
            plan_start_s_dot, plan_start_s_dot2 = \
                EM_Planner.calc_speed_planning_start_condition(plan_start_vx, plan_start_vy,
                                                               plan_start_ax, plan_start_ay, plan_start_heading)
            # 速度的动态规划模块
            self.speed_planner.set_s_list(self.index2s[-1])
            dp_speed_s, dp_speed_t = self.speed_planner.speed_dp(obs_st_s_in_set, obs_st_s_out_set,
                                                                 obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot)
            # self.speed_planner.plot_ST_graph(obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
            #                                  dp_speed_s, dp_speed_t)
            if len(dp_speed_s) > 1:
                # 不知道什么原因，规划的路径有可能小于60，那我直接赋值就可以了
                self.speed_planner.set_s_list(self.index2s[-1])
                # 根据动态规划结果生成凸空间并计算凸空间的上下界约束
                s_lb, s_ub, s_dot_lb, s_dot_ub = \
                    self.speed_planner.convex_space_gen(dp_speed_s, dp_speed_t, self.index2s, obs_st_s_in_set,
                                                        obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                        self.trajectory_kappa_init, self.max_lateral_accel)
                # 二次规划解决速度规划问题
                qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init, status = \
                    self.speed_planner.speed_planning_osqp(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t,
                                                         s_lb, s_ub, s_dot_lb, s_dot_ub)
                if status != 'solved':
                    print('二次规划失败')
                    state = [av.position[0], av.position[1], av.heading]
                    steer = self.pp_controller_xy(state, current_speed)
                    return [-4, steer]
                # 为二次规划算的变量增密
                self.s, self.s_dot, s_dot2, self.relative_time_set = \
                    self.speed_planner.increase_points(qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init)

                self.s_final = np.hstack((stitch_s[:-1], self.s + stitch_s[-1]))
                self.s_dot_final = np.hstack((stitch_s_dot, self.s_dot))
                self.traj_x_final = np.hstack((stitch_x, self.traj[0, :]))
                self.traj_y_final = np.hstack((stitch_y, self.traj[1, :]))
                self.theta_final = np.hstack((stitch_theta, self.theta))
                s_dot2_final = np.hstack((stitch_s_dot_2, s_dot2))
                index2s = traj_index_trans2s(self.traj_x_final, self.traj_y_final)
                self.pre_traj = self.speed_planner.merge_path(self.s_final, index2s, self.traj_x_final, self.traj_y_final,
                                                              self.theta_final, self.s_dot_final, s_dot2_final)
                self.first_run = False

            print('time cost: ', time.time() - begin)
            self.prev_timestamp = self.time
            # self.s_pid.clear_err()
            # self.v_pid.clear_err()

        targets_index = int(self.relative_time * 40)
        target_s = self.s_final[targets_index]
        # 找到车辆在当前位置在s方向的投影
        real_s, _, _, _, _ = \
            EM_Planner.find_match_point(np.array([current_x - self.traj_x_final[0]]),
                                        np.array([current_y - self.traj_y_final[0]]),
                                        self.traj_x_final - self.traj_x_final[0], self.traj_y_final - self.traj_y_final[0],
                                        self.theta_final, self.trajectory_kappa_init)
        err_s = target_s - real_s[0]
        delta_pos = self.s_pid.output_cal(err_s)
        acc = self.v_pid.output_cal(self.s_dot_final[targets_index] - current_speed) + delta_pos

        state = [av.position[0], av.position[1], av.heading]
        steer = self.pp_controller_xy(state, current_speed)

        self.relative_time += 0.025
        self.frame += 1
        # print('the target speed: ', self.s_dot[target_index], 'the real speed: ', current_speed)
        print('the cal action: ', [acc, steer])
        return [acc, steer]

    def select_best_path_index_numpy_dyn(self, av, dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy, index2s_set, time_step):
        # av_lane_index = av.lane_index[2]
        path_num = self.paths.shape[0]
        cost4 = virtual_dynamic_obs_planning(dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy, index2s_set,
                                             self.paths[:, 0, :], self.paths[:, 1, :], av.velocity, time_step)
        # 与上一次选的路径越接近权重越小
        # cost1 = np.abs(np.sum(self.paths[:, 1, :] - av_lane_index * 4, axis=1))
        cost1 = np.abs(np.arange(9) - self.best_index) * 10
        global_path_index = len(self.paths) // 2
        # 与全局路径越接近,规划的路径与全局路径最接近的是中间路径
        cost2 = np.sqrt(np.square(np.arange(0, path_num) - np.ones(path_num) * global_path_index))
        # 超过道路边界
        cost3 = np.zeros(path_num)
        for i in range(path_num):
            if np.all(self.paths[i, 1, :] < 8.1) and np.all(self.paths[i, 1, :] > -1.1):
                cost3[i] = 0
            else:
                cost3[i] = float('inf')
        # 障碍物cost
        # lane_index = (paths[:, 1, -1] + 2) // 4
        # lane_index = np.clip(lane_index, 0, 2).astype(np.int32)
        weight = np.array([[0.4, 35, 1, 100]])      # np.array([[0.3, 40, 1, 100]])
        cost = weight @ np.vstack((cost1, cost2, cost3, cost4))
        if np.all(cost == np.inf):
            best_index = global_path_index
        else:
            best_index = np.argmin(cost)
        return best_index

    def pp_controller_(self, state, speed):
        l = 5.0
        pre_index = max(min(int(speed * 5), 300), 50)
        pre_ref_state = self.traj[:, pre_index]
        alpha = np.arctan2(pre_ref_state[1] - state[1], pre_ref_state[0] - state[0]) - state[2]
        ld = np.sqrt((pre_ref_state[1] - state[1]) ** 2 + (pre_ref_state[0] - state[0]) ** 2)
        u = np.arctan(2 * l * np.sin(alpha) / ld)
        # print(u)
        return np.clip(u, -np.pi/4, np.pi/4)

    def pp_controller_xy(self, state, speed):
        l = 5.0
        pre_index = max(min(int(speed * 5), 300), 50)
        pre_ref_state = self.traj_xy[:, pre_index]
        alpha = np.arctan2(pre_ref_state[1] - state[1], pre_ref_state[0] - state[0]) - state[2]
        ld = np.sqrt((pre_ref_state[1] - state[1]) ** 2 + (pre_ref_state[0] - state[0]) ** 2)
        u = np.arctan(2 * l * np.sin(alpha) / ld)
        return np.clip(u, -np.pi/4, np.pi/4)

    def calc_car_dist(self, car_1, car_2):
        dist = np.linalg.norm([car_1[0] - car_2[0], car_1[1] - car_2[1]]) - \
            (car_1[4] + car_2[4]) / 2
        return dist

    def generate_control(self, local_waypoints, vehicle):
        local_waypoints = np.vstack(local_waypoints)
        result_x = local_waypoints[:, 0]
        result_y = local_waypoints[:, 1]
        yaw_global = np.arctan2(np.diff(result_y), np.diff(result_x))
        yaw_global = np.append(yaw_global, yaw_global[-1])
        yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
        yaw_global = yaw_global.reshape(-1, 1)
        local_waypoints = np.concatenate((local_waypoints, yaw_global), axis=1)

        speeds_x = local_waypoints[:, 2] * np.cos(local_waypoints[:, 3])
        speeds_y = local_waypoints[:, 2] * np.sin(local_waypoints[:, 3])

        init_state = {
            'px': vehicle.position[0],
            'py': vehicle.position[1],
            'v': vehicle.velocity,
            'heading': vehicle.heading,
        }

        target_state = {
            'px': result_x[1],
            'py': result_y[1],
            'vx': speeds_x[1],
            'vy': speeds_y[1],
        }

        vel_len = vehicle.LENGTH / 1.7
        dt = self.dt

        x0_vals = np.array([init_state['px'], init_state['py'],
                            init_state['v'], init_state['heading']])
        x1_vals = np.array(
            [target_state['px'], target_state['py'], target_state['vx'], target_state['vy']])
        u0 = np.array([0, 0])
        bnds = ((self.hyperparameters['min_accel'], self.hyperparameters['max_accel']),
                (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))

        # Minimize difference between simulated state and next state by varying input u
        u0 = minimize(self.position_orientation_objective, u0, args=(x0_vals, x1_vals, vel_len, dt),
                      options={'disp': False, 'maxiter': 100, 'ftol': 1e-9},
                      method='SLSQP', bounds=bnds).x

        # Get simulated state using the found inputs

        # x1_sim_array = self.vehicle_dynamic(init_state, u0, vel_len, dt)
        # x1_sim = np.array([x1_sim_array['px'], x1_sim_array['py'],
        #                    x1_sim_array['v'] * np.cos(x1_sim_array['heading']),
        #                    x1_sim_array['v'] * np.sin(x1_sim_array['heading'])])
        return u0

    def vehicle_dynamic(self, state, action, vel_len, dt):
        init_state = state
        a, rot = action

        final_state = {"px": 0, 'py': 0, 'v': 0, 'heading': 0 }
        # 首先根据旧速度更新本车位置
        # 更新本车转向角
        final_state['heading'] = init_state['heading'] + \
            init_state['v'] / vel_len * np.tan(rot) * dt

        # 更新本车速度
        final_state['v'] = init_state['v'] + a * dt

        # 更新X坐标
        final_state['px'] = init_state['px'] + init_state['v'] * \
            dt * np.cos(init_state['heading'])  # *np.pi/180

        # 更新Y坐标
        final_state['py'] = init_state['py'] + init_state['v'] * \
            dt * np.sin(init_state['heading'])  # *np.pi/180

        return final_state

    def pp_controller(self, state, speed):
        l = 5.0
        pre_index = max(min(int(speed * 0.6), 30), 20)
        pre_ref_state =np.array(self.paths[self.best_index])[:, pre_index]
        alpha = np.arctan2(pre_ref_state[1] - state[1], pre_ref_state[0] - state[0]) - state[2]
        ld = np.sqrt((pre_ref_state[1] - state[1]) ** 2 + (pre_ref_state[0] - state[0]) ** 2)
        u = np.arctan(2 * l * np.sin(alpha) / ld)
        return np.clip(u, -np.pi/4, np.pi/4)

    def position_orientation_objective(self, u: np.array, x0_array: np.array, x1_array: np.array, vel_l: float, dt: float, e: np.array = np.array([2e-3, 2e-3, 3e-3])) -> float:
        """
        Position-Orientation objective function to be minimized for the state transition feasibility.

        Simulates the next state using the inputs and calculates the norm of the difference between the
        simulated next state and actual next state. Position, velocity and orientation state fields will
        be used for calculation of the norm.

        :param u: input values
        :param x0: initial state values
        :param x1: next state values
        :param dt: delta time
        :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
        :param ftol: ftol parameter used by the optimizer
        :param e: error margin, function will return norm of the error vector multiplied with 100 as cost
            if the input violates the friction circle constraint or input bounds.
        :return: cost
        """
        x0 = {
            'px': x0_array[0],
            'py': x0_array[1],
            'v': x0_array[2],
            'heading': x0_array[3],
        }

        x1_target = x1_array
        x1_sim = self.vehicle_dynamic(x0, u, vel_l, dt)
        x1_sim_array = np.array([x1_sim['px'], x1_sim['py'], x1_sim['v'] *
                                np.cos(x1_sim['heading']), x1_sim['v']*np.sin(x1_sim['heading'])])

        # if the input violates the constraints
        if x1_sim is None:
            return np.linalg.norm(e * 100)

        else:
            diff = np.subtract(x1_target, x1_sim_array)
            cost = np.linalg.norm(diff)
            return cost

    def select_best_path_index(self, obs, av, paths):
        """
        1、距离当前车道越接近，权重越小
        2、备选路径超过边界，权重为inf
        3、距离全局路径越接近，权重越小
        4、检测碰撞风险越高，权重越大
        """
        # 获取当前的av车道
        av_lane_index = av.lane_index[2]
        lane_available = [True] * 3
        f1, r1, f0, r0, f2, r2 = obs
        if r0 is not None:  # 判断左后方车辆是否有威胁
            s0 = r0.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1**2 + av.position[1] - r0.position[1]
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            if s1 <= s0:    # 有威胁
                lane_available[r0.lane_index[2]] = False

        if r2 is not None:  # 判断右后方车辆是否有威胁
            s0 = r2.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1 ** 2 + av.position[1] - r2.position[1]
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            if s1 <= s0:    # 有威胁
                lane_available[r2.lane_index[2]] = False

        if f0 is not None and self.bp._lookahead >= f0.position[1] - av.position[1]:  # 判断左前方车辆是否有威胁
            s0 = f0.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1 ** 2 + f0.position[1] - av.position[1]
            if s1 >= s0:    # 有威胁
                lane_available[f0.lane_index[2]] = False

        if f2 is not None and self.bp._lookahead >= f2.position[1] - av.position[1]:  # 判断右前方车辆是否有威胁
            s0 = f2.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1 ** 2 + f2.position[1] - av.position[1]
            if s1 >= s0:    # 有威胁
                lane_available[f2.lane_index[2]] = False

        # 距离当前车道越接近，权重越小
        cost1 = [0] * len(paths)
        for i in range(len(paths)):
            cost1[i] = sum(abs(paths[i][1] - av_lane_index * 4))

        # 距离全局路径越接近，权重越小
        cost2 = [0] * len(paths)
        global_path_index = len(paths) // 2
        for i in range(len(paths)):
            diff = i - global_path_index
            cost2[i] = np.sqrt(diff ** 2)

        # 备选路径超过边界，权重为inf
        cost3 = [0] * len(paths)
        for i in range(len(paths)):
            if np.all(np.array(paths[i][1]) < 8.1) and np.all(np.array(paths[i][1]) > -0.1):
                cost3[i] = 0
            else:
                cost3[i] = float('inf')

        # 检测碰撞风险越高，权重越大
        cost4 = [0] * len(paths)
        for i in range(len(paths)):
            index = (paths[i][-1] + 2) // 4
            if index < 0:
                index = 0
            elif index > 2:
                index = 2
            cost4[i] = 100 * lane_available[index]

        weight = [1, 1, 1, 1]
        best_index = len(paths) // 2
        min_cost = float('inf')
        for i in range(len(paths)):
            cost = weight[0] * cost1 + weight[1] * cost2 + weight[2] * cost3 + weight[3] * cost4
            if min_cost > cost:
                best_index = i
                min_cost = cost

        return best_index

    def select_best_path_index_numpy(self, obs, av, paths):
        """
        1、距离当前车道越接近，权重越小
        2、备选路径超过边界，权重为inf
        3、距离全局路径越接近，权重越小
        4、检测碰撞风险越高，权重越大
        """
        # 获取当前的av车道
        av_lane_index = av.lane_index[2]
        lane_not_available =np.array([False] * 3)
        path_len = len(paths)
        f1, r1, f0, r0, f2, r2 = obs
        if r0 is not None:  # 判断左后方车辆是否有威胁
            s0 = r0.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1**2 + av.position[1] - r0.position[1]
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            if s1 <= s0:    #有威胁
                lane_not_available[r0.lane_index[2]] = True

        if r2 is not None:  # 判断右后方车辆是否有威胁
            s0 = r2.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1 ** 2 + av.position[1] - r2.position[1]
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            if s1 <= s0:    # 有威胁
                lane_not_available[r2.lane_index[2]] = True

        if f0 is not None and self.bp._lookahead >= f0.position[1] - av.position[1]:  # 判断左前方车辆是否有威胁
            s0 = f0.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1 ** 2 + f0.position[1] - av.position[1]
            if s1 >= s0:    # 有威胁
                lane_not_available[f0.lane_index[2]] = True

        if f2 is not None and self.bp._lookahead >= f2.position[1] - av.position[1]:  # 判断右前方车辆是否有威胁
            s0 = f2.velocity * 1 + 0.5 * self.hyperparameters['min_accel'] * 1 ** 2
            s1 = av.velocity * 1 + 0.5 * self.hyperparameters['max_accel'] * 1 ** 2 + f2.position[1] - av.position[1]
            if s1 >= s0:    # 有威胁
                lane_not_available[f2.lane_index[2]] = True

        paths_np = np.array(paths)
        cost1 = np.abs(np.sum(paths_np[:, 1, :] - av_lane_index * 4, axis=1))
        global_path_index = len(paths) // 2
        cost2 =np.sqrt(np.square(np.arange(0, path_len) - np.ones(path_len) * global_path_index))
        cost3 = np.zeros(path_len)
        for i in range(path_len):
            if np.all(paths_np[i, 1, :] < 8.1) and np.all(paths_np[i, 1, :] > -0.1):
                cost3[i] = 0
            else:
                cost3[i] = float('inf')
        lane_index = (paths_np[:, 1, -1] + 2) // 4
        lane_index = np.clip(lane_index, 0, 2).astype(np.int32)
        cost4 = lane_not_available[lane_index] * 100
        weight = np.array([[0.6, 40, 1, 1]])
        cost = weight @ np.vstack((cost1, cost2, cost3, cost4))
        best_index = np.argmin(cost)
        return best_index

class MyMDPVehicle(MDPVehicle):
    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.cycle = 0.04
        self.local_planner = Lattice_Planner(self.cycle)
        self.global_update_fre = 1
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def update_global_path(self, points, time, fre):
        t = time / fre
        if t % self.global_update_fre == 0:
            self.local_planner.set_global_points(points)

    def act(self, action=None, essential_flag=False, t=0):
        self.local_planner.time = t * self.cycle
        # actions = self.local_planner.exec_waypoint_nav_demo_em(self)

        lane = self.road.network.get_lane(("a", "b", 0))
        actions = self.local_planner.exec_waypoint_nav_demo_em_stitch(self, lane)

        self.follow_road()  # 判断有没有到道路的尽头，到了就自动转到下一个开始
        action = {'steering': actions[1],
                  'acceleration': actions[0]}
        self.longi_acc = actions[0]
        # action = {'steering': 0,
        #           'acceleration': 0}
        # print(action)
        self.vehicle_plot()
        super(ControlledVehicle, self).act(action)

    def reset_path(self):
        self.local_planner.vehicle_path_reset()
        self.local_planner.time = 0

    def vehicle_plot(self):
        """
        绘制全局路径和局部路径以及各车辆之间的位置关系
        """
        self.ax.clear()
        lane = self.road.network.get_lane(("a", "b", 0))
        vehicles = self._get_range_vehicle(150, 150)
        pos = (- 5 / 2,  self.position[1]-2.5/2)  # 矩形右下角坐标位置
        rect = patches.Rectangle(pos, 5, 2, linewidth=1, edgecolor='r', facecolor='r')
        self.ax.add_patch(rect)
        for v in vehicles:
            x = self.position[0]
            pos = (v.position[0]-5/2-x, v.position[1]-2.5/2)  # 矩形右下角坐标位置
            rect = patches.Rectangle(pos, 5, 2, linewidth=1, edgecolor='m', facecolor='m')
            self.ax.add_patch(rect)
        self.ax.set_xlim(0, 150)
        self.ax.set_ylim(-50, 50)
        self.ax.set_aspect('equal')
        g_traj = lane.position(self.local_planner.waypoints[:, 0].reshape(-1, 1),
                               self.local_planner.waypoints[:, 1].reshape(-1, 1))
        self.ax.plot(g_traj[:, 0] - self.position[0], g_traj[:, 1], color='blue', linewidth=2.5, alpha=0.5)
        for i, path in enumerate(self.local_planner.paths):
            position = lane.position(path[0, :].reshape(-1, 1), path[1, :].reshape(-1, 1))
            if i == self.local_planner.best_index:
                self.ax.plot(position[:, 0] - self.position[0], position[:, 1], color='red', linewidth=1.5, alpha=0.5)
            else:
                self.ax.plot(position[:, 0] - self.position[0], position[:, 1], color='green', linewidth=1.5, alpha=0.5)
        self.ax.invert_yaxis()
        plt.draw()
        plt.pause(0.01)
        # plt.show()

    def close_plot(self):
        plt.close()



