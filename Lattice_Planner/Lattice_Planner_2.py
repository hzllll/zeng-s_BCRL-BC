from __future__ import print_function
from __future__ import division

# System level imports
import os
import sys
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)
import math
import numpy as np
import matplotlib.pyplot as plt
import controller2d
import local_planner
import behavioural_planner

from scipy.interpolate import interp1d
from scipy.optimize import minimize
import copy

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

class Lattice_Planner_2():

    def __init__(self, observation, traj, waypoints, scenario:dict):

        self.start = {
            'x' : observation['vehicle_info']['ego']['x'],
            'y' : observation['vehicle_info']['ego']['y'],
            'yaw' : observation['vehicle_info']['ego']['yaw']
        }
        self.goal = observation['test_setting']['goal']

        self.hyperparameters = {
            # 'max_accel': 10.0,
            'max_accel': 10.0,
            'max_steering': 0.7,
            'overtaking_speed': 15
        }

        # global path
        if waypoints == []:
            self.waypoints = self.generate_global_path(self.start['x'], self.start['y'], np.mean(self.goal['x']), np.mean(self.goal['y']))
        else:
            self.waypoints = waypoints

        self.pre_local_waypoints = None

        self.pre_u0 = np.array([0.0, 0.0])
        
        self.controller = controller2d.Controller2D(self.waypoints)

        """
        Configurable params
        """

        # Planning Constants
        # self.NUM_PATHS = 11
        self.NUM_PATHS = 15
        self.BP_LOOKAHEAD_BASE      = 15.0             # m
        self.BP_LOOKAHEAD_TIME      = 1.5              # s
        self.PATH_OFFSET            = 1.0              # m
        self.CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
        self.CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
        self.TIME_GAP               = 1.0              # s
        self.PATH_SELECT_WEIGHT     = 10
        self.A_MAX                  = self.hyperparameters['max_accel']              # m/s^2
        self.SLOW_SPEED             = 2.0              # m/s
        self.STOP_LINE_BUFFER       = 3.5              # m
        self.LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the
                                                # local planner operate at a lower
                                                # frequency than the controller
                                                # (which operates at the simulation
                                                # frequency). Must be a natural
                                                # number.

        # Path interpolation parameters
        self.INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points
       
        # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
        stopsign_fences = np.array([
            [self.goal['x'][0], self.goal['y'][0], self.goal['x'][1], self.goal['y'][1]]
        ])
        self.lp = local_planner.LocalPlanner(self.NUM_PATHS,
                                        self.PATH_OFFSET,
                                        self.CIRCLE_OFFSETS,
                                        self.CIRCLE_RADII,
                                        self.PATH_SELECT_WEIGHT,
                                        self.TIME_GAP,
                                        self.A_MAX,
                                        self.SLOW_SPEED,
                                        self.STOP_LINE_BUFFER)
        self.bp = behavioural_planner.BehaviouralPlanner(self.BP_LOOKAHEAD_BASE, stopsign_fences)
        self.best_index = int(self.NUM_PATHS / 2)
        # 计时
        self.time = 0
        # # 高斯分布获得的随机数
        # self.a = guass['a']
        # self.b = guass['b']
        # self.c = guass['c']
        # self.d = guass['d']

    # 控制输出接口
    def send_control_command(self, throttle=0.0, steer=0.0, brake=0.0,
                            hand_brake=False, reverse=False):

        # control = VehicleControl()

        # Clamp all values within their limits
        steer = np.fmax(np.fmin(steer, 1.0), -1.0)
        throttle = np.fmax(np.fmin(throttle, 1.0), 0)
        brake = np.fmax(np.fmin(brake, 1.0), 0)

        if throttle > 0:
            accelerate = throttle
        else:
            accelerate = - brake

        # 映射到加减速度范围为[-3, 3]
        accelerate *= self.hyperparameters['max_accel']

        u0 = np.array([accelerate, steer])

        return u0

    def exec_waypoint_nav_demo(self, observation, traj, road_info, frame):
        
        lead_car_found = False
        follow_car_found = False

        lead_car_info = {
            'middle':{
                'pre_long': float('inf')
            },
            'left':{
                'pre_long': float('inf')
            },
            'right':{
                'pre_long': float('inf')
            },
        }

        follow_car_info = {
            'middle':{
                'pre_long': -float('inf')
            },
            'left':{
                'pre_long': -float('inf')
            },
            'right':{
                'pre_long': -float('inf')
            },
        }

        for vehi in observation['vehicle_info'].items():
            self_x = observation['vehicle_info']['ego']['x']
            self_y = observation['vehicle_info']['ego']['y']
            self_yaw = observation['vehicle_info']['ego']['yaw']
            self_width = observation['vehicle_info']['ego']['width']
            self_length = observation['vehicle_info']['ego']['length']
            # road_width = abs(road_info.discretelanes[0].left_vertices[0][1] - \
            #                  road_info.discretelanes[0].right_vertices[0][1])
            road_width = 3.5
            id = vehi[0]
            if id != 'ego':
                vec_x = vehi[1]['x'] - self_x
                vec_y = vehi[1]['y'] - self_y
                long = vec_x * math.cos(self_yaw) + vec_y * math.sin(self_yaw)
                lat = - vec_x * math.sin(self_yaw) + vec_y * math.cos(self_yaw)
                err = road_width + (vehi[1]['width'] + self_width) / 2
                # if long > 0:
                #     if abs(lat) < (vehi[1]['width'] + self_width) / 2 + 0.8:
                #         if long < lead_car_info['middle']['pre_long']:
                #             lead_car_info['middle']['pre_long'] = long
                #             for key in vehi[1].keys():
                #                 lead_car_info['middle'][key] = vehi[1][key]
                #             lead_car_found = True
                #     elif lat > 0:
                #         if long < lead_car_info['left']['pre_long'] and \
                #             lat < err:
                #             lead_car_info['left']['pre_long'] = long
                #             for key in vehi[1].keys():
                #                 lead_car_info['left'][key] = vehi[1][key]
                #     else:
                #         if long < lead_car_info['right']['pre_long'] and \
                #             lat > -err:
                #             lead_car_info['right']['pre_long'] = long
                #             for key in vehi[1].keys():
                #                 lead_car_info['right'][key] = vehi[1][key]
                # else:
                #     if abs(lat) < (vehi[1]['width'] + self_width) / 2 + 0.8:
                #         if long > follow_car_info['middle']['pre_long']:
                #             follow_car_info['middle']['pre_long'] = long
                #             for key in vehi[1].keys():
                #                 follow_car_info['middle'][key] = vehi[1][key]
                #             follow_car_found = True
                #     elif lat > 0:
                #         if long > follow_car_info['left']['pre_long'] and \
                #             lat < err:
                #             follow_car_info['left']['pre_long'] = long
                #             for key in vehi[1].keys():
                #                 follow_car_info['left'][key] = vehi[1][key]
                #     else:
                #         if long > follow_car_info['right']['pre_long'] and \
                #             lat > -err:
                #             follow_car_info['right']['pre_long'] = long
                #             for key in vehi[1].keys():
                #                 follow_car_info['right'][key] = vehi[1][key]
                if long > 0:
                    if abs(lat) < (vehi[1]['width'] + self_width) / 2 + 0.8 and \
                            lead_car_info['middle']['pre_long'] > long > (vehi[1]['length'] + self_length) / 2:
                        lead_car_info['middle']['pre_long'] = long
                        for key in vehi[1].keys():
                            lead_car_info['middle'][key] = vehi[1][key]
                        lead_car_found = True
                    elif 0 < lat < err and \
                            long < lead_car_info['left']['pre_long']:
                        lead_car_info['left']['pre_long'] = long
                        for key in vehi[1].keys():
                            lead_car_info['left'][key] = vehi[1][key]
                    elif 0 > lat > -err and \
                            long < lead_car_info['right']['pre_long']:
                        lead_car_info['right']['pre_long'] = long
                        for key in vehi[1].keys():
                            lead_car_info['right'][key] = vehi[1][key]
                else:
                    if abs(lat) < (vehi[1]['width'] + self_width) / 2 + 0.8 and \
                            follow_car_info['middle']['pre_long'] < long < -(vehi[1]['length'] + self_length) / 2:
                        follow_car_info['middle']['pre_long'] = long
                        for key in vehi[1].keys():
                            follow_car_info['middle'][key] = vehi[1][key]
                        follow_car_found = True
                    elif 0 < lat < err and \
                            long > follow_car_info['left']['pre_long']:
                        follow_car_info['left']['pre_long'] = long
                        for key in vehi[1].keys():
                            follow_car_info['left'][key] = vehi[1][key]
                    elif 0 > lat > -err and \
                            long > follow_car_info['right']['pre_long']:
                        follow_car_info['right']['pre_long'] = long
                        for key in vehi[1].keys():
                            follow_car_info['right'][key] = vehi[1][key]

        # Obtain Lead Vehicle information.
        if lead_car_found:
            lead_car_pos = np.array([
                [lead_car_info['middle']['x'], lead_car_info['middle']['y'], lead_car_info['middle']['yaw']]])
            lead_car_length = np.array([
                [lead_car_info['middle']['length']]])
            lead_car_speed = np.array([
                [lead_car_info['middle']['v']]])
        else:
            lead_car_pos = np.array([
                [99999.0, 99999.0, 0.0]])
            lead_car_length = np.array([
                [99999.0]])
            lead_car_speed = np.array([
                [99999.0]])
            
        # Obtain Follow Vehicle information.    
        if follow_car_found:
            follow_car_pos = np.array([
                [follow_car_info['middle']['x'], follow_car_info['middle']['y'], follow_car_info['middle']['yaw']]])
            follow_car_length = np.array([
                [follow_car_info['middle']['length']]])
            follow_car_speed = np.array([
                [follow_car_info['middle']['v']]])
        else:
            follow_car_pos = np.array([
                [99999.0, 99999.0, 0.0]])
            follow_car_length = np.array([
                [99999.0]])
            follow_car_speed = np.array([
                [99999.0]])
        
        # Obtain parkedcar_box_pts
        parkedcar_box_pts = []  # [x,y]
        # 与当前车道的前车进行速度判断，若小于当前车速按照障碍车绕行
        if lead_car_found and lead_car_info['middle']['v'] < self.hyperparameters['overtaking_speed'] and lead_car_info['middle']['v'] < observation['vehicle_info']['ego']['v']:#obs.v < self.v: #与onsite接口对接
            parkedcar_num = 0
            # 只需要当前和左右车道中最近的前车的信息
            for vehi in lead_car_info.items():
                id = vehi[0]
                if 'v' in lead_car_info[id].keys():

                    x, y, yaw, xrad, yrad = [vehi[1][key] for key in ['x', 'y', 'yaw', 'length', 'width']]

                    xrad = 0.5 * xrad # 0.5 * 车长
                    yrad = 0.5 * yrad # 0.5 * 车宽

                    cpos = np.array([
                        [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
                        [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
                    rotyaw = np.array([
                        [np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])
                    cpos_shift = np.array([
                        [x, x, x, x, x, x, x, x],
                        [y, y, y, y, y, y, y, y]])
                    cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

                    if len(parkedcar_box_pts) <= parkedcar_num:
                            parkedcar_box_pts.append([])

                    for j in range(cpos.shape[1]):
                        parkedcar_box_pts[parkedcar_num].append([cpos[0, j], cpos[1, j]])

                    parkedcar_num += 1
                

        local_waypoints = None
        path_validity = np.zeros((self.NUM_PATHS, 1), dtype=bool)
        
        reached_the_end = False

        # Update pose and timestamp
        prev_timestamp = observation['test_setting']['t']

        # 当前参数获取
        current_x, current_y, current_yaw, current_speed, ego_length = \
            [observation['vehicle_info']['ego'][key] for key in ['x', 'y', 'yaw', 'v', 'length']]
        current_timestamp = observation['test_setting']['t'] + \
            observation['test_setting']['dt']


        if frame % self.LP_FREQUENCY_DIVISOR == 0:

            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp, observation['vehicle_info']['ego']['v'])

            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_pos[0][2], lead_car_speed[0][0], lead_car_length[0][0]]
            follow_car_state = [follow_car_pos[0][0], follow_car_pos[0][1], follow_car_pos[0][2], follow_car_speed[0][0], follow_car_length[0][0]]
            
            # Set lookahead based on current speed.
            self.bp.set_lookahead(self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed)

            # # 时间
            # _time = self.bp._lookahead / observation['vehicle_info']['ego']['v']
            # print(f"用时：{_time}")

            # Perform a state transition in the behavioural planner.
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            # Check to see if we need to follow the lead vehicle.
            ego_rear_dist = self.calc_car_dist(ego_state, follow_car_state) - 2
            ego_rear_dist = ego_rear_dist if ego_rear_dist > 0 else 1e-8
            ego_front_dist = self.calc_car_dist(ego_state, lead_car_state) - 2
            ego_front_dist = ego_front_dist if ego_front_dist > 0 else 1e-8
            a = (2 * follow_car_speed[0][0] - current_speed) * current_speed / (2 * ego_rear_dist)
            a_x = max(1e-8, min(self.hyperparameters['max_accel'], a))
            follow_dist = current_speed ** 2 / (2 * a_x) - lead_car_speed[0][0] ** 2 / (2 * self.hyperparameters['max_accel'])
            LEAD_VEHICLE_LOOKAHEAD = follow_dist if follow_dist > 0 else 0
            # LEAD_VEHICLE_LOOKAHEAD = 15
            self.bp.check_for_lead_vehicle(ego_state, lead_car_pos[0], LEAD_VEHICLE_LOOKAHEAD)

            # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints, ego_state)
            # Calculate planned paths in the local frame.
            paths, path_validity = self.lp.plan_paths(goal_state_set)

            # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)

            # Perform collision checking.
            collision_check_array = self.lp._collision_checker.collision_check(paths, parkedcar_box_pts)

            # # Compute the best local path.
            # if self.lp._prev_best_path == None:
            #     best_index = int(len(paths) / 2)
            # else:
            #     best_index = self.lp._collision_checker.select_best_path_index(paths, collision_check_array, self.bp._goal_state)
            # # If no path was feasible, continue to follow the previous best path.
            # if best_index == None or paths == []:
            #     best_path = self.lp._prev_best_path
            # else:
            #     best_path = paths[best_index]
            #     self.lp._prev_best_path = best_path


            # 根据预测的周围车辆的未来轨迹，选择最佳路径
            if paths != []:
                best_index = self.select_best_path_index_future(observation, paths, traj, road_info)
                self.best_index = best_index

            # self.best_index = int(len(paths) / 2)

            if paths != []:
                best_path = paths[self.best_index]
                self.lp._prev_best_path = best_path
            else:
                best_path = self.lp._prev_best_path

            # # 画出路径
            # for i in range(len(paths)):
            #     path = paths[i]
            #     plt.plot(path[0], path[1], 'g')
            # plt.plot(best_path[0], best_path[1], 'r')
            # plt.pause(0.1)

            # Compute the velocity profile for the path, and compute the waypoints.
            # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
            # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
            if lead_car_found:
                desired_speed = np.sqrt(lead_car_speed[0][0] ** 2 + 2 * ego_front_dist * self.hyperparameters['max_accel'])
            else:
                desired_speed = self.bp._goal_state[2]
            decelerate_to_stop = self.bp._state == behavioural_planner.DECELERATE_TO_STOP
            local_waypoints = self.lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state,
                                                                            current_speed, decelerate_to_stop,
                                                                            lead_car_state, self.bp._follow_lead_vehicle)
            # --------------------------------------------------------------

            if local_waypoints != None:
                # Update the controller waypoint path with the best local path.
                # This controller is similar to that developed in Course 1 of this
                # specialization.  Linear interpolation computation on the waypoints
                # is also used to ensure a fine resolution between points.
                wp_distance = []  # distance array
                local_waypoints_np = np.array(local_waypoints)
                for i in range(1, local_waypoints_np.shape[0]):
                    wp_distance.append(
                        np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                                (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
                wp_distance.append(0)  # last distance is 0 because it is the distance
                # from the last waypoint to the last waypoint

                # Linearly interpolate between waypoints and store in a list
                wp_interp = []  # interpolated values
                # (rows = waypoints, columns = [x, y, v])
                for i in range(local_waypoints_np.shape[0] - 1):
                    # Add original waypoint to interpolated waypoints list (and append
                    # it to the hash table)
                    wp_interp.append(list(local_waypoints_np[i]))

                    # Interpolate to the next waypoint. First compute the number of
                    # points to interpolate based on the desired resolution and
                    # incrementally add interpolated points until the next waypoint
                    # is about to be reached.
                    num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                    float(self.INTERP_DISTANCE_RES)) - 1)
                    wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = self.INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                        wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                # add last waypoint at the end
                wp_interp.append(list(local_waypoints_np[-1]))

                # Update the other controller values and controls
                self.controller.update_waypoints(wp_interp)

        ###
        # Controller Update
        ###
        # if local_waypoints != None and local_waypoints != []:
        #     self.controller.update_values(current_x, current_y, current_yaw,
        #                             current_speed,
        #                             current_timestamp, frame)
        #     self.controller.update_controls()
        #     cmd_throttle, cmd_steer, cmd_brake = self.controller.get_commands()
        #     # 尝试输出控制值
        #     u0 = self.send_control_command(throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)
        # else:
        #     cmd_throttle = 0.0
        #     cmd_steer = 0.0
        #     cmd_brake = 0.0
        #     u0 = self.send_control_command(throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)
        
        if local_waypoints != None:
            self.pre_local_waypoints = local_waypoints

        if local_waypoints != None and local_waypoints != []:
            u0 = self.generate_control(local_waypoints, observation)
            self.pre_u0 = u0
        elif self.pre_local_waypoints != None:
            u0 = self.generate_control(self.pre_local_waypoints, observation)
            self.pre_u0 = u0
        else:
            u0 = self.pre_u0

        return u0

    def generate_global_path(self, x0, y0, x1, y1):
        # Create an array of x and y values for the two points
        x = np.array([x0, x1])
        y = np.array([y0, y1])
        
        # Create a function that interpolates between the two points
        f = interp1d(x, y, kind='linear')
        
        # Create an array of x values for the smooth path
        x_global = np.linspace(x0, x1, num=100)
        
        # Use the interpolation function to get the corresponding y values
        y_global = f(x_global)

        # yaw
        yaw_global = np.arctan2(np.diff(y_global), np.diff(x_global))
        yaw_global = np.append(yaw_global, yaw_global[-1])
        yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)

        waypoints = np.stack((x_global, y_global, yaw_global), axis=1)
    
        # Return the x and y values of the smooth path
        return waypoints

    def generate_control(self, local_waypoints, observation):

        # local_waypoints = np.vstack(local_waypoints)
        # result_x = local_waypoints[:, 0]
        # result_y = local_waypoints[:, 1]
        # yaw_global = np.arctan2(np.diff(result_y), np.diff(result_x))
        # yaw_global = np.append(yaw_global, yaw_global[-1])
        # yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
        # yaw_global = yaw_global.reshape(-1, 1)
        # local_waypoints = np.concatenate((local_waypoints, yaw_global), axis=1)
        #
        # speeds_x = local_waypoints[:, 2] * np.cos(local_waypoints[:, 3])
        # speeds_y = local_waypoints[:, 2] * np.sin(local_waypoints[:, 3])
        #
        # init_state = {
        #     'px': observation['vehicle_info']['ego']['x'],
        #     'py': observation['vehicle_info']['ego']['y'],
        #     'v': observation['vehicle_info']['ego']['v'],
        #     'heading': observation['vehicle_info']['ego']['yaw'],
        # }
        #
        # target_state = {
        #     'px': result_x[1],
        #     'py': result_y[1],
        #     'vx': speeds_x[1],
        #     'vy': speeds_y[1],
        # }
        #
        # vel_len = observation['vehicle_info']['ego']['length'] / 1.7
        # dt = observation['test_setting']['dt']
        #
        # x0_vals = np.array([init_state['px'], init_state['py'],
        #                    init_state['v'], init_state['heading']])
        # x1_vals = np.array(
        #     [target_state['px'], target_state['py'], target_state['vx'], target_state['vy']])
        # u0 = np.array([0, 0])
        # bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
        #         (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))
        #
        # # Minimize difference between simulated state and next state by varying input u
        # u0 = minimize(self.position_orientation_objective, u0, args=(x0_vals, x1_vals, vel_len, dt),
        #               options={'disp': False, 'maxiter': 100, 'ftol': 1e-9},
        #               method='SLSQP', bounds=bnds).x
        #
        # # Get simulated state using the found inputs
        #
        # # x1_sim_array = self.vehicle_dynamic(init_state, u0, vel_len, dt)
        # # x1_sim = np.array([x1_sim_array['px'], x1_sim_array['py'],
        # #                    x1_sim_array['v']*np.cos(x1_sim_array['heading']),
        # #                    x1_sim_array['v']*np.sin(x1_sim_array['heading'])])
        # #
        # # x_delta = np.linalg.norm(x1_sim-x1_vals)
        # # print('error:', x_delta)
        # # x1_sim = vehicle_dynamics.forward_simulation(x0_vals, u0, dt, throw=False)
        #
        # # print('action:', u0)


        # 0901修改输出为动作序列
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

        N = 5
        # N = len(result_x) - 2
        init_state = {
            'px': observation['vehicle_info']['ego']['x'],
            'py': observation['vehicle_info']['ego']['y'],
            'v': observation['vehicle_info']['ego']['v'],
            'heading': observation['vehicle_info']['ego']['yaw'],
        }
        target_state = {
            'px': result_x[1],
            'py': result_y[1],
            'vx': speeds_x[1],
            'vy': speeds_y[1],
        }

        vel_len = observation['vehicle_info']['ego']['length'] / 1.7
        # dt = observation['test_setting']['dt']
        bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
                (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))
        u0 = np.array([0, 0])
        # 初始化存10个动作序列
        u = np.zeros((N, 2))

        for num in range(1, N + 1):
            x0_vals = np.array([init_state['px'], init_state['py'], init_state['v'], init_state['heading']])
            x1_vals = np.array([target_state['px'], target_state['py'], target_state['vx'], target_state['vy']])

            dt = ((local_waypoints[num, 1] - local_waypoints[num - 1, 1])**2 + (local_waypoints[num, 0] - local_waypoints[num - 1, 0])**2)**0.5/local_waypoints[num - 1, 2]
            # Minimize difference between simulated state and next state by varying input u
            u0 = minimize(self.position_orientation_objective, u0, args=(x0_vals, x1_vals, vel_len, dt),
                          options={'disp': False, 'maxiter': 100, 'ftol': 1e-9},
                          method='SLSQP', bounds=bnds).x
            u[num - 1] = u0

            # Get simulated state using the found inputs
            x1_sim_array = self.vehicle_dynamic(init_state, u0, vel_len, dt)
            x1_sim = np.array([x1_sim_array['px'], x1_sim_array['py'],
                               x1_sim_array['v'] * np.cos(x1_sim_array['heading']),
                               x1_sim_array['v'] * np.sin(x1_sim_array['heading'])])

            x_delta = np.linalg.norm(x1_sim - x1_vals)
            # 将x1_sim_array['px']和x1_sim_array['py']和target_state['px']和target_state['py']画在图上
            # plt.plot(x1_sim_array['px'], x1_sim_array['py'], 'yo')
            # plt.plot(target_state['px'], target_state['py'], 'bo')

            init_state = {
                'px': x1_sim_array['px'],
                'py': x1_sim_array['py'],
                'v': x1_sim_array['v'],
                'heading': x1_sim_array['heading'],
            }
            target_state = {
                'px': result_x[num + 1],
                'py': result_y[num + 1],
                'vx': speeds_x[num + 1],
                'vy': speeds_y[num + 1],
            }

        # u0 = u[0]

        # u0 = u

        u0 = u
        dt = ((local_waypoints[0:N, 1] - local_waypoints[1:N+1,1])**2 + (local_waypoints[0:N, 0] - local_waypoints[1:N+1,0])**2)**0.5/local_waypoints[0:N, 2]
        t = np.zeros(N)
        t[0] = dt[0]
        for i in range(1, len(dt)):
            t[i] = t[i - 1] + dt[i]
        # 根据t和u0生成每隔0.04s的动作序列，采用插值的方法
        t_interp = np.arange(0, t[-1], 0.04)
        u0_interp = np.zeros((len(t_interp), 2))
        for i in range(2):
            u0_interp[:, i] = np.interp(t_interp, t, u0[:, i])

        # # 画出从初始位置用u0_interp的动作序列走到终点的轨迹
        x0 = observation['vehicle_info']['ego']['x']
        y0 = observation['vehicle_info']['ego']['y']
        v0 = observation['vehicle_info']['ego']['v']
        heading0 = observation['vehicle_info']['ego']['yaw']
        x = np.zeros(len(t_interp) + 1)
        y = np.zeros(len(t_interp) + 1)
        v = np.zeros(len(t_interp) + 1)
        heading = np.zeros(len(t_interp) + 1)
        x[0] = x0
        y[0] = y0
        v[0] = v0
        heading[0] = heading0
        for i in range(len(t_interp)):
            x[i + 1] = x[i] + v[i] * 0.04 * np.cos(heading[i])
            y[i + 1] = y[i] + v[i] * 0.04 * np.sin(heading[i])
            v[i + 1] = v[i] + u0_interp[i, 0] * 0.04
            heading[i + 1] = heading[i] + v[i] / vel_len * np.tan(u0_interp[i, 1]) * 0.04
        # # 用ro画出
        # plt.plot(x, y, 'ro')

        # 取u0_interp的前10个动作序列
        # u0_interp = u0_interp[0:10]

        self.time += 0.04

        return u0_interp[0]
    
    def vehicle_dynamic(self, state, action, vel_len, dt):
        init_state = state
        a, rot = action

        final_state = {
            'px': 0,
            'py': 0,
            'v': 0,
            'heading': 0
        }
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
    
    def calc_car_dist(self, car_1, car_2):
        dist = np.linalg.norm([car_1[0] - car_2[0], car_1[1] - car_2[1]]) - \
            (car_1[4] + car_2[4]) / 2
        return dist

    def create_controller_output_dir(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def write_control_file(self, u0):
        self.create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
        file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'control.txt')
        with open(file_name, 'a') as control_file:
                control_file.write('%.2f\t%.2f\n' %\
                                    (u0[0], u0[1]))

    def select_best_path_index_future(self, obs, path, traj, road_info):
        """
        根据预测的周围车辆的未来轨迹，选择最佳路径,参考如下代码
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
            # weight = np.array([[0.4, 35, 1, 100]])      # np.array([[0.3, 40, 1, 100]])
            weight = np.array([[0.4, 35, 1, 100]])
            cost = weight @ np.vstack((cost1, cost2, cost3, cost4))
            if np.all(cost == np.inf):
                best_index = global_path_index
            else:
                best_index = np.argmin(cost)
            return best_index

        def virtual_dynamic_obs_planning(x_set, y_set, vx_set, vy_set,index2s_vec, paths_x, paths_y, plan_start_v, time_step=np.linspace(0, 2, 11)):
            # time_step = np.linspace(0, 1, 11)
            factor = np.ones(len(time_step))    # np.linspace(1, 0.5, len(time_step))
            # 障碍物预测坐标
            bv_x_predict = x_set[:, np.newaxis] + vx_set[:, np.newaxis] @ time_step[np.newaxis, :]
            bv_y_predict = y_set[:, np.newaxis] + vy_set[:, np.newaxis] @ time_step[np.newaxis, :]
            # 自车预测s长度,预测k个步长
            s = plan_start_v * time_step
            # N条备选路径, 每条路径上M个点 delta(kxNxM),这里s[:, None, None]对s做升高维度操作,为的就是用每个s减去index2s_vec
            delta_s = np.abs(s[:, None, None] - index2s_vec)
            # 找到每个点最接近的位置, kxN，相当于用index2s_vec代替时间下的位置点
            min_index = np.argmin(delta_s, axis=2).T
            av_x_predict = paths_x[np.arange(paths_x.shape[0])[:, np.newaxis], min_index]
            av_y_predict = paths_y[np.arange(paths_y.shape[0])[:, np.newaxis], min_index]
            # 这一项判断只适合道路在直线的情况下使用
            dy = np.abs(av_y_predict[:, np.newaxis, :] - bv_y_predict[np.newaxis, :, :])
            dx = np.abs(av_x_predict[:, np.newaxis, :] - bv_x_predict[np.newaxis, :, :])
            y_cost = 1 / (1 + np.exp(np.minimum((dy - 3) * 5, 100)))
            x_cost = 1 / (1 + np.exp(np.minimum((dx - 15) * 5, 100)))      #限制指数最大为100，防止溢出
            single_cost = x_cost * y_cost
            collision_index = np.any((dy < 3) & (dx < 15), axis=(1, 2))
            # 衰减系数，越远的距离越小，因为预测越不准确而且还有一个问题就是越远的障碍物，我越能够通过速度规划来做最后的保证
            factor_cost = single_cost * factor[np.newaxis, np.newaxis, :]
            cost = np.sum(factor_cost, axis=(1, 2)) * 1000 + 100 * collision_index
            # path_index = np.argmin(cost)
            return cost
                """
        # 用3s的预测轨迹来选择最佳路径，3s内划分为11个时间片，path中的路径长度/av的速度可能不足3s
        # 所以先用av的速度补充到3s的路径，然后再用插值的方法获得11个时间片的路径点
        # 他车的3s的轨迹在traj中通过插值的方式获得11个时间片的路径点
        # 用11个时间片的他车未来路径点和path中的路径点计算cost
        av = obs['vehicle_info']['ego']['v']
        cost1 = np.abs(np.arange(len(path)) - self.best_index) * 10
        global_path_index = len(path) // 2
        cost2 = np.abs(np.arange(0, len(path)) - np.ones(len(path)) * global_path_index)
        cost3 = np.zeros(len(path))
        b = np.zeros(2)
        b[0] = road_info[0]['left_bound']
        b[1] = road_info[2]['right_bound']
        b.sort()
        for i in range(len(path)):
            if np.all(path[i][1] < b[1] - 0.5 * obs['vehicle_info']['ego']['width']) and np.all(path[i][1] > b[0] + 0.5 * obs['vehicle_info']['ego']['width']):
            # if np.all(path[i][1] < b[1]) and np.all(path[i][1] > b[0]):
                cost3[i] = 0
            else:
                cost3[i] = float('inf')

        # 障碍物cost
        # 根据当前时刻和traj信息，获取未来3s的障碍物位置，分为11个时间片，traj[1][0,t]为1车(t*0.04)时刻的x坐标
        # 预测时间可更改，最多为3s
        t = 3
        t1 = self.time
        step = t1 / 0.04
        if step > 40:
            t1 = t1
            t1 = t1
        t2 = t1 + t
        t_num = 11
        time = np.linspace(t1, t2, t_num)
        time_av = np.linspace(0, t, t_num)
        time_index = [int(t / 0.04) for t in time]
        dyn_obs_x = np.zeros((len(traj), t_num))
        dyn_obs_y = np.zeros((len(traj), t_num))
        for i in range(len(traj)):
            for j in range(t_num):
                dyn_obs_x[i][j] = traj[i+1][0][time_index[j]]
                dyn_obs_y[i][j] = traj[i+1][1][time_index[j]]

        # # 障碍物cost
        # # 匀速模型
        # t = 3
        # t1 = self.time
        # t2 = t1 + t
        # t_num = 11
        # time = np.linspace(t1, t2, t_num)
        # time_av = np.linspace(0, t, t_num)
        # dyn_obs_x = np.zeros((len(obs['vehicle_info']) - 1, t_num))
        # dyn_obs_y = np.zeros((len(obs['vehicle_info']) - 1, t_num))
        # for i, (vehicle_id, vehicle_info) in enumerate(obs['vehicle_info'].items()):
        #     if vehicle_id != 'ego':
        #         current_x = vehicle_info['x']
        #         current_y = vehicle_info['y']
        #         current_vx = vehicle_info['v'] * np.cos(vehicle_info['yaw'])
        #         current_vy = vehicle_info['v'] * np.sin(vehicle_info['yaw'])
        #         for j in range(t_num):
        #             dyn_obs_x[i - 1][j] = current_x + current_vx * time_av[j]
        #             dyn_obs_y[i - 1][j] = current_y + current_vy * time_av[j]

        # 根据path和av的速度，计算每条路径t_num个时间片的路径点
        path_num = len(path)
        path_new = copy.deepcopy(path)
        # 根据最后两个点的位置多补充一个点，y不变
        if path_new[0][0][-1] > path_new[0][0][-2]:
            k = 1
        else:
            k = -1
        for i in range(path_num):
            x = path_new[i][0][-1] + 50 * k
            y = path_new[i][1][-1]
            path_new[i][0] = np.append(path_new[i][0], x)
            path_new[i][1] = np.append(path_new[i][1], y)

        # 计算每条路径每个路径点之间的距离进行累加
        s = np.zeros((path_num, len(path_new[0][0])))
        for i in range(path_num):
            for j in range(1, len(path_new[0][0])):
                s[i][j] = s[i][j - 1] + np.sqrt((path_new[i][0][j] - path_new[i][0][j - 1]) ** 2 + (path_new[i][1][j] - path_new[i][1][j - 1]) ** 2)
        # 先计算时间，再用插值的方法获得t_num个时间片的路径点
        ts = s / av
        # 根据s和path进行插值得到路径点
        paths_x = np.zeros((path_num, t_num))
        paths_y = np.zeros((path_num, t_num))
        for i in range(path_num):
            paths_x[i] = np.interp(time_av, ts[i], path_new[i][0])
            paths_y[i] = np.interp(time_av, ts[i], path_new[i][1])

        # 计算cost4
        dx = np.abs(paths_x[:, np.newaxis, :] - dyn_obs_x[np.newaxis, :, :])
        dy = np.abs(paths_y[:, np.newaxis, :] - dyn_obs_y[np.newaxis, :, :])
        y_cost = 1 / (1 + np.exp(np.minimum((dy - 3) * 5, 100)))
        x_cost = 1 / (1 + np.exp(np.minimum((dx - 15) * 5, 100)))
        single_cost = x_cost * y_cost
        collision_index = np.any((dy < 3) & (dx < 15), axis=(1, 2))
        factor = np.ones(t_num)
        factor_cost = single_cost * factor[np.newaxis, np.newaxis, :]
        cost4 = np.sum(factor_cost, axis=(1, 2)) * 1000 + 100 * collision_index

        # # 计算cost4，用到每辆车的长度和宽度，纵向安全距离为10m，横向安全距离为0.5m
        # # 获取每辆车的长度和宽度，位于obs['vehicle_info']['1']-['n']中
        # dyn_obs_l = np.array([obs['vehicle_info'][i]['length'] for i in range(1, len(obs['vehicle_info']))])
        # dyn_obs_w = np.array([obs['vehicle_info'][i]['width'] for i in range(1, len(obs['vehicle_info']))])
        # ego_l = obs['vehicle_info']['ego']['length']
        # ego_w = obs['vehicle_info']['ego']['width']
        # dx = np.abs(paths_x[:, np.newaxis, :] - dyn_obs_x[np.newaxis, :, :])
        # dy = np.abs(paths_y[:, np.newaxis, :] - dyn_obs_y[np.newaxis, :, :])
        # # 用每辆车的长度和宽度，纵向安全距离为10m，横向安全距离为0.5m
        # y_cost = np.zeros_like(dy)
        # for i in range(dy.shape[1]):  # 遍历每辆车
        #     y_cost[:, i, :] = 1 / (1 + np.exp(np.minimum((dy[:, i, :] - (dyn_obs_w[i] + ego_w) / 2 - 0.5) * 5, 100)))
        # x_cost = np.zeros_like(dx)
        # for i in range(dx.shape[1]):
        #     x_cost[:, i, :] = 1 / (1 + np.exp(np.minimum((dx[:, i, :] - (dyn_obs_l[i] + ego_l) / 2 - 10) * 5, 100)))
        # y_cost = 1 / (1 + np.exp(np.minimum((dy - 3) * 5, 100)))
        # x_cost = 1 / (1 + np.exp(np.minimum((dx - 15) * 5, 100)))
        # single_cost = x_cost * y_cost
        # collision_index = np.any((dy < (dyn_obs_w[:, np.newaxis] + ego_w) / 2) & (dx < (dyn_obs_l[:, np.newaxis] + ego_l) / 2), axis=(1, 2))
        # factor = np.ones(t_num)
        # factor_cost = single_cost * factor[np.newaxis, np.newaxis, :]
        # cost4 = np.sum(factor_cost, axis=(1, 2)) * 1000 + 100 * collision_index

        # 总cost weight = np.array([[0.4, 35, 1, 100]])
        weight = np.array([[0, 35, 1, 100]])
        # weight = np.array([[self.a, self.b, 1, self.d]])
        cost = weight @ np.vstack((cost1, cost2, cost3, cost4))
        if np.all(cost == np.inf):
            best_index = global_path_index
        else:
            best_index = np.argmin(cost)

        if 100 + 50 < np.abs(path[0][0][0] - self.waypoints[-1][0]) < 120 + 50 and np.abs(path[0][1][0] - self.waypoints[-1][1]) < 1 and av > 25:
            best_index = len(path) // 2
        if 80 + 50 < np.abs(path[0][0][0] - self.waypoints[-1][0]) < 100 + 50 and np.abs(path[0][1][0] - self.waypoints[-1][1]) < 1 and av > 20:
            best_index = len(path) // 2
        if np.abs(path[0][0][0] - self.waypoints[-1][0]) < 80 + 50:
            best_index = len(path) // 2
        # if np.abs(path[0][0][0] - self.waypoints[-1][0]) < 50 + 50:
        #     best_index = len(path) // 2

        return best_index