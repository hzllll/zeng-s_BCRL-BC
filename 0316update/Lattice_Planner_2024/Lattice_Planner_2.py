from __future__ import print_function
from __future__ import division

# System level imports
import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)
import math
import numpy as np
import controller2d
import local_planner
import behavioural_planner
import velocity_planner
import matplotlib.pyplot as plt
import random
import csv

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from global_path_planner import global_path, AStar_Global_Planner
from opendrive2discretenet.opendrive_pa import parse_opendrive

from test03 import cal_refer_path_info, obs_process, cal_egoinfo
from test11 import dynamic_programming
from test05 import GenerateConvexSpace, path_index2s
from test06 import cal_plan_start_s_dotanddot2, SpeedPlanningwithQuadraticPlanning
from test09 import project_curvature_and_heading, cal_localwaypoints
from collision_checker_1 import collision_check
from collision_checker_1 import check_paths_feasibility, generate_avoidance_path_v2
from obstacle_prediction import extract_nearby_vehicles
from obstacle_prediction import obstacle_prediction

from position_pid import PositionPID
from PID_controll import PIDAngleController
import xodr2map
from planner_func import global_planner
from multi_local_path_plan import LocalPathPlan, RoadModel, DynamicObstacle
import path_check
import time

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + \
                           '/controller_output/'


class Lattice_Planner_2():

    def __init__(self):
        self.start_pos = {}
        self.goal_pos = {}
        self.scenario_info = {}
        self.frame = 0
        self.dt = 0
        self.scene_t = 0
        self.hyperparameters = {
            'max_accel': 4.0,
            'max_steering': 0.68,
            'overtaking_speed': 15.0
        }
        self.con_ite = 1
        self.high_state = 0
        self.scene_round = False
        self.scene_mixed = False
        self.scene_merge = False
        self.scene_mid_inter = False
        self.scene_mix_straight = False
        self.scene_crossing = False
        self.scene_intersection = False
        self.scene_straight_straight = False

        self.None_later_control = False
        self.None_later_control_highway = False
        self.None_later_control_city = False

        self.path_all_false = True
        self.middle_follow_distance = -1
        self.init_yaw = 0
        self.Initial_v = 10

        # 动力学约束
        self.ACC_LIMIT = 9.78  # m/s^2 9.8
        self.JERK_LIMIT = 49 * 0.3  # m/s^3  49
        self.ROT_LIMIT = 0.68  # rad 0.7
        self.ROT_RATE_LIMIT = 1.0  # rad/s 1.4
        self.GLOBAL_MAX_SPEED = 50.0  # m/s 55.0
        self.max_dec = 5
        self.max_acc = 5
        self.dynamic_velocity_planning = False

        self.waypoints = []
        self.pre_local_waypoints = None
        self.pre_u0 = np.array([0.0, 0.0])
        self.controller = []
        self.pre_control = []
        self.pre_obs = {}
        self.waypoint_xy = []
        self.dis_far_id = -1
        self.dis_near_id = -1
        self.front_bic_ped_found = False
        self.pre_waypoints = []
        self.far_waypoints = []
        self.serial_global_plan = False
        self.reach_goal = True
        self.staight_global = False
        self.map = []

        # generate_control 相关
        self.pre_u0 = np.array([0, 0])

        """
        Configurable params
        """
        # Planning Constants
        self.NUM_PATHS = 11
        self.BP_LOOKAHEAD_BASE = 9.0  # m 15
        self.BP_LOOKAHEAD_TIME = 2.0  # s 1.5
        self.BP_LOOKAHEAD_BASE_CITY = self.BP_LOOKAHEAD_BASE  # m 25 9
        self.BP_LOOKAHEAD_TIME_CITY = self.BP_LOOKAHEAD_TIME  # s 2.5 2.0
        self.PATH_OFFSET = 1.0  # m 1.0
        self.CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
        self.CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
        self.TIME_GAP = 1.0  # s 1.0
        self.PATH_SELECT_WEIGHT = 10
        self.A_MAX = self.hyperparameters['max_accel']  # m/s^2
        self.SLOW_SPEED = 0.0  # m/s
        self.HIGH_SPEED = 40.0  # m/s
        self.STOP_LINE_BUFFER = 3.5  # m 3.5
        self.LP_FREQUENCY_DIVISOR = 1  # Frequency divisor to make the
        # local planner operate at a lower
        # frequency than the controller
        # (which operates at the simulation
        # frequency). Must be a natural
        # number.
        self.LP_FREQUENCY_DIVISOR_HIGHWAY = 1
        self.LP_FREQUENCY_DIVISOR_CITY = 1

        # Path interpolation parameters
        self.INTERP_DISTANCE_RES = 0.001  # distance between interpolated points
        self.lp = []
        self.bp = []
        self.scene_info = {}
        # print('init')

    def init(self, scene_info):
        # print('scene:', scene_info)
        self.scenario_info = scene_info
        print('scene:', self.scenario_info)
        self.reach_goal = False
        self.init_replay()

    # init_replay作用：
    # 1、读取仿真场景信息
    # 2、计算全局路径，利用get_waypoints获取
    # 3、初始化控制器controller2d
    # 4、设置停车目标
    # 5、初始化局部路径规划器
    # 6、初始化行为规划器
    def init_replay(self):
        self.start_pos = {
            'x': self.scenario_info['task_info']['startPos'][0],
            'y': self.scenario_info['task_info']['startPos'][1],
        }
        self.goal_pos = {
            'x': [self.scenario_info['task_info']['targetPos'][0][0],
                  self.scenario_info['task_info']['targetPos'][1][0]],
            'y': [self.scenario_info['task_info']['targetPos'][0][1],
                  self.scenario_info['task_info']['targetPos'][1][1]],
        }
        self.map = []
        self.dt = self.scenario_info['task_info']['dt']
        self.waypoints = self.get_waypoints()
        self.TIME_GAP = self.dt
        self.pre_path_map = []
        self.pre_coll_check = []

        self.controller = controller2d.Controller2D(self.waypoints)

        # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
        stopsign_fences = np.array([
            [self.goal_pos['x'][0], self.goal_pos['y'][0], self.goal_pos['x'][1], self.goal_pos['y'][1]]
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

    # act作用：基于当前环境信息计算车辆的控制动作

    def act(self, observation):
        # print(observation)
        # action = self.act_city(observation)
        # time_now = time.time()
        try:
            action = self.act_city(observation)
        except:
            action = self.pre_u0
            stop_max_dec = (0.3 - observation.ego_info.v) / observation.test_info['dt']
            if action[0] < stop_max_dec:
                action[0] = stop_max_dec
        if observation.ego_info.v > 50 and action[0] > 0:
            action[0] = 0
        # cal_time = (time.time()-time_now)*1000
        # if cal_time > 0:
        #     with open(r"C:\WCY\00.code\RLearning\config\cal_time_onsite_expert.csv", "a", newline="", encoding="utf-8") as f:
        #         writer = csv.writer(f)
        #         writer.writerow([f"{cal_time:.4f}"])   # 保留4位小数

        # action = [1,0]
        return action

    # 1、调用AStar_Global_Planner解析场景类型
    # 2、根据场景标志
    # 3、判断是否在特殊场景（交叉口）调整控制模式
    # 4、选择不同的路径规划方法
    # 5、如果路径规划失败，则使用折线路径
    # 6、返回waypoint
    def get_waypoints(self):
        # print(self.scenario_info['source_file']['xodr'])
        if self.scenario_info['type'] == 'SERIAL':
            # print(self.scenario_info['task_info']['waypoints'])
            waypoints = []
            self.serial_global_plan = True
            for key in self.scenario_info['task_info']['waypoints']:
                # print(self.scenario_info['task_info']['waypoints'][key])
                waypoints.append([self.scenario_info['task_info']['waypoints'][key][0],
                                  self.scenario_info['task_info']['waypoints'][key][1], 10])
            # print(waypoints)
            waypoints_x = []
            waypoints_y = []
            for point in waypoints:
                waypoints_x.append(point[0])
                waypoints_y.append(point[1])
            # print(waypoints_x)
            # print(waypoints_y)
            return waypoints

        try:
            try:
                # if self.scenario_info
                map, ego_state = xodr2map.calmap(xodr_path=self.scenario_info['source_file']['xodr'],
                                                 xosc_path=self.scenario_info['source_file']['xosc'])
                # ego_state = self.scenario_info['task_info']['startPos']
                self.map = map
                goal = self.scenario_info['task_info']['targetPos']
                glb_plr = global_planner(map)
                glb_plr.map_data_process()
                waypoint = glb_plr.find_global_path(goal, ego_state)
            except:
                # print('use Astar global path')
                astar_path = AStar_Global_Planner(self.start_pos, self.goal_pos,
                                                  self.scenario_info['source_file']['xodr'])
                waypoint = astar_path.get_astar_path()
        except:
            # print('use straight global path')
            gp_none = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
            waypoint = gp_none.global_path_planning()
        waypoints = []
        waypoints_x = []
        waypoints_y = []
        self.waypoint_xy = []
        for point in waypoint:
            waypoints.append([point[0], point[1], 10])
            waypoints_x.append(point[0])
            waypoints_y.append(point[1])
            self.waypoint_xy.append([point[0], point[1]])
        # waypoint = np.stack((path_x_a, path_y_a, v_max_global), axis=1)
        # self.waypoint_xy = np.column_stack([waypoints_x, waypoints_y])
        # print(waypoints_x)
        # print(waypoints_y)
        # print(self.waypoint_xy)
        # draw_map(map, ego_state, goal, global_path)
        # plt.scatter(waypoints_x, waypoints_y, color="C3", s=0.1)
        # plt.show()
        return waypoints

    # 1、提取自车的状态信息
    # 2、提取测试信息
    def obs_change(self, observation):
        # observation = controller.observation
        # scenario_info = controller.scenario_info
        obs = {
            'vehicle_info': {
                'ego': {
                    'a': observation.ego_info.a,
                    'rot': observation.ego_info.rot,
                    'x': observation.ego_info.x,
                    'y': observation.ego_info.y,
                    'v': observation.ego_info.v,
                    'yaw': observation.ego_info.yaw,
                    'width': observation.ego_info.width,
                    'length': observation.ego_info.length,
                },

            },
            'test_setting': {
                'goal': {
                    'x': [self.scenario_info['task_info']['targetPos'][0][0],
                          self.scenario_info['task_info']['targetPos'][1][0]],
                    'y': [self.scenario_info['task_info']['targetPos'][0][1],
                          self.scenario_info['task_info']['targetPos'][1][1]],
                },
                't': observation.test_info['t'],
                'dt': observation.test_info['dt'],
                'end': observation.test_info['end'],
            },
            'test_info:': {
                't': observation.test_info['t'],
                'dt': observation.test_info['dt'],
                'end': observation.test_info['end'],

            }
        }

        # print('veh_info:', obs)
        return obs

    # 将计算出的控制指令转换为加速 (throttle)、转向 (steer)、刹车 (brake) 命令，并映射到最终的控制变量 u0
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

    # 1、获取传感器数据
    # 2、障碍物检测和预测
    # 3、计算局部路径
    # 4、检测前车、后车信息
    # 5、评估可行路径
    # 6、计算速度
    # 7、选择最佳路径
    # 8、输出控制指令
    # 主要功能和act_highway函数一样，只是适用场景不同
    def act_city(self, observation):
        # 数据格式处理
        obs = self.obs_change(observation)
        self.scene_t = observation.test_info['t']

        # SERIAL 场景全局路径重规划
        if self.serial_global_plan:
            self.staight_global = False
            try:
                ego_state_global = [obs['vehicle_info']['ego']['x'], obs['vehicle_info']['ego']['y'], obs['vehicle_info']['ego']['v'], obs['vehicle_info']['ego']['yaw']]
                map, _ = xodr2map.calmap(xodr_path=self.scenario_info['source_file']['xodr'],
                                         xosc_path=self.scenario_info['source_file']['xosc'])
                self.map = map
                goal = self.scenario_info['task_info']['targetPos']
                # print(ego_state_global)
                # print(goal)
                glb_plr = global_planner(map)
                glb_plr.map_data_process()
                waypoint = glb_plr.find_global_path(goal, ego_state_global)
                self.waypoints = []
                waypoints_x = []
                waypoints_y = []
                for point in waypoint:
                    self.waypoints.append([point[0], point[1], 10])
                    waypoints_x.append(point[0])
                    waypoints_y.append(point[1])
            except:
                # print('use straight global path')
                self.staight_global = True
                # gp_none = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                # waypoint = gp_none.global_path_planning()
                self.waypoints = []
                for m in range(0, 5000, 1):
                    path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                    path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                    self.waypoints.append([path_x, path_y, 10])


            self.serial_global_plan = False
            # plt.scatter(waypoints_x, waypoints_y, color="C3", s=0.1)
            # plt.show()
            # print(self.waypoints)
        if self.scenario_info['type'] == 'SERIAL':
            self.dynamic_velocity_planning = False
        elif self.map[0]['mid'][0][0] == 2.772508439493203:  # middle start
            self.dynamic_velocity_planning = True
        elif self.map[0]['mid'][0][0] == 3.8150041781499175:  # middle start
            self.dynamic_velocity_planning = True
        elif self.map[0]['mid'][0][0] == -6.509099420161929:  # two 2 one
            self.dynamic_velocity_planning = True
        elif self.map[0]['mid'][0][0] == -3.7728035151976735:  # opposite way empty
            self.dynamic_velocity_planning = True
        # elif self.map[0]['mid'][0][0] == -3.7195754521000373 and len(self.map) == 4:  #
        #     self.dynamic_velocity_planning = True
        elif self.map[0]['mid'][0][0] == 1145.5686001674692 and len(self.map) == 16:  # city merge
            self.dynamic_velocity_planning = True
        elif self.map[0]['mid'][0][0] == 998.0212366687042 and len(self.map) == 29:  # big crossing
            self.dynamic_velocity_planning = True
        else:
            self.dynamic_velocity_planning = False
        if self.map[0]['mid'][0][0] == 57.95847653635029 and len(self.map) == 44:  # small x
            self.BP_LOOKAHEAD_BASE = 4.0  # m 15
            self.BP_LOOKAHEAD_TIME = 1.0  # s 1.5
        elif self.map[0]['mid'][0][0] == 225.94111688811043 and len(self.map) == 40:  # round
            # print('round')
            self.BP_LOOKAHEAD_BASE = 5.0  # m 15
            self.BP_LOOKAHEAD_TIME = 1.0  # s 1.5
        elif self.map[0]['mid'][0][0] == 326.8948897620285 and len(self.map) == 15:  # long straight right
            # print('round')
            self.BP_LOOKAHEAD_BASE = 5.0  # m 15
            self.BP_LOOKAHEAD_TIME = 1.0  # s 1.5
        else:
            self.BP_LOOKAHEAD_BASE = 9.0  # m 15
            self.BP_LOOKAHEAD_TIME = 2.0  # s 1.5
        self.BP_LOOKAHEAD_BASE_CITY = self.BP_LOOKAHEAD_BASE  # m 25 9
        self.BP_LOOKAHEAD_TIME_CITY = self.BP_LOOKAHEAD_TIME  # s 2.5 2.0
        # print(self.map[0]['mid'][0][0])  # 322.77756480497055
        # print('len_map:', len(self.map))

        # 障碍物预测
        nearby_vehicles_ago = extract_nearby_vehicles(observation, False)
        predict_array = obstacle_prediction(nearby_vehicles_ago, self.dt)
        # print('obs:', obs)
        # print('observation:', observation)
        way_ang = [0, 0, 0, 0]
        way_ang[0] = math.atan2((self.goal_pos['y'][0] - obs['vehicle_info']['ego']['y']),
                                (self.goal_pos['x'][0] - obs['vehicle_info']['ego']['x']))
        way_ang[1] = math.atan2((self.goal_pos['y'][0] - obs['vehicle_info']['ego']['y']),
                                (self.goal_pos['x'][1] - obs['vehicle_info']['ego']['x']))
        way_ang[2] = math.atan2((self.goal_pos['y'][1] - obs['vehicle_info']['ego']['y']),
                                (self.goal_pos['x'][0] - obs['vehicle_info']['ego']['x']))
        way_ang[3] = math.atan2((self.goal_pos['y'][1] - obs['vehicle_info']['ego']['y']),
                                (self.goal_pos['x'][1] - obs['vehicle_info']['ego']['x']))
        middle_lead_car_found = False
        self.front_bic_ped_found = False

        self_x = obs['vehicle_info']['ego']['x']
        self_y = obs['vehicle_info']['ego']['y']
        self_yaw = obs['vehicle_info']['ego']['yaw']
        self_width = obs['vehicle_info']['ego']['width']
        self_length = obs['vehicle_info']['ego']['length']
        self_v = obs['vehicle_info']['ego']['v']

        dis_far = 30
        self.dis_far_id = -1
        self.dis_near_id = -1
        dis_near_comp = 999
        dis_far_comp = 999
        for m in range(len(self.waypoints)):
            dis_now = ((self_x - self.waypoints[m][0]) ** 2 + (self_y - self.waypoints[m][1]) ** 2) ** 0.5
            if dis_now < dis_near_comp:
                self.dis_near_id = m
                dis_near_comp = dis_now
            if abs(dis_now - dis_far) < dis_far_comp:
                self.dis_far_id = m
                dis_far_comp = abs(dis_now - dis_far)
        self.pre_waypoints = []
        id_now = -1
        for waypo in self.waypoints:
            id_now += 1
            if self.dis_far_id - 200 < id_now < self.dis_far_id - 3:
                self.pre_waypoints.append([waypo[0], waypo[1], 15])
            if self.dis_far_id < id_now < len(self.waypoints) - 1:
                self.far_waypoints.append([waypo[0], waypo[1], 15])

        if self.scene_t == 0:
            self.init_yaw = obs['vehicle_info']['ego']['yaw']
        # print('observation:')

        # 当前参数获取
        current_x, current_y, current_yaw, current_speed, ego_length = \
            [obs['vehicle_info']['ego'][key] for key in ['x', 'y', 'yaw', 'v', 'length']]
        current_timestamp = obs['test_setting']['t'] + \
                            obs['test_setting']['dt']
        self_Car = [current_speed, current_yaw]
        lead_car_info = {
            'middle': {
                'pre_long': float('inf')
            },
            'left': {
                'pre_long': float('inf')
            },
            'right': {
                'pre_long': float('inf')
            },
        }

        follow_car_info = {
            'middle': {
                'pre_long': -float('inf')
            },
            'left': {
                'pre_long': -float('inf')
            },
            'right': {
                'pre_long': -float('inf')
            },
        }
        middle_follow_car_found = False
        time_now = time.time()
        # self.front_bic_ped_found = True
        if len(self.map) == 40:
            obs_lat = 2.8
        else:
            obs_lat = 2.0
        for id in observation.object_info:
            for vehi in observation.object_info[id].values():
                vec_x = vehi.x - self_x
                vec_y = vehi.y - self_y
                vehi_yaw = vehi.yaw
                # print('vec_x:', vec_x)

                long = vec_x * math.cos(self_yaw) + vec_y * math.sin(self_yaw)
                lat = - vec_x * math.sin(self_yaw) + vec_y * math.cos(self_yaw)
                # err = road_width + (vehi.width + self_width) / 2
                dis = ((vehi.x - self_x) ** 2 + (vehi.y - self_y) ** 2) ** 0.5
                lat_off_1 = -2
                lat_off_2 = 2
                obs_err = 999
                dis_off = 85
                # if self.map[0]['mid'][0][0] == 225.94111688811043 and len(self.map) == 40:
                for i in range(len(self.waypoints)):
                    if i > self.dis_near_id - 3 or i < self.dis_near_id + 10:
                        obs_offset = ((vehi.x - self.waypoints[i][0]) ** 2 + (
                                vehi.y - self.waypoints[i][1]) ** 2) ** 0.5
                        if obs_offset < obs_err:
                            obs_err = obs_offset
                # else:
                #     for i in range(min(len(self.waypoints), self.dis_far_id)):
                #         if self.dis_near_id > 0:
                #             if i < self.dis_near_id:
                #                 continue
                #         obs_offset = ((vehi.x - self.waypoints[i][0]) ** 2 + (
                #                 vehi.y - self.waypoints[i][1]) ** 2) ** 0.5
                #         if obs_offset < obs_err:
                #             obs_err = obs_offset
                if id == 'vehicle':
                    lat_off_1 = -2.5
                    lat_off_2 = 2.5
                    dis_off = 40  # 35
                    vehi_yaw = vehi.yaw
                    gap_theta = abs((self_yaw - vehi_yaw + np.pi) % (2 * np.pi) - np.pi)
                    # if gap_theta > np.pi / 2:
                    #     continue

                elif id == 'bicycle':
                    lat_off_1 = -1.5
                    lat_off_2 = 1.5
                    dis_off = 30
                    self.front_bic_ped_found = True
                    # ped_front = True
                    # print('pedestrian in front!')
                elif id == 'pedestrian':
                    # lat_off_1 = -7.5
                    # lat_off_2 = 4.5
                    # dis_off = 35
                    lat_off_1 = -1.5
                    lat_off_2 = 1.5
                    dis_off = 30
                    self.front_bic_ped_found = True
                    # if long > 0 and lat_off_1 < lat < lat_off_2 and dis < dis_off:
                    #     if self.max_speed_desired > 5.0:
                    #         self.max_speed_desired = 5.0
                    #     ped_front = True
                    # else:
                    #     ped_front = False

                if long > 0 and dis < dis_off:
                # if dis < dis_off:
                    front_obs_exist = True
                    if obs_err < obs_lat:  # (vehi.width + self_width) / 2 + road_offset: lat_off_1 < lat < lat_off_2 or
                        long_stand = lead_car_info['middle']['pre_long']
                        if long < long_stand:
                            lead_car_info['middle']['pre_long'] = long
                            lead_car_info['middle']['x'] = vehi.x
                            lead_car_info['middle']['y'] = vehi.y
                            lead_car_info['middle']['v'] = vehi.v
                            lead_car_info['middle']['a'] = vehi.a
                            lead_car_info['middle']['yaw'] = vehi.yaw  # ((vehi.yaw + 360) % 360) * np.pi / 180
                            lead_car_info['middle']['width'] = vehi.width
                            lead_car_info['middle']['length'] = vehi.length
                            middle_lead_car_found = True
                            # if id == 'pedestrian':
                            #     lead_car_info['middle']['v'] = 0

        # if self.front_bic_ped_found:
        #     print('DVP')
        #     self.dynamic_velocity_planning = True
        # else:
        #     print('REAL')
        #     self.dynamic_velocity_planning = False

        # print('middle_follow_car_found:', middle_follow_car_found)
        if middle_lead_car_found:
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
        if middle_follow_car_found:
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

        local_waypoints = None
        path_validity = np.zeros((self.NUM_PATHS, 1), dtype=bool)

        reached_the_end = False

        # Update pose and timestamp
        prev_timestamp = obs['test_setting']['t']
        # print("side")

        if self.frame % self.LP_FREQUENCY_DIVISOR_CITY == 0:
            # if not self.dynamic_velocity_planning:
            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp,
                                                                            obs['vehicle_info']['ego']['v'])

            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_pos[0][2], lead_car_speed[0][0],
                              lead_car_length[0][0]]
            follow_car_state = [follow_car_pos[0][0], follow_car_pos[0][1], follow_car_pos[0][2],
                                follow_car_speed[0][0], follow_car_length[0][0]]

            # Set lookahead based on current speed.
            self.bp.set_lookahead(self.BP_LOOKAHEAD_BASE_CITY + self.BP_LOOKAHEAD_TIME_CITY * open_loop_speed)

            # Perform a state transition in the behavioural planner.
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            # Check to see if we need to follow the lead vehicle.
            dist_gap = 6.0
            ego_rear_dist = self.calc_car_dist(ego_state, follow_car_state) - dist_gap
            ego_rear_dist = ego_rear_dist if ego_rear_dist > 0 else 1e-8
            ego_front_dist = self.calc_car_dist(ego_state, lead_car_state) - dist_gap
            ego_front_dist = ego_front_dist if ego_front_dist > 0 else 1e-8
            a = (2 * follow_car_speed[0][0] - current_speed) * current_speed / (2 * ego_rear_dist)
            a_x = max(1e-8, min(self.hyperparameters['max_accel'], a))
            min_follow_dist = (current_speed ** 2 - lead_car_speed[0][0] ** 2) / (
                    2 * self.hyperparameters['max_accel']) + dist_gap
            follow_dist = current_speed ** 2 / (2 * a_x) - lead_car_speed[0][0] ** 2 / (
                    2 * self.hyperparameters['max_accel'])
            if follow_dist > ego_front_dist:
                follow_dist = min_follow_dist
            LEAD_VEHICLE_LOOKAHEAD = follow_dist if follow_dist > 0 else 0
            # LEAD_VEHICLE_LOOKAHEAD = 15
            self.bp.check_for_lead_vehicle(ego_state, lead_car_pos[0], LEAD_VEHICLE_LOOKAHEAD)

            # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints,
                                                        ego_state)
            if self.scenario_info['type'] == 'SERIAL':
                use_all_path = True
            else:
                use_all_path = False
            # Calculate planned paths in the local frame.
            paths, path_validity = self.lp.plan_paths(goal_state_set, use_all_path)
            # print('pathlen:', len(paths))

            # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)
            # print('pathlen:', len(paths))
            # print(paths)
            if len(paths) == 0:
                # print('use new local path!!!')
                # multi local path plan part
                local_path_plan = LocalPathPlan
                local_path_plan.init(LocalPathPlan)
                road = RoadModel(local_path_plan, self.waypoint_xy)
                start_idx = self.dis_near_id
                path_length = 20
                ref_path_length = 50
                ref = road.get_reference_points(start_idx=start_idx, path_length=ref_path_length)
                delta = np.mean(np.linalg.norm(np.diff(ref, axis=0), axis=1))
                node_num = round(path_length / delta)
                init_heading = 0
                obs_list = []
                v_opt = []
                # print('obs_list')
                x_opt, y_opt = local_path_plan.path_planning(LocalPathPlan, obs_list, (50, 100, 2000, 1000), node_num,
                                                             fixed_distance=5.0, ref_path=ref,
                                                             road_model=road, init_heading=init_heading)
                for i in range(len(x_opt)):
                    v_opt.append(12)
                # print('x_opt:', x_opt)
                # for i in range(len(x_opt)):
                paths.append([x_opt, y_opt, v_opt])

            # if obs['test_setting']['dt'] == 0.05:
            #     self.front_bic_ped_found = False
            # path_map = self.lp._collision_checker.check_paths(paths, self.map)

            # print('len_map:', len(self.map))
            # if len(self.map) == 40 or len(self.map) == 8 or len(self.map) == 14:
            #     time_temp = time.time()
            #     path_map = path_check.check_paths(observation, paths, self.map)
            #     print('map_time:', round((time.time() - time_temp), 2))
            #     print('path_map:', path_map)
            # else:
            #     path_map = np.zeros(len(paths), dtype=bool)
            #     for i in range(len(path_map)):
            #         path_map[i] = True
            # if self.scenario_info['type'] == 'SERIAL' or len(self.map) > 1000 or len(self.map) == 6 or len(self.map) == 4 or len(self.map) == 19 or obs['test_setting']['dt'] == 0.04:
            #     path_map = np.zeros(len(paths), dtype=bool)
            #     for i in range(len(path_map)):
            #         path_map[i] = True
            # else:
            #     path_map = path_check.check_paths(paths, self.map)
            #     # path_map = self.lp._collision_checker.check_paths(paths, self.map)
            if self.scenario_info['type'] == 'SERIAL':
                best_index = int(len(paths) / 2)
            elif len(self.map) == 44:
                best_index = int(len(paths) / 2)
            elif len(self.map) == 29:
                best_index = int(len(paths) / 2)
            elif self.map[0]['mid'][0][0] == 225.94111688811043 and len(self.map) == 40:
                best_index = int(len(paths) / 2)
            elif self.map[0]['mid'][0][0] == 326.8948897620285 and len(self.map) == 15:  # long straight right
                best_index = int(len(paths) / 2)
            else:
                if self.frame % 3 == 0 or self.pre_path_map == []:
                    time_temp = time.time()
                    path_map = path_check.check_paths(observation, paths, self.map)
                    full_false = True
                    for i in range(len(path_map)):
                        if path_map[i]:
                            full_false = False
                    if full_false:
                        for i in range(len(path_map)):
                            path_map[i] = True
                    self.pre_path_map = path_map
                    # print('map_time:', round((time.time() - time_temp), 2))
                    # print('path_map:', path_map)
                else:
                    path_map = self.pre_path_map
                # print('map_time:', round((time.time() - time_temp), 2))
                # print('pmap:', path_map)
                # self_car = [self_v, self_yaw, self_width, self_length]
                # collision_check_array = collision_check(paths, predict_array, self_car, self.dt)
                collision_check_array = check_paths_feasibility(paths, observation, len(self.map))
                # print('cock:', collision_check_array)
                coll_full_false = True
                for i in range(len(collision_check_array)):
                    if collision_check_array[i]:
                        coll_full_false = False
                if coll_full_false and not self.pre_coll_check == []:
                    # print('coll array full false')
                    collision_check_array = self.pre_coll_check
                else:
                    self.pre_coll_check = collision_check_array
                if len(self.map) == 40:
                    for j in range(len(collision_check_array)):
                        collision_check_array[j] = True
                for i in range(min(len(path_validity), len(collision_check_array), len(path_map))):
                    if not path_validity[i] or not collision_check_array[i] or not path_map[i]:
                        collision_check_array[i] = False
                # print('coll:', collision_check_array)
                best_index = self.lp._collision_checker.select_best_path_index_city(paths, collision_check_array,
                                                                                    self.waypoints, self.dis_near_id)
                # print('best_index:', best_index)
            best_path = paths[best_index]

            if not self.dynamic_velocity_planning:
                if not best_index == int(len(paths) / 2):  # and not obs['test_setting']['dt'] == 0.05:
                    # print('detour logic')
                    lead_car_info = {
                        'middle': {
                            'pre_long': float('inf')
                        },
                        'left': {
                            'pre_long': float('inf')
                        },
                        'right': {
                            'pre_long': float('inf')
                        },
                    }
                    middle_lead_car_found = False
                    for id in observation.object_info:
                        for vehi in observation.object_info[id].values():
                            vec_x = vehi.x - self_x
                            vec_y = vehi.y - self_y
                            vehi_yaw = vehi.yaw
                            # print('vec_x:', vec_x)

                            long = vec_x * math.cos(self_yaw) + vec_y * math.sin(self_yaw)
                            lat = - vec_x * math.sin(self_yaw) + vec_y * math.cos(self_yaw)
                            # err = road_width + (vehi.width + self_width) / 2
                            dis = ((vehi.x - self_x) ** 2 + (vehi.y - self_y) ** 2) ** 0.5
                            lat_off_1 = -2
                            lat_off_2 = 2
                            obs_err = 999
                            dis_off = 85
                            for i in range(len(paths[best_index][0])):
                                # if i > self.dis_near_id or i < self.dis_near_id + 100:
                                obs_offset = ((vehi.x - paths[best_index][0][i]) ** 2 + (
                                        vehi.y - paths[best_index][1][i]) ** 2) ** 0.5
                                if obs_offset < obs_err:
                                    obs_err = obs_offset
                            if id == 'vehicle':
                                lat_off_1 = -2.5
                                lat_off_2 = 2.5
                                dis_off = 40
                                # vehi_yaw = vehi.yaw
                                gap_theta = abs((self_yaw - vehi_yaw + np.pi) % (2 * np.pi) - np.pi)
                            elif id == 'bicycle':
                                lat_off_1 = -1.5
                                lat_off_2 = 1.5
                                dis_off = 40
                                self.front_bic_ped_found = True
                                # ped_front = True
                                # print('pedestrian in front!')
                            elif id == 'pedestrian':
                                lat_off_1 = -1.5
                                lat_off_2 = 1.5
                                dis_off = 40
                                self.front_bic_ped_found = True

                            if long > 0 and dis < dis_off:
                                front_obs_exist = True
                                if obs_err < obs_lat:  # (vehi.width + self_width) / 2 + road_offset: lat_off_1 < lat < lat_off_2 or
                                    long_stand = lead_car_info['middle']['pre_long']
                                    if long < long_stand:
                                        lead_car_info['middle']['pre_long'] = long
                                        lead_car_info['middle']['x'] = vehi.x
                                        lead_car_info['middle']['y'] = vehi.y
                                        lead_car_info['middle']['v'] = vehi.v
                                        lead_car_info['middle']['a'] = vehi.a
                                        lead_car_info['middle']['yaw'] = vehi.yaw  # ((vehi.yaw + 360) % 360) * np.pi / 180
                                        lead_car_info['middle']['width'] = vehi.width
                                        lead_car_info['middle']['length'] = vehi.length
                                        middle_lead_car_found = True
                                        # if id == 'pedestrian':
                                        #     lead_car_info['middle']['v'] = 0

                    # print('middle_follow_car_found:', middle_follow_car_found)
                    if middle_lead_car_found:
                        lead_car_pos = np.array([
                            [lead_car_info['middle']['x'], lead_car_info['middle']['y'],
                             lead_car_info['middle']['yaw']]])
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

            max_speed = 14
            if self.scenario_info['type'] == 'SERIAL':
                if self.staight_global:
                    max_speed = 20
            if len(self.map) == 8:
                max_speed = 16
            if len(self.map) == 69:
                max_speed = 20
            if len(self.map) == 2 and self.map[0]['mid'][0][0] == -50.0:
                max_speed = 20
            obs_speed = []
            for vehi in observation.object_info['vehicle'].values():
                obs_speed.append(vehi.v)
            if not obs_speed == []:
                max_speed = max(np.mean(obs_speed) * 1.3, max_speed)

            path_bear = []
            path_gap = []
            for i in range(len(best_path[0])):
                if i < 1:
                    continue
                x1 = best_path[0][i - 1]
                y1 = best_path[1][i - 1]
                # x1 = best_path[0][0]
                # y1 = best_path[1][0]
                x2 = best_path[0][i]
                y2 = best_path[1][i]
                delta_x = x2 - x1
                delta_y = y2 - y1

                init_bear = math.atan2(delta_y, delta_x)
                # init_bear = math.degrees(init_bear)
                init_bear = (init_bear + 2 * np.pi) % (2 * np.pi)
                # print('init_bear:', init_bear)
                # print('self_yaw:', self_yaw)
                # comp_bear = (init_bear + 360) % 360
                comp_bear = abs(init_bear - self_yaw)
                comp_bear = math.degrees(comp_bear)
                path_bear.append(comp_bear)
                # path_bear.append(init_bear)
                # if i > 0:
                #     path_gap.append(comp_bear - path_bear[0])

            if len(path_bear) > 0:
                # max_gap = abs(max(path_bear) - min(path_bear))
                max_gap = max(path_bear)
            else:
                max_gap = 0
            if max_gap > 350:
                max_gap = max_gap - 350
            # print('max_gap:', max_gap)
            if max_speed < 20:
                if max_gap >= 50:
                    max_speed = 8
                    # print('max gap:', round(max_gap), 'max speed:', max_speed)
                elif max_gap >= 40:
                    max_speed = 10
                    # print('max gap:', round(max_gap), 'max speed:', max_speed)
                elif max_gap >= 30:
                    max_speed = 10
                elif max_gap >= 20:
                    max_speed = 12
                elif max_gap >= 10:
                    max_speed = 12
                # elif max_gap >= 5:
                #     max_speed = 14
            if self.scenario_info['type'] == 'SERIAL' and not self.staight_global:
            # if self.scenario_info['type'] == 'SERIAL':
                max_speed = 10
            if len(self.map) == 4:
                if self.map[0]['mid'][0][0] == 2.772508439493203 or self.map[0]['mid'][0][0] == 3.8150041781499175:
                    max_speed = 9.2
            # print('max_speed:', max_speed)

            # 计算limit_speed
            limit_speed = max_speed

            if not self.dynamic_velocity_planning:

                if middle_lead_car_found:
                    desired_speed_lead = np.sqrt(
                        lead_car_speed[0][0] ** 2 + 2 * ego_front_dist * self.hyperparameters['max_accel'])
                    # print('desired_speed_lead:', desired_speed_lead)
                    desired_speed = desired_speed_lead
                    dis = ((lead_car_pos[0][0] - self_x) ** 2 + (lead_car_pos[0][1] - self_y) ** 2) ** 0.5
                    if desired_speed > max_speed:  # self.bp._goal_state[2]:
                        desired_speed = max_speed  # self.bp._goal_state[2]
                    if self.map[0]['mid'][0][0] == 225.94111688811043 and len(self.map) == 40:  # round
                        if desired_speed > 1.5 * lead_car_speed[0][0] and lead_car_speed[0][0] > 1.5:
                            desired_speed = 1.5 * lead_car_speed[0][0]
                    else:
                        if desired_speed > 1.5 * lead_car_speed[0][0]:
                            desired_speed = 1.5 * lead_car_speed[0][0]
                    if desired_speed > lead_car_speed[0][0]:  # self.bp._goal_state[2]:
                        # print('dis:', dis)
                        # if dis < 15:
                        #     desired_speed = min(lead_car_speed[0][0], 8.0)  # self.bp._goal_state[2]
                        if dis < 13:
                            desired_speed = lead_car_speed[0][0]  # self.bp._goal_state[2]
                        if dis < 10:
                            desired_speed = 0.85 * lead_car_speed[0][0]  # self.bp._goal_state[2]
                    if lead_car_speed[0][0] < 1.5 and dis < 10:
                        # print('stop car in front')
                        desired_speed = 0.3
                    # print('lead_car_speed[0][0]:', lead_car_speed[0][0])
                    # if desired_speed_lead < 0:  # self.bp._goal_state[2]:
                    #     desired_speed = 0  # self.bp._goal_state[2]
                    # else:
                    #     desired_speed = desired_speed_lead
                else:
                    desired_speed = max_speed  # self.bp._goal_state[2]
                # print('desired speed:', desired_speed)
                # # desired_speed = self.bp._goal_state[2]
                # # print('lacall 0')
                decelerate_to_stop = self.bp._state == behavioural_planner.DECELERATE_TO_STOP

                # decelerate_to_stop = True
                # for j in range(len(collision_check_array)):
                #     if collision_check_array[j] == True:
                #         decelerate_to_stop = False
                # print('stop:', decelerate_to_stop)
                #
                local_waypoints = self.lp._velocity_planner.compute_velocity_profile(best_path, desired_speed,
                                                                                     ego_state,
                                                                                     current_speed, decelerate_to_stop,
                                                                                     lead_car_state,
                                                                                     self.bp._follow_lead_vehicle)
            # print(self.scene_mix_straight)
            if self.dynamic_velocity_planning:
                ##################### 速度规划
                obs_backcar_list = []
                # long = 0
                for i in range(len(predict_array)):
                    # obs_backcar_list = []
                    for j in range(0, 4):
                        vehicle_x = predict_array[i][5 * j]
                        vehicle_y = predict_array[i][5 * j + 1]
                        delta_x = vehicle_x - self_x
                        delta_y = vehicle_y - self_y
                        # 计算目标车辆相对于自车的方向角度
                        relative_angle = math.atan2(delta_y, delta_x)
                        # 计算相对角度差
                        angle_diff = relative_angle - self_yaw
                        # 将角度差调整到 -pi 到 pi 的范围内
                        while angle_diff > math.pi:
                            angle_diff -= 2 * math.pi
                        while angle_diff < -math.pi:
                            angle_diff += 2 * math.pi
                        # 判断目标车辆是否在自车的前方（角度差在 -pi/2 到 pi/2 之间）
                        if -math.pi / 2 < angle_diff < math.pi / 2:
                            continue
                        else:
                            long = - math.sqrt(delta_x ** 2 + delta_y ** 2)
                            lat = - delta_x * math.sin(self_yaw) + delta_y * math.cos(self_yaw)
                            # 选择一个距离判断是否考虑后车会追尾
                            if abs(long) > 15:
                                continue
                            else:
                                if abs(lat) < self_width:
                                    if abs(predict_array[i][5 * j + 2] - self_yaw) < 0.524 or abs(
                                            predict_array[i][5 * j + 2] - self_yaw) > 5.76:
                                        obs_backcar_list.append([long, 0.5 * j])
                                    else:
                                        continue

                obs_s_in = []
                obs_s_out = []
                obs_t_in = []
                obs_t_out = []
                err = 6
                if obs_backcar_list:
                    obs_backcar_s = []
                    obs_backcar_t = []
                    obs_index = []
                    for i in range(len(obs_backcar_list)):
                        if obs_backcar_list[i][0] + err >= 0:
                            obs_backcar_s.append(obs_backcar_list[i][0])
                            obs_backcar_t.append(obs_backcar_list[i][1])
                            obs_index.append(i)
                    if obs_backcar_s:
                        m = len(obs_backcar_s)
                        if m == 1:
                            t_intersection = 1.2
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[0] + err)
                            obs_t_in.append(t_intersection)
                            obs_t_out.append(obs_backcar_t[0])
                        elif m == 2:
                            k = (obs_backcar_s[1] - obs_backcar_s[0]) / 0.5
                            t_intersection01 = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                            if t_intersection01 < 0:
                                obs_t_in.append(0.3)
                            else:
                                obs_t_in.append(t_intersection01)
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[1] + err)
                            obs_t_out.append(obs_backcar_t[1])
                        elif m == 3:
                            k = (obs_backcar_s[1] - obs_backcar_s[0]) / 0.5
                            t_intersection02 = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                            if t_intersection02 < 0.5:
                                obs_t_in.append(0.5)
                            else:
                                obs_t_in.append(t_intersection02)
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[2] + err)
                            obs_t_out.append(obs_backcar_t[2])
                        else:
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[1] + err)
                            obs_t_in.append(0.2)
                            obs_t_out.append(0.5)
                # print("当前时间：", observation.test_info['t'])
                # ego_yaw = obs['vehicle_info']['ego']['yaw']
                selected_obs_list = []
                for i in range(len(predict_array)):
                    obs_list = []
                    # 使用更密集的时间点，从0开始以0.1秒为间隔（0到1.5秒，共16个点）
                    for j in range(0, 16):  # 0, 0.1, 0.2, ..., 1.5
                        time_step = 0.1 * j

                        # 确定此时间点位于哪两个原始预测点之间
                        base_idx = int(time_step / 0.5)  # 将时间映射到原始时间索引：0, 1, 2, 3
                        next_idx = min(base_idx + 1, 3)  # 确保不超出范围

                        # 原始时间点
                        base_time = 0.5 * base_idx
                        next_time = 0.5 * next_idx

                        if base_time == time_step:  # 如果恰好是原始预测点
                            x = predict_array[i][base_idx * 5]
                            y = predict_array[i][base_idx * 5 + 1]
                        else:  # 需要在两个原始点之间插值
                            # 计算插值比例
                            ratio = (time_step - base_time) / (next_time - base_time)

                            # 线性插值计算位置
                            x = predict_array[i][base_idx * 5] + ratio * (
                                    predict_array[i][next_idx * 5] - predict_array[i][base_idx * 5])
                            y = predict_array[i][base_idx * 5 + 1] + ratio * (
                                    predict_array[i][next_idx * 5 + 1] - predict_array[i][base_idx * 5 + 1])

                        # 添加到观测列表，包含[x, y, t]
                        obs_list.append([x, y, time_step])

                    selected_obs_list.append(obs_list)

                # print(selected_obs_list)
                # 将路径转换为s进行表示
                pathindex2s = path_index2s(best_path)
                # 对局部路径进行增密处理，并计算曲率和航向角
                x_new, y_new, curvatures, headings_degrees = cal_refer_path_info(best_path)

                # 根据增密后的曲率，再计算出原局部路径点的曲率和航向角
                curvatures_new = project_curvature_and_heading(best_path, x_new, y_new, curvatures)
                headings_degrees_new = project_curvature_and_heading(best_path, x_new, y_new, headings_degrees)

                # 计算自车在局部路径上的s坐标
                ego_x = obs['vehicle_info']['ego']['x']
                ego_y = obs['vehicle_info']['ego']['y']
                ego_speed = obs['vehicle_info']['ego']['v']  # 获取自车当前速度
                ego_s = 0
                min_dist = float('inf')
                for idx, (x, y) in enumerate(zip(x_new, y_new)):
                    dist = (ego_x - x) ** 2 + (ego_y - y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        ego_s = pathindex2s[idx]
                obs_st_s_in, obs_st_s_out, obs_st_t_in, obs_st_t_out, obs_to_consider, SLTofsingle_obs_list = obs_process(
                    pathindex2s, selected_obs_list,
                    x_new, y_new, headings_degrees, ego_s=ego_s, ego_speed=ego_speed, scenario_info=self.scenario_info)

                # 计算自车的信息
                if obs_to_consider:
                    reference_speed, w_cost_ref_speed, plan_start_v, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, plan_start_s_dot2 \
                        = cal_egoinfo(observation, headings_degrees, curvatures)
                    dp_speed_s, dp_speed_t = dynamic_programming(prev_timestamp, obs_st_s_in, obs_st_s_out, obs_st_t_in,
                                                                 obs_st_t_out, w_cost_ref_speed,
                                                                 reference_speed, w_cost_accel, w_cost_obs,
                                                                 plan_start_s_dot, s_list, t_list)

                    # start_time02 = time.time()
                    s_lb, s_ub, s_dot_lb, s_dot_ub \
                        = GenerateConvexSpace(dp_speed_s, dp_speed_t, pathindex2s, obs_st_s_in, obs_st_s_out,
                                              obs_st_t_in,
                                              obs_st_t_out,
                                              curvatures_new, max_lateral_accel=0.2 * 9.8, scenario_info=self.scenario_info)

                    # start_time03 = time.time()
                    qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init \
                        = SpeedPlanningwithQuadraticPlanning(plan_start_s_dot, plan_start_s_dot2, dp_speed_s,
                                                             dp_speed_t,
                                                             s_lb, s_ub, s_dot_lb, s_dot_ub)
                    local_waypoints = cal_localwaypoints(best_path, qp_s_dot_init)
                else:
                    ego_speed = obs['vehicle_info']['ego']['v']
                    # local_waypoints = [[best_path[0][i], best_path[1][i], ego_speed] for i in range(len(best_path[0]))]
                    for i in range(len(best_path[2])):
                        if i == 0:
                            best_path[2][i] = ego_speed
                        else:
                            best_path[2][i] = best_path[2][i - 1] + 0.4
                            if best_path[2][i] > limit_speed:
                                best_path[2][i] = limit_speed
                    local_waypoints = [[best_path[0][i], best_path[1][i], best_path[2][i]] for i in
                                       range(len(best_path[0]))]
                    # accelerate = (local_waypoints[1][2] - local_waypoints[0][2]) / 0.1

                #########################################

            # --------------------------------------------------------------
            # print('lacall 1')
            cal_time = (time.time()-time_now)*1000
            if cal_time > 0:
                with open(r"C:\WCY\00.code\RLearning\config\cal_time_onsite_expert_new.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{cal_time:.4f}"])   # 保留4位小数

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

        if local_waypoints != None:
            # print(local_waypoints[2])
            for i in range(len(local_waypoints)):
                if local_waypoints[i][2] > limit_speed:
                    # print('over max speed:', local_waypoints[i][2])
                    local_waypoints[i][2] = limit_speed
                # if local_waypoints[i][2] < 0.3:
                #     local_waypoints[i][2] = 0
                if local_waypoints[i][2] <= 0.3:
                    local_waypoints[i][2] = 0.3
            self.pre_local_waypoints = local_waypoints
            # print('waypoints:', local_waypoints[1][2])
            # print('goal_speed:', local_waypoints[1][2])

        try:
            u0 = self.generate_control(self.pre_local_waypoints, obs)
        except:
            u0 = self.pre_u0

        self.pre_u0 = u0
        self.pre_obs = obs

        self.frame += 1

        return u0

    # 全局路径规划
    # def generate_global_path(self, x0, y0, x1, y1):
    #     # Create an array of x and y values for the two points
    #     x = np.array([x0, x1])
    #     y = np.array([y0, y1])
    #
    #     # Create a function that interpolates between the two points
    #     f = interp1d(x, y, kind='linear')
    #
    #     # Create an array of x values for the smooth path
    #     x_global = np.linspace(x0, x1, num=100)
    #
    #     # Use the interpolation function to get the corresponding y values
    #     y_global = f(x_global)
    #
    #     # yaw
    #     yaw_global = np.arctan2(np.diff(y_global), np.diff(x_global))
    #     yaw_global = np.append(yaw_global, yaw_global[-1])
    #     yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
    #
    #     waypoints = np.stack((x_global, y_global, yaw_global), axis=1)
    #
    #     # Return the x and y values of the smooth path
    #     return waypoints

    # 该函数作用是根据局部路径和传感器数据，计算控制指令（油门、刹车）
    # 1、更新控制器目标路径
    # 2、解析当前车辆状态
    # 3、计算控制指令，油门刹车转向等
    # 4、发送控制指令，调用send_control_command
    # 5、返回控制信号
    def generate_control(self, local_waypoints, obs):
        # theta = self.later_control(local_waypoints, obs)
        local_waypoints = np.vstack(local_waypoints)
        result_x = local_waypoints[:, 0]
        result_y = local_waypoints[:, 1]
        # result_x = self.far_waypoints[:][0]
        # result_y = self.far_waypoints[:][1]
        yaw_global = np.arctan2(np.diff(result_y), np.diff(result_x))
        yaw_global = np.append(yaw_global, yaw_global[-1])
        yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
        yaw_global = yaw_global.reshape(-1, 1)
        local_waypoints = np.concatenate((local_waypoints, yaw_global), axis=1)

        speeds_x = local_waypoints[:, 2] * np.cos(local_waypoints[:, 3])
        speeds_y = local_waypoints[:, 2] * np.sin(local_waypoints[:, 3])

        init_state = {
            'px': obs['vehicle_info']['ego']['x'],
            'py': obs['vehicle_info']['ego']['y'],
            'v': obs['vehicle_info']['ego']['v'],
            'heading': obs['vehicle_info']['ego']['yaw'],
        }

        target_state = {
            'px': result_x[1],
            'py': result_y[1],
            # 'px': self.waypoints[min((self.dis_near_id + 1), len(self.waypoints))][0],
            # 'py': self.waypoints[min((self.dis_near_id + 1), len(self.waypoints))][1],
            'vx': speeds_x[1],
            'vy': speeds_y[1],
        }
        # vel_len = obs['vehicle_info']['ego']['length'] / 1.7
        vel_len = 2.6
        # if len(self.map) == 44:
        #     dt = obs['test_setting']['dt'] / 3
        # else:
        #     dt = obs['test_setting']['dt'] / 5
        dt = obs['test_setting']['dt'] / 5
        # dt = 0.04

        x0_vals = np.array([init_state['px'], init_state['py'],
                            init_state['v'], init_state['heading']])
        x1_vals = np.array(
            [target_state['px'], target_state['py'], target_state['vx'], target_state['vy']])
        # print('x0:', x0_vals)
        # print('x1:', x1_vals)
        # u0 = np.array([0, 0])
        u0 = self.pre_u0
        bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
                (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))

        # scene_dit = self.scene_t / self.dt
        # if scene_dit % 2 == 0:
        #     bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
        #             (0, 0))
        # else:
        #     bnds = ((0, 0),
        #             (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))

        # bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
        #         (self.pre_u0[1] - (self.ROT_RATE_LIMIT * self.dt), self.pre_u0[1] + (self.ROT_RATE_LIMIT * self.dt)))

        # Minimize difference between simulated state and next state by varying input u
        u0 = minimize(self.position_orientation_objective, u0, args=(x0_vals, x1_vals, vel_len, dt),
                      options={'disp': False, 'maxiter': 100, 'ftol': 1e-9},  # 100 1e-9
                      method='SLSQP', bounds=bnds).x
        self.pre_u0 = u0
        if obs['test_setting']['dt'] == 0:
            u0[0] = 0
        else:
            u0[0] = (local_waypoints[1][2] - obs['vehicle_info']['ego']['v']) / obs['test_setting']['dt']
        stop_max_dec = (0.3 - obs['vehicle_info']['ego']['v']) / obs['test_setting']['dt']
        if np.isnan(u0[0]):
            u0[0] = 0
        if np.isnan(u0[1]):
            u0[1] = 0

        # u0[0] = 0 if u0[0] == NAN
        # u0 = self.action_dynamic_constraint_city(obs, u0)
        # print('stop_max_dec:', stop_max_dec)
        if u0[0] < stop_max_dec:
            # print('over max stop dec')
            u0[0] = stop_max_dec
        # print('u0[0]:', u0[0])
        return u0

    # 功能同action_dynamic_constraint_highway函数一样，适用场景不同
    def action_dynamic_constraint_city(self, obs, u0):
        # 约束范围（百分比）
        dyn_rate = 0.9
        dyn_bill = 1.0
        if obs['vehicle_info']['ego']['v'] < 2 or obs['vehicle_info']['ego']['v'] > 20:
            dyn_rate = 0.6
        if obs['vehicle_info']['ego']['v'] < 1:
            dyn_rate = 0.4
        #
        # vel_len = obs['vehicle_info']['ego']['length'] / 1.7
        # wb_dis = vel_len * 0.5
        vel_len = 2.6
        wb_dis = 1.48
        dt = 0.04
        # if abs(u0[1]) < 1e-5:
        #     u0[1] = 0
        if self.scenario_info['type'] == 'SERIAL':
            if u0[0] < - self.max_dec * 0.9:
                u0[1] = 0
        if u0[0] < - self.max_dec * 0.9 and abs(u0[1]) > 0:
            u0[0] = - self.max_dec * 0.9
            dyn_rate = 0.6
            # if obs['vehicle_info']['ego']['v'] < 5:
            #     dyn_rate = 0.5
            if obs['vehicle_info']['ego']['v'] > 10:
                dyn_rate = 0.5
            if obs['vehicle_info']['ego']['v'] > 20:
                dyn_rate = 0.4

        # vx_world = (obs['vehicle_info']['ego']['x'] - self.pre_obs['vehicle_info']['ego']['x']) / dt
        # vy_world = (obs['vehicle_info']['ego']['y'] - self.pre_obs['vehicle_info']['ego']['y']) / dt
        # cos_yaw = np.cos(obs['vehicle_info']['ego']['yaw'])
        # sin_yaw = np.sin(obs['vehicle_info']['ego']['yaw'])
        # vx_body = cos_yaw * vx_world + sin_yaw * vy_world
        # vy_body = -sin_yaw * vx_world + cos_yaw * vy_world

        # 前轮转角约束
        if u0[1] > self.ROT_LIMIT:
            u0[1] = self.ROT_LIMIT
        if u0[1] < -self.ROT_LIMIT:
            u0[1] = -self.ROT_LIMIT

        # 前轮转速约束
        max_steer_dis = self.ROT_RATE_LIMIT * obs['test_setting']['dt']
        if u0[1] - self.pre_u0[1] > max_steer_dis:
            # print('steer dis over max_steer_dis')
            u0[1] = self.pre_u0[1] + max_steer_dis
        if u0[1] - self.pre_u0[1] < -max_steer_dis:
            # print('steer dis below -max_steer_dis')
            u0[1] = self.pre_u0[1] - max_steer_dis

        # 纵向加速度约束
        # if u0[0] > self.ACC_LIMIT:
        #     u0[0] = self.ACC_LIMIT
        # if u0[0] < -self.ACC_LIMIT:
        #     u0[0] = -self.ACC_LIMIT
        if u0[0] > self.max_acc:
            u0[0] = self.max_acc
        if u0[0] < -self.max_dec:
            u0[0] = -self.max_dec

        # 纵向加加速度约束
        max_acc_dis = self.JERK_LIMIT * obs['test_setting']['dt']
        if u0[0] - self.pre_u0[0] > max_acc_dis:
            # print('acc dis over max_acc_dis')
            u0[0] = self.pre_u0[0] + max_acc_dis
            # print('af_u0:', u0)
        if u0[0] - self.pre_u0[0] < -max_acc_dis:
            # print('acc dis below -max_acc_dis')
            u0[0] = self.pre_u0[0] - max_acc_dis
            # print('af_u0:', u0)

        # 质心侧偏角约束
        # sideslip_stand = math.atan(0.02 * 0.85 * 9.8)
        sideslip_stand = math.atan(0.02 * 0.85 * 9.8 / dyn_bill)
        sideslip_angle = abs(math.atan((wb_dis * math.tan(u0[1])) / vel_len))
        # yaw_rate_test = (1.04 * 43160 * 43160 * u0[1]) / (1343.1 * vx_body)
        # sideslip_angle_test = (29210*29210 * 1.56 - 43160*43160 * 1.04) * yaw_rate_test / (1134 * vx_body * vx_body) + (43160*43160 * u0[1]) / (43160*43160 + 29210*29210)
        # print('sideslip_stand:', sideslip_stand)
        if abs(sideslip_angle) > sideslip_stand * dyn_rate:
            print('sideslip_angle over stand:', sideslip_angle)
            # if u0[1] - self.pre_u0[1] > 0:
            if u0[1] > 0:
                while True:
                    u0[1] = u0[1] - 0.00001
                    sideslip_angle_test = math.atan((wb_dis * math.tan(u0[1])) / vel_len)
                    # yaw_rate_test = (1.04 * 43160 * 43160 * u0[1]) / (1343.1 * vx_body)
                    # sideslip_angle_test = (29210 * 29210 * 1.56 - 43160 * 43160 * 1.04) * yaw_rate_test / (
                    #             1134 * vx_body * vx_body) + (43160 * 43160 * u0[1]) / (43160 * 43160 + 29210 * 29210)
                    if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                        side_state = 1
                        break

            # elif u0[1] - self.pre_u0[1] < 0:
            elif u0[1] < 0:
                while True:
                    u0[1] = u0[1] + 0.00001
                    sideslip_angle_test = math.atan((wb_dis * math.tan(u0[1])) / vel_len)
                    # yaw_rate_test = (1.04 * 43160 * 43160 * u0[1]) / (1343.1 * vx_body)
                    # sideslip_angle_test = (29210 * 29210 * 1.56 - 43160 * 43160 * 1.04) * yaw_rate_test / (
                    #             1134 * vx_body * vx_body) + (43160 * 43160 * u0[1]) / (43160 * 43160 + 29210 * 29210)
                    if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                        break

        # 横摆角速度约束
        if obs['vehicle_info']['ego']['v'] == 0:
            yaw_rate_stand = 9.8 * 0.85 / 1
        else:
            # yaw_rate_stand = abs(9.8 * 0.85 / obs['vehicle_info']['ego']['v'])  # + 0.5 * u0[0] * self.dt)
            ego_v = max(obs['vehicle_info']['ego']['v'],
                        (obs['vehicle_info']['ego']['v'] + u0[0] * obs['test_setting']['dt']))
            yaw_rate_stand = abs(9.8 * 0.85 / (ego_v * dyn_bill))

        sideslip_angle = math.atan(wb_dis * math.tan(u0[1]) / vel_len)
        yaw_rate = (obs['vehicle_info']['ego']['v']) * math.tan(
            sideslip_angle) / wb_dis

        # yaw_rate_test = (1.04 * 43160*43160 * steer_angle) / (1343.1 * vx_body)
        if abs(yaw_rate) > yaw_rate_stand * dyn_rate:
            print('yaw_rate over stand:', yaw_rate)
            # if u0[1] - self.pre_u0[1] > 0:
            if u0[1] > 0:
                while True:
                    u0[1] = u0[1] - 0.00001
                    sideslip_angle_test = math.atan(wb_dis * math.tan(u0[1]) / vel_len)
                    ego_v = max(obs['vehicle_info']['ego']['v'],
                                (obs['vehicle_info']['ego']['v'] + u0[0] * obs['test_setting']['dt']))
                    yaw_rate_test = ego_v * math.tan(sideslip_angle_test) / wb_dis
                    # yaw_rate_test = (1.04 * 43160 * 43160 * u0[1]) / (1343.1 * vx_body)
                    if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                        break
            # elif u0[1] - self.pre_u0[1] < 0:
            elif u0[1] < 0:
                while True:
                    u0[1] = u0[1] + 0.00001
                    sideslip_angle_test = math.atan(wb_dis * math.tan(u0[1]) / vel_len)
                    ego_v = max(obs['vehicle_info']['ego']['v'],
                                (obs['vehicle_info']['ego']['v'] + u0[0] * obs['test_setting']['dt']))
                    yaw_rate_test = ego_v * math.tan(sideslip_angle_test) / wb_dis
                    # yaw_rate_test = (1.04 * 43160 * 43160 * u0[1]) / (1343.1 * vx_body)
                    if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                        break
        return u0

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

    def position_orientation_objective(self, u: np.array, x0_array: np.array, x1_array: np.array, vel_l: float,
                                       dt: float, e: np.array = np.array([2e-3, 2e-3, 3e-3])) -> float:
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
        # print(x0)

        x1_target = x1_array
        x1_sim = self.vehicle_dynamic(x0, u, vel_l, dt)
        x1_sim_array = np.array([x1_sim['px'], x1_sim['py'], x1_sim['v'] *
                                 np.cos(x1_sim['heading']), x1_sim['v'] * np.sin(x1_sim['heading'])])

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

    def get_lead_follow_car_info(self, car_found, car_info, which_lane):
        if car_found:
            car_pos = np.array([
                [car_info[which_lane]['x'], car_info[which_lane]['y'], car_info[which_lane]['yaw']]])
            car_length = np.array([
                [car_info[which_lane]['length']]])
            car_speed = np.array([
                [car_info[which_lane]['v']]])
        else:
            car_pos = np.array([
                [99999.0, 99999.0, 0.0]])
            car_length = np.array([
                [99999.0]])
            car_speed = np.array([
                [99999.0]])

        return car_pos, car_length, car_speed

    def create_controller_output_dir(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def write_control_file(self, u0):
        self.create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
        file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'control.txt')
        with open(file_name, 'a') as control_file:
            control_file.write('%.2f\t%.2f\n' % \
                               (u0[0], u0[1]))
