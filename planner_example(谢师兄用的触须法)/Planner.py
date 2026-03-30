from vehicle_control import PID
import numpy as np
from scipy import integrate
from EM_Planner import Speed_planner, traj_index_trans2s_vec, virtual_dynamic_obs_planning, traj_index_trans2s
import local_planner
import behavioural_planner
import time
import EM_Planner

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
        x = np.linspace(0, 300, int(300 / 0.1) + 1)
        y = np.ones_like(x) * 4
        heading = np.zeros_like(x)
        self.waypoints = np.vstack((x, y, heading))
        self.waypoint_kappa = np.zeros_like(x)
        self.target_acc= []
        self.plot = True

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
        self.dt = dt
        self.paths = None
        self.best_index = 0
        self.prev_timestamp = 0
        self.time = 0
        self.dt = dt
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
        self.s_pid = PID(KP=1.0, KI=0.01, KD=0.1)
        self.v_pid = PID(KP=0.7, KI=0.01, KD=0.5)
        self.planner_begin = [0, 0]
        self.index2s = None
        self.speed_planner = Speed_planner(self.plot)
        self.first_run = True
        self.pre_traj = None
        self.traj_xy = None
        self.s_final = None
        self.s_dot_final = None
        self.traj_x_final = None
        self.traj_y_final = None
        self.theta_final = None
        self.s_dot2_final = None
        self.replanner = False


    def exec_waypoint_nav_demo_em_stitch_sn(self, av):
        """
        这个版涉及曲线拼接
        """
        obs = av._get_veh_obs()
        dyn_obs_x_list = []
        dyn_obs_y_list = []
        dyn_obs_vx_list = []
        dyn_obs_vy_list = []

        for vehi in obs:
            if vehi is not None:
                longitudinal, lateral = vehi.position
                dyn_obs_x_list.append(longitudinal)
                dyn_obs_y_list.append(lateral)
                dyn_obs_vx_list.append(vehi.velocity * np.cos(vehi.heading))
                dyn_obs_vy_list.append(vehi.velocity * np.sin(vehi.heading))

        dyn_obs_x = np.array(dyn_obs_x_list)
        dyn_obs_y = np.array(dyn_obs_y_list)
        dyn_obs_vx = np.array(dyn_obs_vx_list)
        dyn_obs_vy = np.array(dyn_obs_vy_list)

        # 当前参数获取
        current_x, current_y = av.position
        current_yaw = av.heading
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
                stitch_s_dot = np.array([])
                stitch_theta = np.array([])
                stitch_s_dot_2 = np.array([])
            else:
                if self.replanner is True:
                    plan_start_x = current_x
                    plan_start_y = current_y
                    plan_start_heading = current_yaw
                    plan_start_vx = current_speed * np.cos(plan_start_heading)
                    plan_start_vy = current_speed * np.sin(plan_start_heading)
                    plan_start_ax = -4
                    plan_start_ay = 0
                    stitch_x = np.array([])
                    stitch_y = np.array([])
                    stitch_s_dot = np.array([])
                    stitch_theta = np.array([])
                    stitch_s_dot_2 = np.array([])
                    self.replanner = False
                else:
                    traj_x, traj_y, traj_h, s, s_dot, s_dot2, time_array = self.pre_traj
                    time_index = np.argmin(abs(time_array - (self.time + 0.1)))
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
                    stitch_s_dot = s_dot[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                    stitch_s_dot_2 = s_dot2[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                    stitch_theta = traj_h[time_index - self.LP_FREQUENCY_DIVISOR:time_index]

            # begin = time.time()
            # 相对时间置0， 其作用是可用来索引期望的s
            self.relative_time = 0
            lookahead = 90  # self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed        #直接设置成60m
            ego_state = [plan_start_x, plan_start_y, plan_start_heading, av.velocity, ego_length]
            mypath = self.plan_paths_sn_90(ego_state, self.waypoints, self.waypoint_kappa)

            self.paths = mypath[:, :2, :]
            # self.paths的维度是Nx3xM
            index2s_set = traj_index_trans2s_vec(self.paths[:, 0, :], self.paths[:, 1, :])
            # print(time.time() - begin,"-----")
            # 动态障碍物网格划分
            time_step = np.linspace(0, lookahead / av.velocity, 11)

            best_index = self.select_best_path_index_numpy_dyn(av, dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy,
                                                               index2s_set, time_step)
            self.best_index = best_index
            self.traj = self.paths[best_index, ...]

            # 计算生成的局部路径的航向角和曲率
            self.theta, self.trajectory_kappa_init = mypath[best_index, 2, :], mypath[best_index, 3, :]  # EM_Planner.cal_heading_kappa(self.traj[0, :], self.traj[1, :])

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
            if self.plot is True:
                self.speed_planner.plot_ST_graph(obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                 dp_speed_s, dp_speed_t)
            if len(dp_speed_s) > 1:
                # 不知道什么原因，规划的路径有可能小于30，那我直接赋值就可以了
                self.speed_planner.set_s_list(self.index2s[-1])
                # 根据动态规划结果生成凸空间并计算凸空间的上下界约束
                s_lb, s_ub, s_dot_lb, s_dot_ub = \
                    self.speed_planner.convex_space_gen(dp_speed_s, dp_speed_t, self.index2s, obs_st_s_in_set,
                                                        obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                        self.trajectory_kappa_init, self.max_lateral_accel)
                # 二次规划解决速度规划问题
                try:
                    qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init, status = \
                        self.speed_planner.speed_planning_osqp(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t,
                                                               s_lb, s_ub, s_dot_lb, s_dot_ub)
                except:
                    print('二次出错')
                    state = [av.position[0], av.position[1], av.heading]
                    lane_pos = av.lane_index[2] * 4
                    steer = self.pp_controller(lane_pos, state)
                    self.replanner = True
                    self.reset_state()
                    return [-4, steer]
                if status != 'solved':
                    print('二次规划失败')
                    state = [av.position[0], av.position[1], av.heading]
                    self.replanner = True
                    lane_pos = av.lane_index[2] * 4
                    steer = self.pp_controller(lane_pos, state)
                    self.reset_state()
                    return [-4, steer]
                # 为二次规划算的变量增密
                self.s, self.s_dot, s_dot2, self.relative_time_set = \
                    self.speed_planner.increase_points(qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init)
                # t3 = time.time()
                # print(t3 - begin)
                traj = self.speed_planner.merge_path(self.s, self.index2s, self.traj[0, :], self.traj[1, :], self.theta,
                                                     self.s_dot, s_dot2, self.relative_time_set + self.time)
                x, y, heading, s, s_dot, s_dot_2, time_final = traj
                self.s_dot_final = np.hstack((stitch_s_dot, s_dot))
                self.traj_x_final = np.hstack((stitch_x, x))
                self.traj_y_final = np.hstack((stitch_y, y))
                self.theta_final = np.hstack((stitch_theta, heading))
                self.s_dot2_final = np.hstack((stitch_s_dot_2, s_dot_2))
                self.s_final = traj_index_trans2s(self.traj_x_final, self.traj_y_final)
                self.pre_traj = self.traj_x_final, self.traj_y_final, self.theta_final, self.s_final, self.s_dot_final, \
                    self.s_dot2_final, time_final
                self.first_run = False
                self.traj_xy = np.hstack((self.traj_x_final.reshape(-1, 1), self.traj_y_final.reshape(-1, 1))).T

            # print('time cost: ', time.time() - begin)
            self.prev_timestamp = self.time

        targets_index = int(self.relative_time * 40)
        target_s = self.s_final[targets_index]
        # 找到车辆在当前位置在s方向的投影
        real_s, _, _, _, _ = \
            EM_Planner.find_match_point(np.array([current_x - self.traj_x_final[0]]),
                                        np.array([current_y - self.traj_y_final[0]]),
                                        self.traj_x_final - self.traj_x_final[0], self.traj_y_final - self.traj_y_final[0],
                                        self.theta_final, self.trajectory_kappa_init)
        err_s = target_s - real_s[0]
        # print(err_s, "     ", real_s[0])
        delta_pos = self.s_pid.output_cal(err_s)
        acc = self.v_pid.output_cal(self.s_dot_final[targets_index] - current_speed + delta_pos) + self.s_dot2_final[targets_index]
        self.target_acc.append(self.s_dot2_final[targets_index])
        state = [av.position[0], av.position[1], av.heading]
        steer = self.pp_controller_xy(state, current_speed)

        self.relative_time += 0.025
        self.frame += 1
        self.time += 0.025
        return [acc, steer]

    def exec_waypoint_nav_demo_em_stitch_sn_qp4(self, av):
        """
        这个版涉及曲线拼接
        """
        obs = av._get_veh_obs()
        dyn_obs_x_list = []
        dyn_obs_y_list = []
        dyn_obs_vx_list = []
        dyn_obs_vy_list = []

        for vehi in obs:
            if vehi is not None:
                longitudinal, lateral = vehi.position
                dyn_obs_x_list.append(longitudinal)
                dyn_obs_y_list.append(lateral)
                dyn_obs_vx_list.append(vehi.velocity * np.cos(vehi.heading))
                dyn_obs_vy_list.append(vehi.velocity * np.sin(vehi.heading))

        dyn_obs_x = np.array(dyn_obs_x_list)
        dyn_obs_y = np.array(dyn_obs_y_list)
        dyn_obs_vx = np.array(dyn_obs_vx_list)
        dyn_obs_vy = np.array(dyn_obs_vy_list)

        # 当前参数获取
        current_x, current_y = av.position
        current_yaw = av.heading
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
                plan_start_s_jerk = 0
                stitch_x = np.array([])
                stitch_y = np.array([])
                stitch_s_dot = np.array([])
                stitch_theta = np.array([])
                stitch_s_dot_2 = np.array([])
            else:
                if self.replanner is True:
                    plan_start_x = current_x
                    plan_start_y = current_y
                    plan_start_heading = current_yaw
                    plan_start_vx = current_speed * np.cos(plan_start_heading)
                    plan_start_vy = current_speed * np.sin(plan_start_heading)
                    plan_start_ax = -4
                    plan_start_ay = 0
                    plan_start_s_jerk = 0
                    stitch_x = np.array([])
                    stitch_y = np.array([])
                    stitch_s_dot = np.array([])
                    stitch_theta = np.array([])
                    stitch_s_dot_2 = np.array([])
                    self.replanner = False
                else:
                    traj_x, traj_y, traj_h, s, s_dot, s_dot2, time_array = self.pre_traj
                    time_index = np.argmin(abs(time_array - (self.time + 0.1)))
                    plan_start_x = traj_x[time_index]
                    plan_start_y = traj_y[time_index]
                    plan_start_heading = traj_h[time_index]
                    plan_start_vx = s_dot[time_index] * np.cos(plan_start_heading)
                    plan_start_vy = s_dot[time_index] * np.sin(plan_start_heading)
                    tor = np.array([np.cos(plan_start_heading), np.sin(plan_start_heading)])
                    # nor = np.array([-np.sin(plan_start_heading), np.cos(plan_start_heading)])
                    a_tor = s_dot2[time_index] * tor
                    plan_start_s_jerk = ((s_dot2[time_index-1] - s_dot2[time_index]) * tor)[0] / 0.025
                    # a_nor = s_dot[time_index] ** 2 * traj_k[time_index] * nor
                    plan_start_ax = a_tor[0]
                    plan_start_ay = a_tor[1]
                    stitch_x = traj_x[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                    stitch_y = traj_y[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                    stitch_s_dot = s_dot[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                    stitch_s_dot_2 = s_dot2[time_index - self.LP_FREQUENCY_DIVISOR:time_index]
                    stitch_theta = traj_h[time_index - self.LP_FREQUENCY_DIVISOR:time_index]

            # begin = time.time()
            # 相对时间置0， 其作用是可用来索引期望的s
            self.relative_time = 0
            lookahead = 90  # self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed        #直接设置成60m
            ego_state = [plan_start_x, plan_start_y, plan_start_heading, av.velocity, ego_length]
            mypath = self.plan_paths_sn_90(ego_state, self.waypoints, self.waypoint_kappa)

            self.paths = mypath[:, :2, :]
            # self.paths的维度是Nx3xM
            index2s_set = traj_index_trans2s_vec(self.paths[:, 0, :], self.paths[:, 1, :])
            # print(time.time() - begin,"-----")
            # 动态障碍物网格划分
            time_step = np.linspace(0, lookahead / av.velocity, 11)

            best_index = self.select_best_path_index_numpy_dyn(av, dyn_obs_x, dyn_obs_y, dyn_obs_vx, dyn_obs_vy,
                                                               index2s_set, time_step)
            self.best_index = best_index
            self.traj = self.paths[best_index, ...]

            # 计算生成的局部路径的航向角和曲率
            self.theta, self.trajectory_kappa_init = mypath[best_index, 2, :], mypath[best_index, 3, :]  # EM_Planner.cal_heading_kappa(self.traj[0, :], self.traj[1, :])

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
            if self.plot is True:
                self.speed_planner.plot_ST_graph(obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                 dp_speed_s, dp_speed_t)
            if len(dp_speed_s) > 1:
                # 不知道什么原因，规划的路径有可能小于30，那我直接赋值就可以了
                self.speed_planner.set_s_list(self.index2s[-1])
                # 根据动态规划结果生成凸空间并计算凸空间的上下界约束
                s_lb, s_ub, s_dot_lb, s_dot_ub = \
                    self.speed_planner.convex_space_gen(dp_speed_s, dp_speed_t, self.index2s, obs_st_s_in_set,
                                                        obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                                        self.trajectory_kappa_init, self.max_lateral_accel)
                # 二次规划解决速度规划问题
                try:
                    qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init, status, _ = \
                        self.speed_planner.speed_planning_osqp_4(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t,
                                                               s_lb, s_ub, s_dot_lb, s_dot_ub, plan_start_s_jerk)
                except:
                    print('二次出错')
                    state = [av.position[0], av.position[1], av.heading]
                    lane_pos = av.lane_index[2] * 4
                    steer = self.pp_controller(lane_pos, state)
                    self.replanner = True
                    self.reset_state()
                    return [-4, steer]
                if status != 'solved':
                    print('二次规划失败')
                    state = [av.position[0], av.position[1], av.heading]
                    self.replanner = True
                    lane_pos = av.lane_index[2] * 4
                    steer = self.pp_controller(lane_pos, state)
                    self.reset_state()
                    return [-4, steer]
                # 为二次规划算的变量增密
                self.s, self.s_dot, s_dot2, self.relative_time_set = \
                    self.speed_planner.increase_points(qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init)
                # t3 = time.time()
                # print(t3 - begin)
                traj = self.speed_planner.merge_path(self.s, self.index2s, self.traj[0, :], self.traj[1, :], self.theta,
                                                     self.s_dot, s_dot2, self.relative_time_set + self.time)
                x, y, heading, s, s_dot, s_dot_2, time_final = traj
                self.s_dot_final = np.hstack((stitch_s_dot, s_dot))
                self.traj_x_final = np.hstack((stitch_x, x))
                self.traj_y_final = np.hstack((stitch_y, y))
                self.theta_final = np.hstack((stitch_theta, heading))
                self.s_dot2_final = np.hstack((stitch_s_dot_2, s_dot_2))
                self.s_final = traj_index_trans2s(self.traj_x_final, self.traj_y_final)
                self.pre_traj = self.traj_x_final, self.traj_y_final, self.theta_final, self.s_final, self.s_dot_final, \
                    self.s_dot2_final, time_final
                self.first_run = False
                self.traj_xy = np.hstack((self.traj_x_final.reshape(-1, 1), self.traj_y_final.reshape(-1, 1))).T

            # print('time cost: ', time.time() - begin)
            self.prev_timestamp = self.time

        targets_index = int(self.relative_time * 40)
        target_s = self.s_final[targets_index]
        # 找到车辆在当前位置在s方向的投影
        real_s, _, _, _, _ = \
            EM_Planner.find_match_point(np.array([current_x - self.traj_x_final[0]]),
                                        np.array([current_y - self.traj_y_final[0]]),
                                        self.traj_x_final - self.traj_x_final[0], self.traj_y_final - self.traj_y_final[0],
                                        self.theta_final, self.trajectory_kappa_init)
        err_s = target_s - real_s[0]
        # print(err_s, "     ", real_s[0])
        delta_pos = self.s_pid.output_cal(err_s)
        acc = self.v_pid.output_cal(self.s_dot_final[targets_index] - current_speed) + delta_pos + self.s_dot2_final[targets_index]
        self.target_acc.append(self.s_dot2_final[targets_index])
        state = [av.position[0], av.position[1], av.heading]
        steer = self.pp_controller_xy(state, current_speed)

        self.relative_time += 0.025
        self.frame += 1
        self.time += 0.025
        return [acc, steer]


    def plan_paths_sn(self, ego_state, way_point, way_point_kappa):
        path = []
        locationX = np.array([ego_state[0]])
        locationY = np.array([ego_state[1]])
        proj_x_set, proj_y_set, proj_heading_set, _, proj_match_point_index_set = \
            EM_Planner.find_match_point(locationX, locationY, way_point[:, 0], way_point[:, 1], way_point[:, 2],
                                        way_point_kappa)
        index_s = EM_Planner.traj_index_trans2s(way_point[:, 0], way_point[:, 1])
        s_set, l_set = EM_Planner.CalcSL(locationX, locationY, way_point[:, 0], way_point[:, 1], proj_x_set,
                                         proj_y_set, proj_heading_set, proj_match_point_index_set, index_s)
        pc = np.tan(ego_state[2] - proj_heading_set[0])
        si = s_set[0] + proj_x_set[0]
        qi = l_set[0]
        for goal_state in range(-4, 5):
            Sf = 60
            qf0 = goal_state * 1
            pa = (pc * Sf + 2 * qi - 2 * qf0) / Sf ** 3
            pb = (-3 * pa * Sf ** 2 - pc) / (2 * Sf)
            # ss = np.arange(0, Sf+0.1, 0.2)
            ss = np.linspace(0, Sf, 301)
            Qs = pa * np.power(ss, 3) + pb * np.power(ss, 2) + pc * ss + qi
            dqs1 = 3 * pa * np.power(ss, 2) + 2 * pb * ss + pc
            ddqs1 = 6 * pa * ss + 2 * pb
            # S_G = si + ss
            index = proj_match_point_index_set[0]
            kkb = way_point_kappa[index:index + len(ss)]
            sign = 1 - Qs * kkb
            S = np.sign(sign)
            Q = np.sqrt(dqs1 ** 2 + sign ** 2)
            kkk = S / Q * (kkb + (sign * ddqs1 + kkb * dqs1 ** 2) / Q ** 2)
            theta = way_point[:, 2][index:index + len(ss)] + np.arctan(dqs1)
            XXX = np.hstack((locationX, locationX + integrate.cumtrapz(Q * np.cos(theta), ss)))
            YYY = np.hstack((locationY, locationY + integrate.cumtrapz(Q * np.sin(theta), ss)))
            path.append(np.vstack((XXX, YYY, theta, kkk)).tolist())
        return np.array(path)

    def plan_paths_sn_90(self, ego_state, way_point, way_point_kappa):
        path = []
        locationX = np.array([ego_state[0]])
        locationY = np.array([ego_state[1]])
        proj_x_set, proj_y_set, proj_heading_set, _, proj_match_point_index_set = \
            EM_Planner.find_match_point(locationX, locationY, way_point[:, 0], way_point[:, 1], way_point[:, 2],
                                        way_point_kappa)
        index_s = EM_Planner.traj_index_trans2s(way_point[:, 0], way_point[:, 1])
        s_set, l_set = EM_Planner.CalcSL(locationX, locationY, way_point[:, 0], way_point[:, 1], proj_x_set,
                                         proj_y_set, proj_heading_set, proj_match_point_index_set, index_s)
        pc = np.tan(ego_state[2] - proj_heading_set[0])
        si = s_set[0] + proj_x_set[0]
        qi = l_set[0]
        for goal_state in range(-4, 5):
            Sf = 60
            qf0 = goal_state * 1
            pa = (pc * Sf + 2 * qi - 2 * qf0) / Sf ** 3
            pb = (-3 * pa * Sf ** 2 - pc) / (2 * Sf)
            # ss = np.arange(0, Sf+0.1, 0.2)
            ss = np.linspace(0, Sf, 300)
            ss1 = np.linspace(Sf, Sf + 30, 151)
            Qs = pa * np.power(ss, 3) + pb * np.power(ss, 2) + pc * ss + qi
            Qs1 = np.ones_like(ss1) * qf0
            dqs1 = 3 * pa * np.power(ss, 2) + 2 * pb * ss + pc
            dqs2 = np.zeros_like(ss1)
            ddqs1 = 6 * pa * ss + 2 * pb
            ddqs2 = np.zeros_like(ss1)
            ss = np.concatenate((ss, ss1))
            Qs = np.concatenate((Qs, Qs1))
            dqs1 = np.concatenate((dqs1, dqs2))
            ddqs1 = np.concatenate((ddqs1, ddqs2))
            # S_G = si + ss
            index = proj_match_point_index_set[0]
            kkb = way_point_kappa[index:index + len(ss)]
            sign = 1 - Qs * kkb
            S = np.sign(sign)
            Q = np.sqrt(dqs1 ** 2 + sign ** 2)
            kkk = S / Q * (kkb + (sign * ddqs1 + kkb * dqs1 ** 2) / Q ** 2)
            theta = way_point[:, 2][index:index + len(ss)] + np.arctan(dqs1)
            XXX = np.hstack((locationX, locationX + integrate.cumtrapz(Q * np.cos(theta), ss)))
            YYY = np.hstack((locationY, locationY + integrate.cumtrapz(Q * np.sin(theta), ss)))
            path.append(np.vstack((XXX, YYY, theta, kkk)).tolist())
        return np.array(path)

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

    def reset_state(self):
        self.s_pid.clear_err()
        self.v_pid.clear_err()

    def pp_controller(self, lane_pos, state):
        l = 5
        pre_x = 40.0
        alpha = np.arctan2(lane_pos - state[1], pre_x) - state[2]
        ld = np.sqrt(pre_x ** 2 + (lane_pos - state[1]) ** 2)
        u = np.arctan(2 * l * np.sin(alpha) / ld)
        return np.clip(u, -np.pi/4, np.pi/4)

    def pp_controller_xy(self, state, speed):
        l = 5.0
        pre_index = max(min(int(speed), 30), 10)
        pre_ref_state = self.traj_xy[:, pre_index]
        alpha = np.arctan2(pre_ref_state[1] - state[1], pre_ref_state[0] - state[0]) - state[2]
        ld = np.sqrt((pre_ref_state[1] - state[1]) ** 2 + (pre_ref_state[0] - state[0]) ** 2)
        u = np.arctan(2 * l * np.sin(alpha) / ld)
        return np.clip(u, -np.pi/4, np.pi/4)
