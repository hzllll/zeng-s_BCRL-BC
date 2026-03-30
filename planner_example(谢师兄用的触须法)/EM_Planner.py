import time

import numpy as np
from collections import deque
from scipy.interpolate import interp1d
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
import osqp
from scipy import sparse

class ClassA:
    def __init__(self):
        # 在 ClassA 中创建 Figure 对象
        self.fig_a, self.ax_a = plt.subplots()

    def plot_a(self, x, y):
        self.ax_a.plot(x, y)
        self.ax_a.set_title('Plot in ClassA')
        plt.show()

class Speed_planner():
    def __init__(self, plot_enable):
        # 采用非均匀采样，s越小的越密，越大的越稀疏
        # self.s_list = np.concatenate([np.arange(0, 30.1, 0.75),
        #                               np.arange(30, 60.1, 1.5),
        #                               # np.arange(50, 60.1, 1.0),
        #                               ])
        self.s_list = np.arange(0, 90, 2)
        self.t_list = np.arange(1, 3.1, 1)
        self.s_length = len(self.s_list)
        self.t_length = len(self.t_list)
        self.w_cost_accel = 100.                  # 加速度代价权重
        self.w_cost_obs = 10000000.                    # 障碍物代价权重
        self.w_cost_ref_speed = 4000.              # 推荐速度代价权重
        self.w_cost_s_dot2 = 10                 # qp加速度代价权重
        self.w_cost_v_ref = 50                  # qp推荐速度代价权重
        self.w_cost_jerk = 500                  # qp jerk代价权重
        self.reference_speed = 30
        self.plot = plot_enable
        a = ClassA()
        if self.plot is True:
            self.fig1, self.ax1 = a.fig_a, a.ax_a
            plt.ion()

    def set_s_list(self, length_s):
        # self.s_list = np.concatenate([np.arange(0, 30.1, 0.6),
        #                               np.arange(30, 60.1, 1),
        #                               ])
        # self.t_list = np.arange(0.5, 2.1, 0.5)
        self.s_list = np.arange(0, 90.1, 2)
        self.t_list = np.arange(1, 3.1, 1)

    def plot_ST_graph(self, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, s, t):
        self.ax1.clear()
        for i in range(len(obs_st_s_in_set)):
            if not np.isnan(obs_st_s_in_set[i]):
                x = np.array([obs_st_t_in_set[i], obs_st_t_out_set[i]])
                y = np.array([obs_st_s_in_set[i], obs_st_s_out_set[i]])
                self.ax1.plot(x, y)
        self.ax1.plot(np.hstack((0, t)), np.hstack((0, s)))
        self.ax1.set_xlim(0, 3)
        self.ax1.set_ylim(0, 90)
        plt.draw()
        plt.pause(0.01)

    def close_plot(self):
        plt.close()

    def speed_dp(self, obs_st_s_in_set, obs_st_s_out_set,
                 obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot):
        # 初始化所有的权重都是inf, st矩阵代表从起点开始到(i,j)点的最小值代价为dp_st_cost(i,j),邻接矩阵
        dp_st_cost = np.ones((self.s_length, self.t_length)) * np.inf
        # dp_st_s_dot表示从起点开始到(i,j)点的最优路径的末速度
        dp_st_s_dot = np.zeros((self.s_length, self.t_length), dtype=float)
        # 使用一个矩阵保存最优路径的前一个节点方便回溯
        # dp_st_node(i,j) 表示位置为(i,j)的节点中，最优的上一层节点的行号为dp_st_node(i,j)
        dp_st_node = np.zeros((self.s_length, self.t_length), dtype=float)
        for i in range(self.s_length):
            dp_st_cost[i, 0] = self.CalcDpCost(0, 0, i, 0, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                                                obs_st_t_out_set, plan_start_s_dot, dp_st_s_dot)
            # 计算第一列所有节点的的s_dot，并存储到dp_st_s_dot中
            # 第一列的前一个节点只有起点，起点的s t 都是0
            s_end, t_end = self.CalcSTCoordinate(i, 0)
            dp_st_s_dot[i, 0] = s_end / t_end
        count = 0
        for i in range(1, self.t_length):
            for j in range(self.s_length):
                cur_row = j
                cur_col = i

                for k in range(self.s_length):
                    begin = time.time()
                    pre_row = k
                    pre_col = i - 1
                    cost_temp = self.CalcDpCost(pre_row, pre_col, cur_row, cur_col, obs_st_s_in_set, obs_st_s_out_set,
                                                obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot, dp_st_s_dot)
                    if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                        dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                        # 计算最优的s_dot
                        s_start, t_start = self.CalcSTCoordinate(pre_row, pre_col)
                        s_end, t_end = self.CalcSTCoordinate(cur_row, cur_col)
                        dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / (t_end - t_start)
                        # 把最优前驱节点的行号记录
                        dp_st_node[cur_row, cur_col] = pre_row
                    if time.time() - begin > 1e-5:
                        count += 1
        # print(count)
        dp_speed_s = deque(maxlen=self.t_length)
        dp_speed_t = deque(maxlen=self.t_length)
        # 找到右边界代价最小的点
        min_cost_index = np.argmin(dp_st_cost[:, -1])
        min_cost = dp_st_cost[min_cost_index, -1]
        # 找到上边界代价最小的点
        min_cost_index1 = np.argmin(dp_st_cost[0, :])
        min_cost1 = dp_st_cost[0, min_cost_index1]
        if min_cost < min_cost1:
            min_col = self.t_length - 1
            min_row = min_cost_index
        else:
            min_col = min_cost_index1
            min_row = 0
        # 输出终点st坐标
        s, t = self.CalcSTCoordinate(min_row, min_col)
        # 用min_col列作为基准，因为每个t一定对应一个s, 而有些s对应的t上面可能没有数据
        dp_speed_s.appendleft(s)
        dp_speed_t.appendleft(t)
        # 反向回溯
        while min_col > 0:
            # 求前驱节点
            pre_row = dp_st_node[int(min_row), int(min_col)]
            pre_col = min_col - 1
            # 求前驱节点对应的st坐标
            s, t = self.CalcSTCoordinate(pre_row, pre_col)
            dp_speed_s.appendleft(s)
            dp_speed_t.appendleft(t)
            min_row = pre_row
            min_col = pre_col
        return np.array(dp_speed_s), np.array(dp_speed_t)


    def CalcDpCost(self, row_start, col_start, row_end, col_end, obs_st_s_in_set,
                   obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot, dp_st_s_dot):
        """
        该函数将计算链接两个节点之间边的代价
        :param row_start:边的起点的行号
        :param col_start:边的起点的列号
        :param row_end:边终点的行号
        :param col_end:边终点的列号
        :param obs_st_s_in_set:障碍物st信息
        :param obs_st_s_out_set:障碍物st信息
        :param obs_st_t_in_set:障碍物st信息
        :param obs_st_t_out_set:障碍物st信息
        :param plan_start_s_dot:拼接起点的速度
        :param dp_st_s_dot:用于计算加速度
        :return:start到end的代价
        """
        # 计算终点st坐标
        s_end, t_end = self.CalcSTCoordinate(row_end, col_end)
        # 规定起点的行列号为0
        if row_start == 0:
            # 边的起始点为dp的起点
            s_start = 0
            t_start = 0
            s_dot_start = plan_start_s_dot
        else:
            # 边的起点不是dp起点
            s_start, t_start = self.CalcSTCoordinate(row_start, col_start)
            s_dot_start = dp_st_s_dot[row_start, col_start]
        cur_s_dot = (s_end - s_start) / (t_end - t_start)
        if cur_s_dot < 0:
            return np.inf
        cur_s_dot2 = (cur_s_dot - s_dot_start) / (t_end - t_start)
        # 计算推荐速度代价
        cost_ref_speed = self.w_cost_ref_speed * (cur_s_dot - self.reference_speed) ** 2
        # 加速度代价
        if 4 > cur_s_dot2 > -10:
            cost_accel = self.w_cost_accel * cur_s_dot2 ** 2
        else:
            # 超过车辆运动学限制，增大代价
            return np.inf
            # cost_accel = 100000 * self.w_cost_accel * cur_s_dot2 ** 2
        cost_obs = self.CalcObsCost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set)
        # 计算总代价
        cost = cost_obs + cost_accel + cost_ref_speed
        return cost


    def CalcSTCoordinate(self, row, col):
        """
        该函数将计算矩阵节点的s t 坐标
        矩阵的(1,1) 代表的是最左上角的元素 s 最大 t 最小
        :param row:节点在矩阵的行号
        :param col:节点在矩阵的列号
        :return:矩阵节点的s t 坐标
        """
        s_value = self.s_list[self.s_length - int(row) - 1]
        t_value = self.t_list[col]
        return s_value, t_value

    def CalcObsCost(self,s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                    obs_st_t_out_set):
        """
        该函数将计算边的障碍物代价
            params: 边的起点终点s_start,t_start,s_end,t_end
                    障碍物信息 obs_st_s_in_set,obs_st_s_out_set,obs_st_t_in_set,obs_st_t_out_set
                    障碍物代价权重w_cost_obs
            return: 边的障碍物代价obs_cost
        """
        obs_cost = 0
        # 采样点的个数
        n = 2
        # 采样时间间隔
        dt = (t_end - t_start) / (n - 1)
        # 边的斜率
        k = (s_end - s_start) / (t_end - t_start)
        for i in range(n):
            # 计算采样点的坐标
            t = t_start + (i - 1) * dt
            s = s_start + k * (i - 1) * dt
            # 遍历所有障碍物
            for j in range(len(obs_st_s_in_set)):
                if np.isnan(obs_st_s_in_set[j]):
                    continue
                # 计算点到st线段的最短距离
                vector1 = np.array([obs_st_s_in_set[j], obs_st_t_in_set[j]]) - np.array([s, t])
                vector2 = np.array([obs_st_s_out_set[j], obs_st_t_out_set[j]]) - np.array([s, t])
                vector3 = vector2 - vector1
                dis1 = np.sqrt(vector1.dot(vector1))
                dis2 = np.sqrt(vector2.dot(vector2))
                dis3 = abs(vector1[0] * vector3[1] - vector1[1] * vector3[0]) / np.sqrt(vector3.dot(vector3))
                if (vector1.dot(vector3) > 0 and vector2.dot(vector3) > 0) or (
                        vector1.dot(vector3) < 0 and vector2.dot(vector3) < 0):
                    min_dis = min(dis1, dis2)
                else:
                    min_dis = dis3
                obs_cost = obs_cost + self.CalcCollisionCost(min_dis)
        return obs_cost


    def CalcCollisionCost(self, min_dis):
        """
        计算obs的cost
        :param w_cost_obs:权重
        :param min_dis:最小距离
        :return:
        """
        if abs(min_dis) < 0.5:
            collision_cost = self.w_cost_obs
        elif 0.5 < abs(min_dis) < 1.5:
            # min_dis = 0.5    collision_cost = w_cost_obs**1
            # min_dis = 1.5    collision_cost = w_cost_obs**0 = 1
            collision_cost = self.w_cost_obs ** ((0.5 - min_dis) + 1)
        else:
            collision_cost = 0

        return collision_cost


    def convex_space_gen(self, dp_speed_s, dp_speed_t, path_index2s, obs_st_s_in_set, obs_st_s_out_set,
                            obs_st_t_in_set,obs_st_t_out_set,trajectory_kappa_init,max_lateral_accel):
        """
        该函数将计算出s,s_dot的上下界，开辟凸空间供二次规划使用
        :param dp_speed_s:
        :param dp_speed_t:
        :param path_index2s:
        :param obs_st_s_in_set:
        :param obs_st_s_out_set:
        :param obs_st_t_in_set:
        :param obs_st_t_out_set:
        :param trajectory_kappa_init:
        :param max_lateral_accel:
        :return:
        """
        n = len(dp_speed_s)
        s_lb = np.ones(n, dtype=float) * -1e10       # 下边界初始化
        s_ub = np.ones(n, dtype=float) * 1e10       # 上边界初始化
        # s_dot_lb = np.ones((n, 1), dtype=float) * -np.inf   # 速度上边界初始化
        # s_dot_ub = np.ones((n, 1), dtype=float) * np.inf    # 速度下边界初始化
        dp_speed_end_index = np.size(dp_speed_s)
        path_index2s_end_index = np.size(path_index2s)

        # 施加车辆动力学约束
        # for i in range(n):
        #     cur_s = dp_speed_s[i]
        si = interp1d(path_index2s[1:path_index2s_end_index], trajectory_kappa_init[1:path_index2s_end_index])
        # cur_kappa = si(dp_speed_s)
        s_dot_ub = np.ones(n, dtype=float) * 1e10#np.sqrt(max_lateral_accel / (np.abs(cur_kappa) + 10e-10))
        s_dot_lb = np.zeros(np.shape(s_dot_ub))
        # 创建插值函数
        dp_s_inter = interp1d(np.append(0, dp_speed_t), np.append(0, dp_speed_s))

        for i in range(len(obs_st_s_in_set)):
            if np.isnan(obs_st_s_in_set[i]):
                continue
            # 计算障碍物的纵向速度
            obs_speed = (obs_st_s_out_set[i] - obs_st_s_in_set[i]) / (obs_st_t_out_set[i] - obs_st_t_in_set[i])
            # 取s t直线的中点,作为obs_s obs_t的坐标
            obs_t = (obs_st_t_in_set[i] + obs_st_t_out_set[i]) / 2
            obs_s = (obs_st_s_in_set[i] + obs_st_s_out_set[i]) / 2
            # 在跑动过程中发现计算出来的情况肯呢个会出现t大于规划的时间，那么这里的主要作用是对大于t的内容进行截断
            max_dp_v_t = np.max(dp_speed_t)
            if obs_t > max_dp_v_t:
                obs_t = max_dp_v_t
                obs_s = obs_st_s_in_set[i] + obs_speed * (max_dp_v_t - obs_st_t_in_set[i])
            # 插值找到当t = obs_t时，dp_speed_s的值
            dp_s = dp_s_inter(obs_t)
            # 找到dp_speed_t中与obs_st_t_in_set(i)最近的时间，并将此时间的编号赋值给t_lb_index
            index_t_in = np.argwhere(dp_speed_t <= obs_st_t_in_set[i])
            t_lb_index = t_ub_index = 0
            if dp_speed_t[0] > obs_st_t_in_set[i]:
                t_lb_index = 0
            elif dp_speed_t[-1] < obs_st_t_in_set[i]:
                t_lb_index = len(dp_speed_t)
            elif len(index_t_in) > 0:
                t_lb_index = index_t_in[-1, 0]
            # 找到dp_speed_t中与obs_st_t_out_set(i)最近的时间，并将此时间的编号赋值给t_ub_index
            index_t_out = np.argwhere(dp_speed_t <= obs_st_t_out_set[i])
            if dp_speed_t[0] > obs_st_t_out_set[i]:
                t_ub_index = 0
            elif len(index_t_out) > 0:
                t_ub_index = index_t_out[-1, 0]+1

            # 稍微做个缓冲，把 t_lb_index 稍微缩小一些，t_ub_index稍微放大一些
            # t_lb_index = max(t_lb_index - 2, 1)
            # t_ub_index = min(t_ub_index + 2, dp_speed_end_index)
            bound = obs_st_s_in_set[i] + obs_speed * (dp_speed_t[t_lb_index:t_ub_index] - obs_st_t_in_set[i])
            if obs_s > dp_s:
                # 决策减速避让,更新上边界
                s_ub[t_lb_index:t_ub_index] = np.min(np.vstack((bound-15, s_ub[t_lb_index:t_ub_index])), axis=0)
            else:
                # 决策超车,更新
                s_lb[t_lb_index:t_ub_index] = np.max(np.vstack((bound+6, s_lb[t_lb_index:t_ub_index])), axis=0)

        return s_lb, s_ub, s_dot_lb, s_dot_ub


    def speed_planning_osqp(self, plan_start_s_dot, plan_start_s_dot2,dp_speed_s,
                          dp_speed_t,s_lb,s_ub,s_dot_lb,s_dot_ub):
        n = len(s_lb) + 1
        qp_s_init = np.ones(n) * np.nan
        qp_s_dot_init = np.ones(n) * np.nan
        qp_s_dot2_init = np.ones(n) * np.nan
        relative_time_init = np.ones(n) * np.nan

        dp_speed_end = np.size(dp_speed_s)
        recommend_T = dp_speed_t[-1]
        # qp的规模应该是dp的有效元素的个数 + 规划起点
        qp_size = dp_speed_end + 1
        Aeq = np.zeros((3 * qp_size, 2 * qp_size - 2))
        beq = np.zeros((2 * qp_size - 2, 1))
        # 不等式约束初始化
        lb = np.ones(3 * qp_size)
        ub = np.ones(3 * qp_size)
        # 计算采样间隔时间dt
        dt = recommend_T / dp_speed_end
        A_sub = np.array([[1, 0],
                          [dt, 1],
                          [(1 / 3) * dt ** 2, (1 / 2) * dt],
                          [-1, 0],
                          [0, -1],
                          [(1 / 6) * dt ** 2, dt / 2]])

        for i in range(qp_size - 1):
            Aeq[3 * i:3 * i + 6, 2 * i:2 * i + 2] = A_sub

        # 这里初始化不允许倒车约束，也就是 s(i) - s(i+1) <= 0
        A_forward = np.zeros((qp_size - 1, 3 * qp_size))
        # 上界
        A_bu = np.zeros((qp_size - 1, 1))
        # 下界
        A_bl = -np.inf * np.ones((qp_size - 1, 1))

        for i in range(qp_size - 1):
            A_forward[i, 3 * i] = 1
            A_forward[i, 3 * i + 3] = -1

        # 由于生成的凸空间约束s_lb s_ub不带起点，所以lb(i) = s_lb(i-1) 以此类推
        # 允许最小加速度为-6 最大加速度为4(基于车辆动力学), 这里和matlab不一样,这里没有上下界的说法，此处上下界应该要加到A里面
        for i in range(1, qp_size):
            lb[3 * i] = s_lb[i - 1]
            lb[3 * i + 1] = s_dot_lb[i - 1]
            lb[3 * i + 2] = -6
            ub[3 * i] = s_ub[i - 1]
            ub[3 * i + 1] = s_dot_ub[i - 1]
            ub[3 * i + 2] = 4

        # 起点约束
        lb[0] = 0
        lb[1] = plan_start_s_dot
        lb[2] = plan_start_s_dot2
        ub[0] = lb[0]
        ub[1] = lb[1]
        ub[2] = lb[2]

        # 把上下界添加到不等式约束中
        A_constraint = np.identity(3 * qp_size)
        A = sparse.csc_matrix(np.vstack((A_forward, A_constraint, Aeq.T)))
        u = np.vstack((A_bu, ub[:, None], beq))
        l = np.vstack((A_bl, lb[:, None], beq))

        # 加速度代价 jerk代价 以及推荐速度代价
        A_s_dot2 = np.zeros((3 * qp_size, 3 * qp_size))
        A_jerk = np.zeros((3 * qp_size, qp_size - 1))
        A_ref = np.zeros((3 * qp_size, 3 * qp_size))

        A4_sub = np.array([[0], [0], [1], [0], [0], [-1]])

        for i in range(1, qp_size + 1):
            A_s_dot2[3 * i - 1, 3 * i - 1] = 1
            A_ref[3 * i - 2, 3 * i - 2] = 1

        for i in range(1, qp_size):
            A_jerk[3 * i - 3:3 * i + 3, i - 1:i] = A4_sub

        # 生成H F
        P = sparse.csc_matrix((self.w_cost_s_dot2 * (A_s_dot2 @ A_s_dot2.T) + self.w_cost_jerk * (A_jerk @ A_jerk.T) + \
            self.w_cost_v_ref * (A_ref @ A_ref.T)) * 2)
        q = np.zeros((3 * qp_size, 1))
        for i in range(1, qp_size + 1):
            q[3 * i - 2] = -2 * self.w_cost_v_ref * self.reference_speed

        begin1 = time.time()
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, alpha=1.0, verbose=False)
        res = prob.solve()
        # print(time.time() - begin1)
        X = res.x
        status = res.info.status
        for i in range(1, qp_size + 1):
            qp_s_init[i - 1] = X[3 * i - 3]
            qp_s_dot_init[i - 1] = X[3 * i - 2]
            qp_s_dot2_init[i - 1] = X[3 * i - 1]
            relative_time_init[i - 1] = (i - 1) * dt

        return qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init, status

    def increase_points(self, s_init, s_dot_init, s_dot2_init, relative_time_init):
        """
        函数增密
        为什么需要增密，因为控制的执行频率是规划的10倍，轨迹点如果不够密，必然会导致规划效果不好
        但是若在速度二次规划中点取的太多，会导致二次规划的矩阵规模太大计算太慢
        所以方法是在二次规划中选取少量的点优化完毕后，在用此函数增密
        :param s_init:
        :param s_dot_init:
        :param s_dot2_init:
        :param relative_time_init:
        :return:增密的位置信息, 增密后的速度, 增密后的加速度, 增密的时间
        """
        time = relative_time_init[np.logical_not(np.isnan(relative_time_init))]
        # 增密
        # 为时间增密
        dense_time = np.arange(0, time[-1]+0.001, 0.025)
        # 统计dense_time边界, 找出dense_time在time中的位置
        bin_indices = (np.digitize(dense_time, time) - 1)
        # 求出增密时间与边界的距离
        x = (dense_time - time[bin_indices])[:-1]
        bin_indices = bin_indices[: -1]
        s = s_init[bin_indices] + s_dot_init[bin_indices] * x + s_dot2_init[bin_indices] * x ** 2 / 3 + \
            s_dot2_init[bin_indices+1] * x ** 2 / 6
        s_dot = s_dot_init[bin_indices] + 0.5 * s_dot2_init[bin_indices] * x + 0.5 * s_dot2_init[bin_indices + 1] * x
        s_dot2 = s_dot2_init[bin_indices] + (s_dot2_init[bin_indices + 1] - s_dot2_init[bin_indices]) * x / (
                relative_time_init[bin_indices + 1] - relative_time_init[bin_indices])
        relative_time = dense_time

        return np.append(s, s_init[-1]), np.append(s_dot, s_dot_init[-1]), np.append(s_dot2, s_dot2_init[-1]), relative_time


    def merge_path(self, s, path_s, traj_x_init, traj_y_init, traj_heading_init, s_dot, s_dot2, time):
        inter_x = interp1d(path_s, traj_x_init)
        inter_y = interp1d(path_s, traj_y_init)
        inter_h = interp1d(path_s, traj_heading_init)

        if s[-1] > path_s[-1]:
            s_inter = s[s < path_s[-1]]
        else:
            s_inter = s

        traj_x = inter_x(s_inter[1:])
        traj_y = inter_y(s_inter[1:])
        traj_h = inter_h(s_inter[1:])

        return np.hstack((traj_x_init[0], traj_x)), np.hstack((traj_y_init[0], traj_y)), \
                np.hstack((traj_heading_init[0], traj_h)), s, s_dot, s_dot2, time


def cal_heading_kappa(path_x:np.ndarray, path_y:np.ndarray):
    """
    计算frenet曲线上每个点的切向角theta（与直角坐标轴之间的角度）和曲率k
    param: frenet_path_xy_list: 包含frenet曲线上每一点的坐标[(x0,y0), (x1, y1), ...]
    return: list类型， theta = [theta0, theta1,...], k = [k0, k1, k2, ...]
    原理:
    theta = arctan(d_y/d_x)
    kappa = d_theta / d_s
    d_s = (d_x^2 + d_y^2)^0.5
    采用中点欧拉法来计算每个点处的斜率,当前点前一个线段斜率和后一个线段斜率求平均值

    注意，角度的多值性会带来很大的麻烦，需要慎重处理
    例，x 与 x + 2pi往往代表同一个角度，这种多值性在计算曲率会带来麻烦
    比如 原本是(0.1 - (-0.1))/ds，但如果计算出的-0.1多了一个2pi
    kappa就等于(0.1 - (-0.1 + 2*pi))/ds，曲率会变得非常大
    还有若采用中点欧拉法计算heading时，(即使用(y2 - y1)/(x2 - x1)的反正切计算一个角度 ，用(y1 - y0)/(x1-x0)的反正切又计算一个角度，
    然后加起来除以2,如果精确值是 (a1 + a2)/2,但最终计算的结果可能是(a1 + a2 + 2pi)/2
    还有 tan(x) = tan(x + pi) 所以arctan(tan(x))可能等于x，也可能等于x + pi
    角度的处理非常麻烦，而且角度处理不当往往是许多奇怪错误的源头
    """
    dx = np.diff(path_x)
    dy = np.diff(path_y)
    pre_dx = np.insert(dx, 0, dx[0])
    after_dx = np.insert(dx, -1, dx[-1])
    final_dx = (pre_dx + after_dx) / 2

    pre_dy = np.insert(dy, 0, dy[0])
    after_dy = np.insert(dy, -1, dy[-1])
    final_dy = (pre_dy + after_dy) / 2

    theta = np.arctan2(final_dy, final_dx)

    d_theta = np.diff(theta)  # 差分计算
    d_theta_pre = np.insert(d_theta, 0, d_theta[0])
    d_theta_aft = np.insert(d_theta, -1, d_theta[-1])
    d_theta = np.sin((d_theta_pre + d_theta_aft) / 2)  # 认为d_theta是个小量，用sin(d_theta)代替d_theta,避免多值性
    ds = np.sqrt(final_dx ** 2 + final_dy ** 2)
    k = d_theta / ds

    return theta, k

def find_match_point(x_set:np.ndarray,y_set,frenet_path_x,frenet_path_y,frenet_path_heading,frenet_path_kapp):
    """
    这个函数负责找到所有障碍物的匹配点,并计算投影距离,由于我的线长度小，可以这么计算
    :param x_set: 待求投影点集合的x坐标
    :param y_set: 待求投影点集合的y坐标
    :param frenet_path_x: 参考路径x坐标
    :param frenet_path_y: 参考路径y坐标
    :param frenet_path_heading: 参考路径航向角
    :param frenet_path_kapp: 参考路径曲率
    :return:
    """
    # assert(x_set.shape[1] == frenet_path_x.shape[0])
    # dis_x计算要求为一个列向量和行向量相减
    dis_x = x_set[:, np.newaxis] - frenet_path_x
    dis_y = y_set[:, np.newaxis] - frenet_path_y
    vectors = np.stack((dis_x, dis_y))
    # 求所有障碍物到参考路径的距离
    distance = np.linalg.norm(vectors, axis=0)
    # 求距离最小的编号
    min_index = np.argmin(distance, axis=1)
    # min_dis = distance[np.arange(0, distance.shape[0]), min_index]
    match_point_x = frenet_path_x[min_index]
    match_point_y = frenet_path_y[min_index]
    match_point_heading = frenet_path_heading[min_index]
    match_path_kapp = frenet_path_kapp[min_index]
    # 计算匹配点的方向向量与法向量
    # vector_match_point = np.stack((match_point_x, match_point_y))
    vector_match_point_direction = np.stack((np.cos(match_point_heading), np.sin(match_point_heading)))
    # 下面这句代码的作用就是取第0个维度的数据为dis_x.shape[0]里面每一个数据即每一行都要取,第1维为每个第0维数据里面编号为min_index
    vector_d_x = dis_x[np.arange(0, dis_x.shape[0]), min_index]
    vector_d_y = dis_y[np.arange(0, dis_y.shape[0]), min_index]
    vector_d = np.stack((vector_d_x, vector_d_y))
    # 这个向量的维度和vector_match_point_direction一致，计算时只要把x*cos(heading)+y*sin(heading)求出来
    ds = np.sum(vector_d * vector_match_point_direction, axis=0)
    proj_x_set = match_point_x + ds * vector_match_point_direction[0, :]
    proj_y_set = match_point_y + ds * vector_match_point_direction[1, :]
    proj_heading_set = match_point_heading + ds * match_path_kapp

    return proj_x_set, proj_y_set, proj_heading_set, match_path_kapp, min_index


def CalcSFromIndex2S(index2s,path_x,path_y,proj_x,proj_y,proj_match_point_index):
    """
    该函数将计算世界坐标系下的x_set，y_set上的点在frenet_path下的坐标s l
    :return:
    """
    # 该函数将计算当指定index2s的映射关系后，计算点proj_x, proj_y的弧长
    vector_1_x = proj_x - path_x[proj_match_point_index]
    vector_1_y = proj_y - path_y[proj_match_point_index]
    # 为了防止开for循环,先把proj_match_point_index的最大索引定位到path的倒数第二个索引
    proj_match_point_index = np.minimum(proj_match_point_index, len(path_x) - 2)
    # 统一使用大索引减小索引
    vector_2_x = path_x[proj_match_point_index + 1] - path_x[proj_match_point_index]
    vector_2_y = path_y[proj_match_point_index + 1] - path_y[proj_match_point_index]
    direction = np.sign(vector_1_x * vector_2_x + vector_1_y * vector_2_y)
    s = index2s[proj_match_point_index] + direction * np.sqrt(vector_1_x * vector_1_x + vector_1_y * vector_1_y)
    return s

def CalcSL(x_set,y_set,frenet_path_x,frenet_path_y, proj_x_set,proj_y_set,proj_heading_set,
           proj_match_point_index_set,index2s):
    s_set = CalcSFromIndex2S(index2s, frenet_path_x, frenet_path_y, proj_x_set, proj_y_set, proj_match_point_index_set)
    n_r_x = -np.sin(proj_heading_set)
    n_r_y = np.cos(proj_heading_set)
    delta_x = x_set - proj_x_set
    delta_y = y_set - proj_y_set
    l_set = n_r_x * delta_x + n_r_y * delta_y
    return s_set, l_set

def CalcDot_SL(l_set,vx_set,vy_set,proj_heading_set,proj_kappa_set):
    # 求切向向量
    t_r_x = np.cos(proj_heading_set)
    t_r_y = np.sin(proj_heading_set)
    # 求法向单位向量
    n_r_x = -np.sin(proj_heading_set)
    n_r_y = np.cos(proj_heading_set)
    l_dot_set = vx_set * n_r_x + vy_set * n_r_y
    s_dot_set = (vx_set * t_r_x + vy_set * t_r_y) / (1 - proj_kappa_set * l_set)
    return l_dot_set, s_dot_set

def traj_index_trans2s(trajectory_x,trajectory_y):
    """
    :param trajectory_x:
    :param trajectory_y:
    :return:
    """
    diff_path_x = np.diff(trajectory_x)
    diff_path_y = np.diff(trajectory_y)
    # 求路径上相邻的路径点之间的距离
    delta_s = np.sqrt(diff_path_x ** 2 + diff_path_y ** 2)
    # 求每个路径上点的长度
    index2s = np.insert(np.cumsum(delta_s), 0, 0.0)
    return index2s

def traj_index_trans2s_vec(trajectory_xs,trajectory_ys):
    """
    :param trajectory_xs:
    :param trajectory_ys:
    :return:
    """
    diff_path_x = np.diff(trajectory_xs)
    diff_path_y = np.diff(trajectory_ys)
    # 求路径上相邻的路径点之间的距离
    delta_s = np.sqrt(diff_path_x ** 2 + diff_path_y ** 2)
    # 求每个路径上点的长度
    index = np.cumsum(delta_s, axis=1)
    index2s = np.hstack((np.zeros((trajectory_xs.shape[0], 1)), index))
    return index2s

def get_b(pos_end, pos_start):
    x1 = pos_end[0]
    x2 = pos_start[0]
    y1 = pos_end[1]
    y2 = pos_start[1]
    b1 = (6 * (y1 - y2)) / (x1 - x2) ** 5
    b2 = -(15 * (x1 * y1 - x1 * y2 + x2 * y1 - x2 * y2)) / ((x1 - x2) ** 3 * (x1 ** 2 - 2 * x1 * x2 + x2 ** 2))
    b3 = (10 * (x1 ** 2 * y1 - x1 ** 2 * y2 + x2 ** 2 * y1 - x2 ** 2 * y2 + 4 * x1 * x2 * y1 - 4 * x1 * x2 * y2)) / (
                (x1 - x2) ** 3 * (x1 ** 2 - 2 * x1 * x2 + x2 ** 2))
    b4 = -(30 * x1 * (x2 ** 2 * y1 - x2 ** 2 * y2 + x1 * x2 * y1 - x1 * x2 * y2)) / (
                (x1 - x2) * (x1 ** 2 - 2 * x1 * x2 + x2 ** 2) ** 2)
    b5 = (30 * x1 * x2 ** 2 * (x1 * y1 - x1 * y2)) / ((x1 - x2) ** 3 * (x1 ** 2 - 2 * x1 * x2 + x2 ** 2))
    b6 = (y2 * x1 ** 5 - 5 * y2 * x1 ** 4 * x2 + 10 * y2 * x1 ** 3 * x2 ** 2 - 10 * y1 * x1 ** 2 * x2 ** 3 +
          5 * y1 * x1 * x2 ** 4 - y1 * x2 ** 5) / ((x1 - x2) ** 2 * (x1 ** 3 - 3 * x1 ** 2 * x2 + 3 * x1 * x2 ** 2 - x2 ** 3))
    return b1,b2,b3,b4,b5,b6

def generate_st_graph(dynamic_obs_s_set, dynamic_obs_l_set, dynamic_obs_s_dot_set, dynamic_obs_l_dot_set):
    n = len(dynamic_obs_s_set)
    obs_st_s_in_set = np.ones(n) * np.nan
    obs_st_s_out_set = np.ones(n) * np.nan
    obs_st_t_in_set = np.ones(n) * np.nan
    obs_st_t_out_set = np.ones(n) * np.nan
    for i in range(len(dynamic_obs_s_set)):
        if abs(dynamic_obs_l_set[i]) > 8 and abs(dynamic_obs_l_dot_set[i]) < 2:  # 侧向缓慢移动的障碍物
            # if abs(dynamic_obs_l_set[i]) > 5:  # 距离横向太远，速度规划直接忽略
            continue
        # t_zero 为动态障碍物的l到0，所需要的时间
        t_zero = - dynamic_obs_l_set[i] / (dynamic_obs_l_dot_set[i]+1e-10)  # 时间等于路程除以速度
        # 计算+-2缓冲时间
        t_boundary1 = 2 / (dynamic_obs_l_dot_set[i]+1e-10) + t_zero
        t_boundary2 = -2 / (dynamic_obs_l_dot_set[i]+1e-10) + t_zero
        if t_boundary1 > t_boundary2:
            t_max = t_boundary1
            t_min = t_boundary2
        else:
            t_max = t_boundary2
            t_min = t_boundary1
        if t_max < 0.1 or t_min > 2.5:
            # 对于切入切出太远的，或者碰瓷的，忽略
            # 车辆运动是要受到车辆动力学制约的，如果有碰瓷的，即使规划出了很大的加速度，车辆也执行不了
            # 碰瓷障碍物也需要做虚拟障碍物和路径规划一起解决
            continue
        if t_min < 0 and t_max > 0:
            # 在感知看到的时候，障碍物已经在+-2的内部了
            obs_st_s_in_set[i] = dynamic_obs_s_set[i]
            # 匀速运动
            obs_st_s_out_set[i] = dynamic_obs_s_set[i] + dynamic_obs_s_dot_set[i] * t_max
            obs_st_t_in_set[i] = 0
            obs_st_t_out_set[i] = t_max
        else:
            # 正常障碍物
            obs_st_s_in_set[i] = dynamic_obs_s_set[i] + dynamic_obs_s_dot_set[i] * t_min
            obs_st_s_out_set[i] = dynamic_obs_s_set[i] + dynamic_obs_s_dot_set[i] * t_max
            obs_st_t_in_set[i] = t_min
            obs_st_t_out_set[i] = t_max

    return obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set

def calc_speed_planning_start_condition(plan_start_vx, plan_start_vy, plan_start_ax, plan_start_ay, plan_start_heading):
    """
    该函数计算速度规划的初始条件
    params: 都为cartesian坐标系下的信息
    """
    tor = np.array([[np.cos(plan_start_heading)], [np.sin(plan_start_heading)]])
    # 计算向量 v 在切向的投影
    v_t = np.dot(tor.T, np.array([[plan_start_vx], [plan_start_vy]]))
    a_t = np.dot(tor.T, np.array([[plan_start_ax], [plan_start_ay]]))
    # 计算s方向上的加速度
    plan_start_s_dot = v_t
    plan_start_s_dot2 = a_t
    return plan_start_s_dot, plan_start_s_dot2

def virtual_dynamic_obs_planning(x_set, y_set, vx_set, vy_set,
                                 index2s_vec, paths_x, paths_y,
                                 plan_start_v, time_step=np.linspace(0, 2, 11)):
    """
    规划路径绕开虚拟障碍物,只针对动态障碍物
    :return:
    """
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
    # bv_x_predict是NxT(N是障碍物个数，T是时间采样个数),av_x_predict是MxT(M是备选路径条数，T是采样个数)
    # 应该使用每个时刻中每个障碍物与自车的距离，那么这里就要保持T维度不变对其他维度进行升高维度操作才能满足相减条件
    # bv_x_predict ---> 1xNxT  av_x_predict ---> Mx1xT
    # 计算得到的delta_dis是MxNxT, delta_dis[0,:,:]的物理意义是第0条备选路径在T个时间采样点中自车与N个bv之间的距离的平方
    # delta_dis = dy ** 2 + dx ** 2
    # 这里再加一项，如果两车之间查找两车之间的距离小于3的所在的路径，有这样的路径，碰撞概率非常大，另外加一个cost
    # collision_index = np.logical_or(np.any(delta_dis < 9, axis=(1, 2)), collision_index)
    # 钟形曲线做压缩
    # single_cost = np.exp(-delta_dis/100)

    # 衰减系数，越远的距离越小，因为预测越不准确而且还有一个问题就是越远的障碍物，我越能够通过速度规划来做最后的保证
    factor_cost = single_cost * factor[np.newaxis, np.newaxis, :]
    cost = np.sum(factor_cost, axis=(1, 2)) * 1000 + 100 * collision_index
    # path_index = np.argmin(cost)
    return cost
