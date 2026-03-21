import numpy as np
import time
import pandas as pd
from datetime import datetime
import os
import importlib


def dynamic_programming(prev_timestamp, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                        w_cost_ref_speed, reference_speed, w_cost_accel, w_cost_obs, plan_start_s_dot,
                        s_list, t_list, save_path=r"E:\onsite3-main", save_format="auto",
                        file_name="dp_performance_data"):
    s_list = np.concatenate([
        np.arange(0, 3, 0.2),  # [0:0.2:3]
        np.arange(4, 8, 1),  # [4:1:8]
        np.arange(9, 19.5, 1.5)  # [9:1.5:16.5]
    ])

    t_list = np.arange(0, 1.7, 0.1)
    # s_list = np.concatenate([np.arange(0, 5, 0.5), np.arange(5.5, 15, 1), np.arange(16, 30, 1.5), np.arange(32, 55, 2.5)])
    # t_list = np.arange(0.5, 8.5, 0.5)
    m = len(s_list)
    n = len(t_list)

    # 初始化矩阵
    dp_st_cost = np.ones((m, n)) * np.inf
    dp_st_s_dot = np.zeros((m, n))
    dp_st_node = np.zeros((m, n))

    # 缓存计算结果
    coordinate_cache = {}  # 添加坐标缓存

    # 计算最大可能的加速度和减速度
    max_accel = 5.0  # 根据车辆动力学特性设置
    max_decel = -5.0

    # 修改calc_st_coordinate调用，使用坐标缓存
    def cached_st_coordinate(row, col):
        """使用缓存的坐标计算函数"""
        cache_key = (row, col)
        if cache_key in coordinate_cache:
            return coordinate_cache[cache_key]

        s_value, t_value = calc_st_coordinate(row, col, s_list, t_list)
        coordinate_cache[cache_key] = (s_value, t_value)
        return s_value, t_value

    # 在get_valid_predecessors函数中也使用缓存的坐标计算
    def get_valid_predecessors(cur_row, cur_col):
        """仅使用动力学约束获取有效的前置节点"""
        if cur_col == 0:
            return [cur_row]

        # 考虑所有可能的前置节点
        potential_rows = list(range(m))

        cur_s, cur_t = cached_st_coordinate(cur_row, cur_col)
        valid_rows = []

        # 对所有潜在前置节点应用动力学约束
        for k in potential_rows:
            pre_s, pre_t = cached_st_coordinate(k, cur_col - 1)
            v_i = dp_st_s_dot[k, cur_col - 1]  # 获取前一时刻的速度
            dt = cur_t - pre_t

            if dt <= 0.0001:  # 防止除以零或极小值
                continue

            # 动力学公式计算可行范围
            min_s = pre_s + v_i * dt + 0.5 * max_decel * dt ** 2
            max_s = pre_s + v_i * dt + 0.5 * max_accel * dt ** 2

            # 放宽约束，增加容错度
            min_s -= 0.5  # 添加0.5米的容错
            max_s += 0.5

            if min_s <= cur_s <= max_s:
                valid_rows.append(k)

        return valid_rows

    # 初始化第一列
    for i in range(m):
        dp_st_cost[i, 0] = calc_dp_cost(0, 0, i, 1, obs_st_s_in_set, obs_st_s_out_set,
                                        obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed,
                                        reference_speed, w_cost_accel, w_cost_obs,
                                        plan_start_s_dot, s_list, t_list, dp_st_s_dot, prev_timestamp)
        s_end, t_end = cached_st_coordinate(i, 1)
        if t_end > 0.0001:  # 防止除以零
            dp_st_s_dot[i, 0] = s_end / t_end
        else:
            dp_st_s_dot[i, 0] = 0

    # 动态规划主循环
    for i in range(1, n):
        for j in range(m):
            cur_row = j
            cur_col = i

            # 获取有效的前置节点
            valid_predecessors = get_valid_predecessors(cur_row, cur_col)

            if not valid_predecessors:
                continue  # 如果没有有效前置节点，跳过当前节点

            # 记录有多少节点被评估
            evaluated_count = 0

            for pre_row in valid_predecessors:
                pre_col = i - 1
                cost_temp = calc_dp_cost(pre_row, pre_col, cur_row, cur_col,
                                         obs_st_s_in_set, obs_st_s_out_set,
                                         obs_st_t_in_set, obs_st_t_out_set,
                                         w_cost_ref_speed, reference_speed,
                                         w_cost_accel, w_cost_obs, plan_start_s_dot,
                                         s_list, t_list, dp_st_s_dot, prev_timestamp)

                # 记录已评估节点数
                evaluated_count += 1

                if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                    dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                    s_start, t_start = cached_st_coordinate(pre_row, pre_col)
                    s_end, t_end = cached_st_coordinate(cur_row, cur_col)
                    dt = t_end - t_start
                    if dt > 0.0001:  # 防止除以零
                        dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / dt
                    else:
                        dp_st_s_dot[cur_row, cur_col] = dp_st_s_dot[pre_row, pre_col]  # 保持前一时刻的速度
                    dp_st_node[cur_row, cur_col] = pre_row

    # 回溯最优路径
    dp_speed_s = np.ones(len(t_list)) * np.nan
    dp_speed_t = np.copy(dp_speed_s)

    # 找到终点
    min_cost = np.inf
    min_row = -1  # 使用 -1 初始化，以便区分未找到的情况
    min_col = -1

    # 检查最后一列
    for i in range(m):
        if dp_st_cost[i, -1] < min_cost:  # 严格使用 <
            min_cost = dp_st_cost[i, -1]
            min_row = i
            min_col = len(t_list) - 1


    for j in range(n):
        if dp_st_cost[0, j] < min_cost: # 严格使用 <
            min_cost = dp_st_cost[0, j]
            min_row = 0
            min_col = j

    # 检查是否找到有效终点 (必须在边界上找到有限代价的点)
    if min_row == -1 or np.isinf(min_cost):
        # 如果在边界上没有找到有效终点，返回 NaN 路径
        print("Warning: No valid path found reaching the boundary.")
        dp_speed_s = np.full(len(t_list), np.nan) # 使用 full 创建 NaN 数组
        dp_speed_t = t_list.copy()
        return dp_speed_s, dp_speed_t

    # 回溯路径 (如果找到了有效终点)
    dp_speed_s[min_col], dp_speed_t[min_col] = cached_st_coordinate(min_row, min_col)
    current_min_row = min_row
    current_min_col = min_col
    while current_min_col > 0:  # 使用循环变量而不是min_col
        pre_row = int(dp_st_node[int(current_min_row), int(current_min_col)])
        pre_col = current_min_col - 1
        dp_speed_s[pre_col], dp_speed_t[pre_col] = cached_st_coordinate(pre_row, pre_col)
        current_min_row = pre_row
        current_min_col = pre_col

    # 确保dp_speed_t与t_list一致
    dp_speed_t = t_list.copy()

    # print(dp_speed_s)

    return dp_speed_s, dp_speed_t


def calc_st_coordinate(row, col, s_list, t_list):
    # 计算矩阵节点的 s t 坐标
    m = len(s_list)
    s_value = s_list[int(m - row - 1)]
    t_value = t_list[col]
    return s_value, t_value


def calc_collision_cost(w_cost_obs, min_dis):
    if abs(min_dis) < 0.5:
        collision_cost = w_cost_obs
    # elif 0.5 < abs(min_dis) < 1.5:
    elif 0.5 < abs(min_dis) < 1.5:
        collision_cost = w_cost_obs ** ((0.5 - abs(min_dis)) + 1)
    else:
        collision_cost = 0
    return collision_cost


def calc_obs_cost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                  w_cost_obs):
    # 计算边的障碍物代价
    obs_cost = 0
    n = 5  # 采样点的个数

    # 防止除以零
    if abs(t_end - t_start) < 0.0001:
        return 0

    dt = (t_end - t_start) / (n - 1)  # 采样时间间隔
    k = (s_end - s_start) / (t_end - t_start)  # 边的斜率

    # 预先计算采样点，避免在循环中重复计算
    t_samples = [t_start + i * dt for i in range(n)]
    s_samples = [s_start + k * i * dt for i in range(n)]

    # 快速过滤掉不相关的障碍物
    valid_obstacles = []
    t_min = min(t_start, t_end)
    t_max = max(t_start, t_end)
    s_min = min(s_start, s_end)
    s_max = max(s_start, s_end)

    # 扩展搜索范围，确保不会遗漏潜在的碰撞
    margin = 2.0  # 安全边界
    s_min -= margin
    s_max += margin

    for j in range(len(obs_st_s_in_set)):
        if np.isnan(obs_st_s_in_set[j]).any():
            continue

        # 快速边界框检查
        obs_s_min = min(obs_st_s_in_set[j], obs_st_s_out_set[j])
        obs_s_max = max(obs_st_s_in_set[j], obs_st_s_out_set[j])
        obs_t_min = min(obs_st_t_in_set[j], obs_st_t_out_set[j])
        obs_t_max = max(obs_st_t_in_set[j], obs_st_t_out_set[j])

        # 如果障碍物的边界框与路径的边界框没有重叠，跳过这个障碍物
        if (obs_s_max < s_min or obs_s_min > s_max or
                obs_t_max < t_min or obs_t_min > t_max):
            continue

        valid_obstacles.append(j)

    # 如果没有有效障碍物，直接返回0
    if not valid_obstacles:
        return 0

    # 对每个采样点计算代价
    for i in range(n):
        t = t_samples[i]
        s = s_samples[i]

        for j in valid_obstacles:
            # 获取障碍物坐标
            obs_s_in = obs_st_s_in_set[j]
            obs_t_in = obs_st_t_in_set[j]
            obs_s_out = obs_st_s_out_set[j]
            obs_t_out = obs_st_t_out_set[j]

            # 计算向量
            v1_s = obs_s_in - s
            v1_t = obs_t_in - t
            v2_s = obs_s_out - s
            v2_t = obs_t_out - t

            # 计算向量3
            v3_s = v2_s - v1_s
            v3_t = v2_t - v1_t

            # 计算距离
            v1_dot_v1 = v1_s * v1_s + v1_t * v1_t
            v2_dot_v2 = v2_s * v2_s + v2_t * v2_t
            dis1 = np.sqrt(v1_dot_v1)
            dis2 = np.sqrt(v2_dot_v2)

            # 计算向量3的模和点积
            v3_dot_v3 = v3_s * v3_s + v3_t * v3_t

            # 防止除以零
            if v3_dot_v3 < 1e-10:
                min_dis = min(dis1, dis2)
            else:
                # 优化点积计算
                dot_v1_v3 = v1_s * v3_s + v1_t * v3_t
                dot_v2_v3 = v2_s * v3_s + v2_t * v3_t

                # 根据点积判断使用哪个距离
                if (dot_v1_v3 > 0 and dot_v2_v3 > 0) or (dot_v1_v3 < 0 and dot_v2_v3 < 0):
                    min_dis = min(dis1, dis2)
                else:
                    cross_product = v1_s * v3_t - v1_t * v3_s
                    min_dis = abs(cross_product) / np.sqrt(v3_dot_v3)

            # 如果距离已经大于1.5，跳过碰撞代价计算
            if min_dis >= 1.5:
                continue

            # 计算碰撞代价
            if min_dis < 0.5:
                obs_cost += w_cost_obs
            elif min_dis < 1.5:
                obs_cost += w_cost_obs ** ((0.5 - min_dis) + 1)

    return obs_cost


def calc_dp_cost(row_start, col_start, row_end, col_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                 obs_st_t_out_set,
                 w_cost_ref_speed, reference_speed, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list,
                 dp_st_s_dot, prev_timestamp):
    # 计算链接两个节点之间边的代价
    s_end, t_end = calc_st_coordinate(row_end, col_end, s_list, t_list)
    # 规定起点的行列号为0
    if row_start == 0:
        s_start, t_start = 0, 0
        s_dot_start = plan_start_s_dot
    else:
        s_start, t_start = calc_st_coordinate(row_start, col_start, s_list, t_list)
        s_dot_start = dp_st_s_dot[row_start][col_start]

    # 防止除以零
    dt = t_end - t_start
    if abs(dt) < 0.0001:
        return np.inf

    cur_s_dot = (s_end - s_start) / dt
    cur_s_dot2 = (cur_s_dot - s_dot_start) / dt

    # 计算推荐速度代价
    cost_ref_speed = w_cost_ref_speed * (cur_s_dot - reference_speed) ** 2

    # 计算加速度代价，这里注意，加速度不能超过车辆动力学上下限
    if -4 <= cur_s_dot2 <= 5.0:  # 修正加速度约束范围
        # 修改加速度代价计算方式，当速度低于参考速度且加速度为正时，降低加速度代价
        if cur_s_dot < reference_speed and cur_s_dot2 > 0:
            # 当速度比参考速度低，且正在加速时，减小代价
            acceleration_factor = max(0.1, 1.0 - (reference_speed - cur_s_dot) / reference_speed * 0.8)
            cost_accel = w_cost_accel * cur_s_dot2 ** 2 * acceleration_factor
        elif cur_s_dot > reference_speed and cur_s_dot2 < 0:
            # 当速度高于参考速度且减速时，同样减小代价
            deceleration_factor = max(0.1, 1.0 - (cur_s_dot - reference_speed) / reference_speed * 0.8)
            cost_accel = w_cost_accel * cur_s_dot2 ** 2 * deceleration_factor
        else:
            # 其他情况正常计算代价
            cost_accel = w_cost_accel * cur_s_dot2 ** 2
    else:
        # 超过车辆动力学限制，代价会增大很多倍
        cost_accel = 100000 * w_cost_accel * cur_s_dot2 ** 2

    cost_obs = calc_obs_cost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                             obs_st_t_out_set, w_cost_obs)

    cost = cost_obs + cost_accel + cost_ref_speed

    return cost





