import matplotlib.pyplot as plt
from onsite import scenarioOrganizer, env
import time
import sys
sys.path.append('.')
from Lattice_Planner.Lattice_Planner_2 import Lattice_Planner_2 as LP2
from Lattice_Planner.a_star import global_path
import os
from Lattice_Planner import backcar
import pandas as pd
import numpy as np
import concurrent.futures

def get_lane_info1(road_info):
    map_info = {}
    map_info[0] = {"left_bound": road_info.discretelanes[0].left_vertices[0, 1],
                   "center": road_info.discretelanes[0].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[0].right_vertices[0, 1]}
    map_info[1] = {"left_bound": road_info.discretelanes[1].left_vertices[0, 1],
                   "center": road_info.discretelanes[1].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[1].right_vertices[0, 1]}
    map_info[2] = {"left_bound": road_info.discretelanes[2].left_vertices[0, 1],
                   "center": road_info.discretelanes[2].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[2].right_vertices[0, 1]}
    return map_info
def get_lane_info2(road_info):
    map_info = {}
    map_info[0] = {"left_bound": road_info.discretelanes[3].left_vertices[0, 1],
                   "center": road_info.discretelanes[3].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[3].right_vertices[0, 1]}
    map_info[1] = {"left_bound": road_info.discretelanes[4].left_vertices[0, 1],
                   "center": road_info.discretelanes[4].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[4].right_vertices[0, 1]}
    map_info[2] = {"left_bound": road_info.discretelanes[5].left_vertices[0, 1],
                   "center": road_info.discretelanes[5].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[5].right_vertices[0, 1]}
    return map_info

def process_input_directory(input_dir, output_dir, output_dir_new):
    so = scenarioOrganizer.ScenarioOrganizer()
    envi = env.Env()
    so.load(input_dir, output_dir)

    # so.scenario_list = so.scenario_list[0:500]
    # so.scenario_list = so.scenario_list[500:2000]
    # so.scenario_list = so.scenario_list[2000:2500]
    # so.scenario_list = so.scenario_list[2500:3500]

    while True:
        scenario_to_test = so.next()
        if scenario_to_test is None:
            break
        scenario_to_test['test_settings']['visualize'] = False
        # scenario_to_test['test_settings']['visualize'] = True
        print("测试:", scenario_to_test)
        observation, traj = envi.make(scenario=scenario_to_test)

        # 对trajectories进行处理，将其转换为numpy数组
        traj_np = {}
        for vehicle_id, vehicle_traj in traj.items():
            x_coords = []
            y_coords = []
            numeric_keys = sorted(
                [k for k in vehicle_traj.keys() if isinstance(k, str) and k.replace('.', '', 1).isdigit()], key=float)
            # Iterate over each time step in the vehicle's trajectory
            for t in numeric_keys:
                x_coords.append(float(vehicle_traj[t]['x']))
                y_coords.append(float(vehicle_traj[t]['y']))
            traj_np[vehicle_id] = np.array([x_coords, y_coords])
        # 根据每个车辆最后时刻的速度，在traj_np的基础上再添加3s的轨迹，y不变，x根据速度变化
        for vehicle_id, coords in traj_np.items():
            # 获取最后一个时间步的速度
            if len(coords[0]) > 1:
                last_x = coords[0][-1]
                second_last_x = coords[0][-2]
                speed_x = (last_x - second_last_x) / 0.04  # 假设时间步长为0.04秒
                # 添加3秒的轨迹
                for _ in range(int(3 / 0.04)):
                    new_x = traj_np[vehicle_id][0][-1] + speed_x * 0.04
                    traj_np[vehicle_id] = np.vstack((np.append(traj_np[vehicle_id][0], new_x),
                                                     np.append(traj_np[vehicle_id][1], traj_np[vehicle_id][1][-1])))

        road_info = envi.controller.observation.road_info

        init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']]
        lane_num = len(road_info.discretelanes)
        if lane_num != 6:
            continue

        goal = [np.mean(observation['test_setting']['goal']['x']), np.mean(observation['test_setting']['goal']['y'])]
        if goal[0] > init_pos[0]:
            road_info_s = get_lane_info1(road_info)
        else:
            road_info_s = get_lane_info2(road_info)

        yaw0 = observation['vehicle_info']['ego']['yaw']
        steer_ego = [0]
        goal_x = [np.mean(observation['test_setting']['goal']['x'])]
        goal_y = [np.mean(observation['test_setting']['goal']['y'])]
        lane1 = [road_info_s[0]['left_bound']]
        lane2 = [road_info_s[0]['right_bound']]
        lane3 = [road_info_s[1]['right_bound']]
        lane4 = [road_info_s[2]['right_bound']]

        # # guass.a从均值为0.4，标准差为0.4*0.2=0.08的正态分布中采样
        # # guass.b从均值为35，标准差为35*0.2=7的正态分布中采样
        # # guass.c固定为1
        # # guass.d从均值为100，标准差为100*0.2=20的正态分布中采样
        # guass = {'a': np.random.normal(0.4, 0.4), 'b': np.random.normal(35, 35), 'c': 1, 'd': np.random.normal(100, 100)}
        # # print(guass)
        try:
            gp = global_path(observation, scenario=scenario_to_test)
            waypoints, ky = gp.global_path_planning(road_info, yaw0)
            planner = LP2(observation, traj, waypoints, scenario=scenario_to_test)
            # planner = LP2(observation, traj, waypoints, scenario=scenario_to_test, guass=guass)

            frame = 0
            exit_loop = False
            while observation['test_setting']['end'] == -1 and not exit_loop:
                action = planner.exec_waypoint_nav_demo(observation, traj_np, road_info_s, 0)
                action_new = backcar.avoid_rear_colission(observation, action[1], ky, yaw0)
                if action_new[0] != 0:
                    action[0] = action_new[0]
                if action_new[1] != 999:
                    action[1] = action_new[1]

                observation = envi.step(action)
                frame += 1

                steer_ego.append(action[1])
                goal_x.append(np.mean(observation['test_setting']['goal']['x']))
                goal_y.append(np.mean(observation['test_setting']['goal']['y']))
                lane1.append(road_info_s[0]['left_bound'])
                lane2.append(road_info_s[0]['right_bound'])
                lane3.append(road_info_s[1]['right_bound'])
                lane4.append(road_info_s[2]['right_bound'])

                if goal[0] > init_pos[0] and observation['vehicle_info']['ego']['x'] - goal[0] > 25:
                    exit_loop = True
                if goal[0] < init_pos[0] and goal[0] - observation['vehicle_info']['ego']['x'] > 25:
                    exit_loop = True

        except Exception as e:
            print(repr(e))

        finally:
            so.add_result(scenario_to_test, observation['test_setting']['end'])
            plt.close()

            # 等待1s
            time.sleep(1)

            file_name = scenario_to_test['data']['scene_name'] + '_result.csv'
            file_path = os.path.join(output_dir, file_name)
            df = pd.read_csv(file_path)
            steer_ego = pd.Series(steer_ego, name='steer_ego')
            goal_x = pd.Series(goal_x, name='goal_x')
            goal_y = pd.Series(goal_y, name='goal_y')
            lane1 = pd.Series(lane1, name='lane1')
            lane2 = pd.Series(lane2, name='lane2')
            lane3 = pd.Series(lane3, name='lane3')
            lane4 = pd.Series(lane4, name='lane4')
            num_columns = len(df.columns)
            df.insert(num_columns, 'steer_ego', steer_ego)
            df.insert(num_columns + 1, 'goal_x', goal_x)
            df.insert(num_columns + 2, 'goal_y', goal_y)
            df.insert(num_columns + 3, 'lane1', lane1)
            df.insert(num_columns + 4, 'lane2', lane2)
            df.insert(num_columns + 5, 'lane3', lane3)
            df.insert(num_columns + 6, 'lane4', lane4)

            df.rename(columns={df.columns[0]: None}, inplace=True)
            file_path_new = os.path.join(output_dir_new, file_name)
            df.to_csv(file_path_new, index=False)

            # 等待1s
            time.sleep(1)

if __name__ == "__main__":

    input_dirs = [f"E:\\python_program\\Onsite\\planner\\inputs\\inputs_all_multi\\inputs{i}" for i in range(5)]
    # input_dirs = [f"D:\\C\\linz\\BS\\onsite\\inputs_all\\inputs_only_in_more\\inputs{i}" for i in range(10)]
    # input_dirs = [f"D:\\C\\linz\\BS\\onsite\\inputs_all\\inputs_75_2\\inputs{i}" for i in range(10)]
    # input_dirs = [f"D:\\C\\linz\\BS\\onsite\\inputs_all\\inputs_only_in\\inputs{i}" for i in range(10)]

    # output_dir = r"D:\C\linz\BS\onsite\outputs_only_in_more_guass"
    # output_dir_new = r"D:\C\linz\BS\onsite\outputs_only_in_more_guass_new"
    output_dir = r"D:\C\linz\BS\onsite\outputs_no_fanhua"
    output_dir_new = r"D:\C\linz\BS\onsite\outputs_no_fanhua_new"

    # 使用进程池并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_input_directory, input_dir, output_dir, output_dir_new) for input_dir in input_dirs]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"进程执行时发生异常: {e}")

