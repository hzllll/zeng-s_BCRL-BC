import matplotlib.pyplot as plt
from onsite import scenarioOrganizer, env
import time
import sys
sys.path.append('.')
import os
import numpy as np
import re # 正则表达式，用于精准匹配和修改文本文件中的特定字段
import shutil

def get_lane_info1(road_info):
    # 从当前道路环境（road_info）中提取离散化车道（discretelanes）的几何边界信息
    map_info = {}
    map_info[0] = {"left_bound":road_info.discretelanes[0].left_vertices[0, 1],
                   "center":road_info.discretelanes[0].center_vertices[0, 1],
                   "right_bound":road_info.discretelanes[0].right_vertices[0, 1]}
    map_info[1] = {"left_bound": road_info.discretelanes[1].left_vertices[0, 1],
                   "center": road_info.discretelanes[1].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[1].right_vertices[0, 1]}
    map_info[2] = {"left_bound": road_info.discretelanes[2].left_vertices[0, 1],
                   "center": road_info.discretelanes[2].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[2].right_vertices[0, 1]}
    return map_info
    # 返回的数据结构是一个字典，
    # 包含每条车道的 left_bound（左边界）、center（中心线）、right_bound（右边界）的纵坐标（y值）
    
def get_lane_info2(road_info):
    map_info = {}
    map_info[0] = {"left_bound":road_info.discretelanes[3].left_vertices[0, 1],
                   "center":road_info.discretelanes[3].center_vertices[0, 1],
                   "right_bound":road_info.discretelanes[3].right_vertices[0, 1]}
    map_info[1] = {"left_bound": road_info.discretelanes[4].left_vertices[0, 1],
                   "center": road_info.discretelanes[4].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[4].right_vertices[0, 1]}
    map_info[2] = {"left_bound": road_info.discretelanes[5].left_vertices[0, 1],
                   "center": road_info.discretelanes[5].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[5].right_vertices[0, 1]}
    return map_info

for i in range(0, 10):
    # 遍历多个文件夹路径 D:\C\linz\BS\onsite\inputs0901\inputs0 ~ D:\C\linz\BS\onsite\inputs0901\inputs9
    input_dir = r"D:\C\linz\BS\onsite\inputs_only_in_more\inputs" + str(i)
    output_dir = r"D:\C\linz\BS\onsite\outputs1021\outputs" + str(i)

    so = scenarioOrganizer.ScenarioOrganizer()
    envi = env.Env()
    so.load(input_dir, output_dir)
    a = 0.2
    # a = 0.1
    while True:
        scenario_to_test = so.next()
        if scenario_to_test is None:
            break
        scenario_to_test['test_settings']['visualize'] = False
        observation, traj = envi.make(scenario=scenario_to_test)
        road_info = envi.controller.observation.road_info
        goal = [np.mean(observation['test_setting']['goal']['x']), np.mean(observation['test_setting']['goal']['y'])] # 目标区域
        init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']] # 获取原始场景中主车的初始位置
        yaw0 = observation['vehicle_info']['ego']['yaw'] # 初始航向角
        v0 = observation['vehicle_info']['ego']['v'] # 初始速度
        # s = 1/5 * v0
        s = 5
        lane_num = len(road_info.discretelanes)
        # 距离目标点的纵向距离
        long_dis = np.abs(goal[0] - init_pos[0]) 
        # x1 = init_pos[0] - long_dis / 4
        # x2 = init_pos[0] - long_dis / 12
        # x3 = init_pos[0]
        # x4 = init_pos[0] + long_dis / 12
        # x5 = init_pos[0] + long_dis / 4
        if lane_num != 6:
            continue
        if goal[0] > init_pos[0]: 
            # 判断方向：从左往右
            map_info = get_lane_info1(road_info)
            # 三个车道的目标区域生成：将3条车道的左右边界作为3个不同的目标区域（goal_y1, goal_y2, goal_y3）
            goal_y1 = np.array([map_info[0]['left_bound'], map_info[0]['right_bound']])
            goal_y2 = np.array([map_info[1]['left_bound'], map_info[1]['right_bound']])
            goal_y3 = np.array([map_info[2]['left_bound'], map_info[2]['right_bound']])
            # 设置5个纵向离散点
            x1 = init_pos[0] - long_dis / 3 #向后退 1/3 距离
            x2 = init_pos[0] - long_dis / 6 #向后退 1/6 距离
            x3 = init_pos[0] # (原位置)
            x4 = init_pos[0] + long_dis / 12 #向前进 1/12 距离
            x5 = init_pos[0] + long_dis / 4 #向前进 1/4 距离
            # 速度从-s~s随机均匀分布
            # 车道内的航向角-a~a随机均匀分布,车道外的-2a~2a随机均匀分布
            # x的位置为x1、x2、x3、x4、x5
            # y的位置为map_info[0]['center']、map_info[1]['center']、map_info[2]['center']
            # 以及map_info[2]['right_bound'] - dy(dy为0~5的随机数)、map_info[0]['left_bound'] + dy
            # x,y共组合为5*5=25种情况
            init_pos_ = np.array([[x1, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x2, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x3, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x4, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x5, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x1, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x2, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x3, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x4, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x5, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x1, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x2, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x3, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x4, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        [x5, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.random.uniform(-a, a)],
                        # [x1, map_info[2]['right_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x2, map_info[2]['right_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x3, map_info[2]['right_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x4, map_info[2]['right_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x5, map_info[2]['right_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x1, map_info[0]['left_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x2, map_info[0]['left_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x3, map_info[0]['left_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x4, map_info[0]['left_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)],
                        # [x5, map_info[0]['left_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.random.uniform(-2*a, 2*a)]
                                  ])
            # init_pos与goal_y共组合为25*3=75种情况，在init_pos每一行后面分别拼接goal_y1、goal_y2、goal_y3组成新的数组
            init_pos_ = np.array([np.hstack((i, j)) for i in init_pos_ for j in [goal_y1, goal_y2, goal_y3]])
            # 随机打乱init_pos的顺序
            np.random.shuffle(init_pos_)
        else:
            # 从右往左
            map_info = get_lane_info2(road_info)
            # 三个车道的目标区域
            goal_y1 = np.array([map_info[0]['left_bound'], map_info[0]['right_bound']])
            goal_y2 = np.array([map_info[1]['left_bound'], map_info[1]['right_bound']])
            goal_y3 = np.array([map_info[2]['left_bound'], map_info[2]['right_bound']])
            x1 = init_pos[0] - long_dis / 4
            x2 = init_pos[0] - long_dis / 12
            x3 = init_pos[0]
            x4 = init_pos[0] + long_dis / 6
            x5 = init_pos[0] + long_dis / 3
            # 速度从-s~s随机均匀分布
            # 车道内的航向角-a~a随机均匀分布,车道外的-2a~2a随机均匀分布
            # x的位置为x1、x2、x3、x4、x5
            # y的位置为map_info[0]['center']、map_info[1]['center']、map_info[2]['center']
            # 以及map_info[2]['right_bound'] - dy(dy为0~5的随机数)、map_info[0]['left_bound'] + dy
            # x,y共组合为5*5=25种情况
            init_pos_ = np.array([[x1, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x2, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x3, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x4, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x5, map_info[0]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x1, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x2, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x3, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x4, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x5, map_info[1]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x1, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x2, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x3, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x4, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        [x5, map_info[2]['center'] + np.random.uniform(-1.5, 1.5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-a, a)],
                        # [x1, map_info[2]['right_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x2, map_info[2]['right_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x3, map_info[2]['right_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x4, map_info[2]['right_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x5, map_info[2]['right_bound'] + np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x1, map_info[0]['left_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x2, map_info[0]['left_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x3, map_info[0]['left_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x4, map_info[0]['left_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)],
                        # [x5, map_info[0]['left_bound'] - np.random.uniform(0, 5), v0 + np.random.uniform(-s, s), np.pi + np.random.uniform(-2*a, 2*a)]
                                  ])
            # init_pos与goal_y共组合为25*3=75种情况，在init_pos每一行后面分别拼接goal_y1、goal_y2、goal_y3组成新的数组
            init_pos_ = np.array([np.hstack((i, j)) for i in init_pos_ for j in [goal_y1, goal_y2, goal_y3]])
            # 随机打乱init_pos的顺序
            np.random.shuffle(init_pos_)

        # 读取scenario_to_test['params']路径中的xosc文本文件
        path = scenario_to_test['data']['params']
        for num in range(len(init_pos_)):
            v_init = init_pos_[num][2]
            x_init = init_pos_[num][0]
            y_init = init_pos_[num][1]
            heading_init = init_pos_[num][3]
            y1_target = init_pos_[num][4]
            y2_target = init_pos_[num][5]
            # 读取该文件夹下后缀为xosc的文件
            file_list = [os.path.join(path, i) for i in os.listdir(path)]
            file = [i for i in file_list if i.endswith('.xosc')]
            # 读取文件内容
            with open(file[0], 'r') as f:
                lines = f.readlines()
            # 依次判断lines的每一行，如果存在字段‘<!--[Initial State]’或‘<!--[Driving Task]’，则记录下该行数据并进行修改
            # 					<!--[Initial State] v_init = 30.41, x_init = 296.34, y_init = 6.550000000000002, heading_init = 3.141592653589793-->
            # 					<!--[Driving Task] x_target = (19.41, 29.41), y_target = (4.73, 8.380000000000003)-->
            for i in range(len(lines)):
                if '<!--[Initial State]' in lines[i]:
                    lines[i] = re.sub(r'v_init = [\d.]+', f'v_init = {v_init}', lines[i])
                    lines[i] = re.sub(r'x_init = [\d.]+', f'x_init = {x_init}', lines[i])
                    lines[i] = re.sub(r'y_init = [\d.]+', f'y_init = {y_init}', lines[i])
                    lines[i] = re.sub(r'heading_init = [\d.]+', f'heading_init = {heading_init}', lines[i])
                    # print(lines[i])
                if '<!--[Driving Task]' in lines[i]:
                    lines[i] = re.sub(r'y_target = \([\d., ]+\)', f'y_target = ({y1_target}, {y2_target})', lines[i])
                    # print(lines[i])
            # 在当前目录的上一级目录下新建一个名为原文件名+num文件夹，用于存放修改后的文件
            index = num
            new_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_' + str(index))
            # 新建文件夹
            os.makedirs(new_path)
            # 将修改后的lines另存到新文件夹下，文件名为原文件名+num
            with open(os.path.join(new_path, os.path.basename(file[0])), 'w') as f:
                f.writelines(lines)
            # 复制当前目录下的.xodr文件到新文件夹下
            xodr_file = [i for i in file_list if i.endswith('.xodr')]
            shutil.copy(xodr_file[0], new_path)

