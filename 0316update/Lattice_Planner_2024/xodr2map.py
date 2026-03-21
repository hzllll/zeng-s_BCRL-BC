from opendrive2discretenet.opendrive_pa import parse_opendrive
import xml.etree.ElementTree as ET
import xml.dom.minidom
# from xml.etree.ElementTree import Comment
import math
import re

# def calmap(sceno,xodr_path,xosc_path):
def calmap(xodr_path,xosc_path):
    # path = './B_scenario_replay/'+sceno+'/'+sceno+'.xodr'
    # path_xosc = './B_scenario_replay/'+sceno+'/'+sceno+'.xosc'
    path = xodr_path
    ego_pos = []
    if not xosc_path == '':
        path_xosc = xosc_path
        # 读取OpenScenario文件
        opens = xml.dom.minidom.parse(path_xosc).documentElement

        # 读取本车信息, 记录为ego_v,ego_x,ego_y,ego_head
        ego_node = opens.getElementsByTagName('Private')[0]
        ego_init = ego_node.childNodes[3].data
        ego_v, ego_x, ego_y, ego_head = [
            float(i.split('=')[1]) for i in ego_init.split(',')]
        ego_v = abs(ego_v)
        ego_head = (ego_head + 2 * math.pi) if -math.pi <= ego_head < 0 else ego_head
        ego_pos = [ego_x, ego_y, ego_v, ego_head]

        # # 获取行驶目标, goal
        # goal_init = ego_node.childNodes[5].data
        # goal = [round(float(i), 3) for i in re.findall('-*\d+\.\d+', goal_init)]
        # goal = [[goal[0], goal[2]], [goal[1], goal[3]]]

    map = []
    """read xodr to lonlat"""
    # 打开并解析OpenDRIVE文件
    tree = ET.parse(path)
    root = tree.getroot()
    # 读xodr
    road_info = parse_opendrive(path)  # 局部坐标xy
    temp_id = 0
    for i in road_info.discretelanes:
        temp_dic = {'id': i.lane_id, 'mid': [], 'left': [], 'right': [], 'next_id': i.successor,
                    'pre_id': i.predecessor}  # , 'pre_id': int(str(i.predecessor[0]).split('.')[0])
        for j in i.center_vertices:
            # 输入xy坐标
            x = j[0]
            y = j[1]
            temp_x = x
            temp_y = y
            temp_dic['mid'].append([temp_x, temp_y])
        for j in i.left_vertices:
            # 输入xy坐标
            x = j[0]
            y = j[1]
            temp_x = x
            temp_y = y
            temp_dic['left'].append([temp_x, temp_y])
        for j in i.right_vertices:
            # 输入xy坐标
            x = j[0]
            y = j[1]
            temp_x = x
            temp_y = y
            temp_dic['right'].append([temp_x, temp_y])
        map.append(temp_dic)
    return map, ego_pos  # , goal

if __name__ == '__main__':
    sceno = 'scenario_ff434345'
    # xodr_path = './map_xodr/changan-v1.2-20241019.xodr'
    map,ego_pos,goal = calmap(sceno)
    print(map, ego_pos, goal)