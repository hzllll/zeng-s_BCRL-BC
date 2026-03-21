import xodr2map
from planner_func import global_planner
import matplotlib.pyplot as plt
import numpy as np

def draw_map(map,ego_pos,goal, global_path):
    map_mid = [[], []]
    map_left = [[], []]
    map_right = [[], []]
    path_x = []
    path_y = []
    plt.figure(figsize=(10, 8))

    # 原地图
    for i in map:
        for xx in range(len(i['mid'])):
            map_mid[0].append(i['mid'][xx][0])
            map_mid[1].append(i['mid'][xx][1])
        for xx in range(len(i['left'])):
            map_left[0].append(i['left'][xx][0])
            map_left[1].append(i['left'][xx][1])
        for xx in range(len(i['right'])):
            map_right[0].append(i['right'][xx][0])
            map_right[1].append(i['right'][xx][1])
    plt.scatter(map_mid[0], map_mid[1], color="C1", s=1)
    plt.scatter(map_left[0], map_left[1], color="C7", s=1)
    plt.scatter(map_right[0], map_right[1], color="C7", s=1)

    # 起始ID
    ego_pos_theta = ego_pos[3]
    lenth = ego_pos[2]
    dx = lenth * np.cos(ego_pos_theta)
    dy = lenth * np.sin(ego_pos_theta)
    plt.quiver(ego_pos[0], ego_pos[1], dx, dy, angles='xy', scale_units='xy', scale=1, color='r')
    plt.scatter(ego_pos[0], ego_pos[1], color="C0", s=10)    
    plt.plot([goal[0][0], goal[1][0]],[goal[0][1],goal[0][1]],color="C3")
    plt.plot([goal[0][0], goal[1][0]],[goal[1][1],goal[1][1]],color="C3")
    plt.plot([goal[0][0], goal[0][0]],[goal[0][1],goal[1][1]],color="C3")
    plt.plot([goal[1][0], goal[1][0]],[goal[0][1],goal[1][1]],color="C3")
    
    # 看最后的路径点
    if global_path != []:
        for i in global_path:
            path_x.append(i[0])
            path_y.append(i[1])
        plt.plot(path_x, path_y, 'g')
        x_max,y_max = np.max(global_path, axis=0)
        x_min,y_min = np.min(global_path, axis=0)
        plt.xlim(x_min-20, x_max+20)
        plt.ylim(y_min-20, y_max+20)

    plt.show(block=True)
    plt.show()

if __name__ == '__main__':
    sceno = 'scenario_0a8a2348'
    map,ego_state,goal = xodr2map.calmap(sceno)
    glb_plr = global_planner(map)
    glb_plr.map_data_process()
    global_path = glb_plr.find_global_path(goal, ego_state)
    draw_map(map, ego_state, goal, global_path)
            