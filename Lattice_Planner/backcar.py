import numpy as np
import pandas as pd
def avoid_rear_colission(observation, steer, ky, yaw0):
    s = 0.005
    ad = 0
    new_acc = 0
    new_steer = 999
    dist_ctrl = 100
    frame = pd.DataFrame()
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'yaw', 'length', 'width'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()

    if ky == 1:
        # 将坐标系逆时针旋转yaw0
        state[:, 0] = state[:, 0] * np.cos(yaw0) + state[:, 1] * np.sin(yaw0)
        state[:, 1] = -state[:, 0] * np.sin(yaw0) + state[:, 1] * np.cos(yaw0)

    if state[0, 3] < np.pi / 2 or state[0, 3] > np.pi * 3 / 2:
        direction = 1.0
    else:
        direction = -1.0

    if ky == 1:
        direction = 1.0

    state[:, 0] = state[:, 0] * direction
    ego = state[0, :]
    v = ego[2]
    # 在本车后侧
    x_ind = ego[0] > state[:, 0]
    y_ind = (np.abs(ego[1] - state[:, 1])) < ((ego[5] + state[:, 5]) / 2 + 0.8)
    ind = x_ind & y_ind
    if ind.sum() > 0:
        state_ind = state[ind, :]
        rear = state_ind[(ego[0] - state_ind[:, 0]).argmin(), :]
        rv = rear[2]
        dist = ego[0] - rear[0] - (ego[4] + rear[4]) / 2 - ad
        # if abs(rv) > abs(v):
        #     relative_speed = abs(rv) - abs(v)
        #     time_to_collision = dist / relative_speed
        #     # 判断短时间内是否可能发生碰撞
        #     if time_to_collision < 5:  # 可根据需要调整时间阈值
        #         # 计算避免碰撞的加速度
        #         new_acc = 5 * relative_speed / time_to_collision
        # if dist < 10:
        #     new_acc = 3


        if dist < dist_ctrl:
            if abs(rv) > abs(v):
                relative_speed = abs(rv) - abs(v)
                new_acc = (12 * relative_speed * relative_speed / (2 * dist))
                if dist < 5:
                    # new_acc = - 0.9 * dist + 14.5
                    # new_acc = - 0.4 * dist + 8
                    new_acc = 0
                    # if dist < 3:
                    #     new_acc = 10

        # 在本车前侧
        x_ind = ego[0] < state[:, 0]
        y_ind = (np.abs(ego[1] - state[:, 1])) < ((ego[5] + state[:, 5]) / 2 + 0.8)
        ind = x_ind & y_ind
        if ind.sum() > 0:
            state_ind = state[ind, :]
            front = state_ind[(state_ind[:, 0] - ego[0]).argmin(), :]
            fv = front[2]
            dist2 = front[0] - ego[0] - (ego[4] + front[4]) / 2 - ad
            if dist2 < dist:
                if abs(fv) < abs(v):
                    relative_speed = abs(v) - abs(fv)
                    new_acc = -(12 * relative_speed * relative_speed / (2 * dist))
                    # new_acc = -(18 * relative_speed * relative_speed / (2 * dist))

                    if dist2 < 5:
                        # new_acc = 0.9 * dist2 - 14.5
                        # new_acc = 0.4 * dist2 - 8
                        new_acc = 0
                        # if dist2 < 3:
                        #     new_acc = -10
            else:
                if dist2 < 8:
                    new_acc = 0
    else:
        x_ind = ego[0] < state[:, 0]
        y_ind = (np.abs(ego[1] - state[:, 1])) < ((ego[5] + state[:, 5]) / 2 + 0.8)
        ind = x_ind & y_ind
        if ind.sum() > 0:
            state_ind = state[ind, :]
            front = state_ind[(state_ind[:, 0] - ego[0]).argmin(), :]
            fv = front[2]
            dist = front[0] - ego[0] - (ego[4] + front[4]) / 2 - ad
            if dist < dist_ctrl:
                if abs(fv) < abs(v):
                    relative_speed = abs(v) - abs(fv)
                    new_acc = -(12 * relative_speed * relative_speed / (2 * dist))
                    if dist < 5:
                        # new_acc = 0.9 * dist - 14.5
                        # new_acc = 0.4 * dist - 8
                        new_acc = 0
                        # if dist < 3:
                        #     new_acc = -10

    # if direction == 1.0:
    #     a = len(observation['vehicle_info'])
    #     for i in range(a):
    #         if abs(state[i,0] - ego[0]) < 20:
    #             if abs(state[i,1] - ego[1]) < 5:
    #                 if state[i,1] > ego[1]:
    #                     if steer > 0:
    #                         new_steer = 0
    #                 if state[i,1] < ego[1]:
    #                     if steer < 0:
    #                         new_steer = 0
    # if direction == -1.0:
    #     a = len(observation['vehicle_info'])
    #     for i in range(a):
    #         if abs(state[i, 0] - ego[0]) < 20:
    #             if abs(state[i, 1] - ego[1]) < 5:
    #                 if state[i, 1] > ego[1]:
    #                     if steer < 0:
    #                         new_steer = 0
    #                 if state[i, 1] < ego[1]:
    #                     if steer > 0:
    #                         new_steer = 0

    if direction == 1.0:
        a = len(observation['vehicle_info'])
        for i in range(a):
            if abs(state[i,0] - ego[0]) < 15:
                if state[i,0] > ego[0]:
                    if abs(state[i,1] - ego[1]) < 5:
                        if state[i,2] < ego[2]:
                            # if state[i, 1] > ego[1]:
                            # 0914修改，他车y要高于主车2m以上才需要干预转弯
                            if state[i, 1] - ego[1] > 2:
                                if steer > s:
                                    new_steer = 0
                            # if state[i,1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if steer < -s:
                                    new_steer = 0

                        if state[i,2] > ego[2]:
                            # if state[i, 1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer > s:
                                        new_steer = 0
                            # if state[i, 1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer < -s:
                                        new_steer = 0

                if state[i,0] < ego[0]:
                    if abs(state[i,1] - ego[1]) < 5:
                        if state[i,2] > ego[2]:
                            # if state[i,1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if steer > s:
                                    new_steer = 0
                            # if state[i,1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if steer < -s:
                                    new_steer = 0

                        if state[i,2] < ego[2]:
                            # if state[i, 1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer > s:
                                        new_steer = 0
                            # if state[i, 1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer < -s:
                                        new_steer = 0
                        # else:
                        #     if abs(state[i,0] - ego[0]) < 4:
                        #         new_steer = 0

    if direction == -1.0:
        a = len(observation['vehicle_info'])
        for i in range(a):
            if abs(state[i,0] - ego[0]) < 15:
                if state[i,0] > ego[0]:
                    if abs(state[i,1] - ego[1]) < 5:
                        if state[i,2] < ego[2]:
                            # if state[i,1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if steer < -s:
                                    new_steer = 0
                            # if state[i,1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if steer > s:
                                    new_steer = 0

                        if state[i,2] > ego[2]:
                            # if state[i, 1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer < -s:
                                        new_steer = 0
                            # if state[i, 1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer > s:
                                        new_steer = 0

                if state[i,0] < ego[0]:
                    if abs(state[i,1] - ego[1]) < 5:
                        if state[i,2] > ego[2]:
                            # if state[i,1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if steer < -s:
                                    new_steer = 0
                            # if state[i,1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if steer > s:
                                    new_steer = 0

                        if state[i,2] < ego[2]:
                            # if state[i, 1] > ego[1]:
                            if state[i, 1] - ego[1] > 2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer < -s:
                                        new_steer = 0
                            # if state[i, 1] < ego[1]:
                            if state[i, 1] - ego[1] < -2:
                                if abs(state[i,0] - ego[0]) < 5:
                                    if steer > s:
                                        new_steer = 0
                        # else:
                        #     if abs(state[i,0] - ego[0]) < 4:
                        #         new_steer = 0

    return new_acc, new_steer