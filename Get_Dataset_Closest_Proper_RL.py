import pandas as pd
import numpy as np
import os

state_list = []
car_num = 8
# 获取当前目录
current_directory = os.getcwd()
# 获取绝对路径
# abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\\test")
# abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\outputs_choose1_onlyin")
# abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\outputs_only_in_more_new")
# abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\outputs_75_2_choose_new")
abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\outputs_only_in_guass_new")
# abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\outputs_only_in_more_guass_new")
# abs_path = os.path.abspath("D:\C\linz\BS\onsite\outputs_all\outputs_75_2_guass_new")
# 获取scenarios_collision.txt文件路径
txt_file_path = os.path.join(current_directory, 'scenarios_collision.txt')
with open(txt_file_path, 'w') as f:
    pass  # 使用pass语句创建空文件
# 获取当前目录下所有的.csv文件
csv_files = [f for f in os.listdir(abs_path) if f.endswith('.csv')]
step = 0
# 输出时间步
time_num = 1
# time_num = 5
# 循环读取所有的.csv文件
for csv_file in csv_files:
    file_path = os.path.join(abs_path, csv_file)
    df = pd.read_csv(file_path)
    if df['end'].iloc[-1] == 2:
    # if df['end'].iloc[-1] != 2:
        with open(txt_file_path, 'a') as f:
            file_name_prefix = csv_file.split('_result.csv')[0]
            f.write(f"{file_name_prefix:<30}end:{df['end'].iloc[-1]:<5}\n")
        continue
    else:
        with open(txt_file_path, 'a') as f:
            file_name_prefix = csv_file.split('_result.csv')[0]
            f.write(f"{file_name_prefix:<30}end:{df['end'].iloc[-1]:<5}\n")
        # 从左往右行驶
        if df['x_ego'].iloc[0] < df['goal_x'].iloc[0]:
            for i in range(0, len(df) - 2 - 5):
                states = []
                # 主车的状态
                states.append(df.loc[i, f'v_ego'])
                states.append((df.loc[i, f'yaw_ego'] + np.pi) % (2 * np.pi) - np.pi)

                # 目标区域中点的坐标
                states.append(df.loc[i, f'goal_x'] - df.loc[i, f'x_ego'])
                states.append(df.loc[i, f'goal_y'] - df.loc[i, f'y_ego'])

                # 偏移量，先判断在哪个车道，然后计算偏移量
                # offset = 0
                # if df.loc[i, 'lane2'] <= df.loc[i, 'y_ego']:
                #     offset = (df.loc[i, 'lane2'] + df.loc[i, 'lane1']) / 2 - df.loc[i, 'y_ego']
                # elif df.loc[i, 'lane3'] < df.loc[i, 'y_ego'] < df.loc[i, 'lane2']:
                #     offset = (df.loc[i, 'lane2'] + df.loc[i, 'lane3']) / 2 - df.loc[i, 'y_ego']
                # elif df.loc[i, 'y_ego'] <= df.loc[i, 'lane3']:
                #     offset = (df.loc[i, 'lane3'] + df.loc[i, 'lane4']) / 2 - df.loc[i, 'y_ego']
                # states.append(offset)
                # 为适配RL改成上下车道边界
                states.append(df.loc[i, 'lane1'] - df.loc[i, 'y_ego'])
                states.append(df.loc[i, 'lane4'] - df.loc[i, 'y_ego'])

                # 他车的状态，最多八辆车，不足的用(200,0,0,0)补充，且按距离从近到远的顺序排列
                distances = []
                # 依次计算每辆车的相对距离
                for j in range(1, (len(df.columns) - 9 - 7 - 2 * (time_num - 1)) // 7 + 1):
                    if abs(df.loc[i, f'y_ego'] - df.loc[i, f'y_{j}']) > 6:
                        continue
                    if df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) > 0:
                        distance = np.sqrt((df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego'])) ** 2 +
                                           (df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego']) ** 2)
                    elif df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) < 0:
                        distance = np.sqrt((df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego'])) ** 2 +
                                           (df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego']) ** 2)
                    else:
                        distance = np.sqrt((df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego']) ** 2 + (df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego']) ** 2)
                    if distance < 200:
                        distances.append((j, distance))

                # 按照距离从近到远排序，最多考虑8辆车
                distances.sort(key=lambda x: x[1])
                distances = distances[:car_num]
                for j, _ in distances:
                    if df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) > 0:
                        states.append(df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']))
                    elif df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) < 0:
                        states.append(df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']))
                    else:
                        states.append(0)

                    states.append(df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego'])
                    states.append(df.loc[i, f'v_{j}'] - df.loc[i, f'v_ego'])
                    states.append((df.loc[i, f'yaw_{j}'] + np.pi) % (2 * np.pi) - np.pi)
                    # 加入当前车辆所属车道的两条车道线
                    if df.loc[i, 'lane2'] <= df.loc[i, f'y_{j}']:
                        states.append(df.loc[i, 'lane1'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, 'lane3'] < df.loc[i, f'y_{j}'] < df.loc[i, 'lane2']:
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, f'y_{j}'] <= df.loc[i, 'lane3']:
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane4'] - df.loc[i, 'y_ego'])
                for _ in range(car_num - len(distances)):
                    states.extend([200, 0, 0, 0])
                    # 加入主车所属车道的两条车道线
                    if df.loc[i, 'lane2'] <= df.loc[i, 'y_ego']:
                        states.append(df.loc[i, 'lane1'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, 'lane3'] < df.loc[i, 'y_ego'] < df.loc[i, 'lane2']:
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, 'y_ego'] <= df.loc[i, 'lane3']:
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane4'] - df.loc[i, 'y_ego'])

                # 加入之后5个时刻的加速度和转角
                for j in range(1, 6):
                    states.append(df.loc[i + j, f'a_ego'])
                    states.append(df.loc[i + j, f'steer_ego'])

                # 将状态信息添加到np列表中
                states = np.array(states)
                state_list.append(states)

        # 从右往左行驶
        if df['x_ego'].iloc[0] > df['goal_x'].iloc[0]:
            # x_new = -x, y_new = -y, yaw_new = yaw - pi, lane_new = -lane, 其他不变
            df['x_ego'] = -df['x_ego']
            df['y_ego'] = -df['y_ego']
            df['yaw_ego'] = df['yaw_ego'] - np.pi
            df['goal_x'] = -df['goal_x']
            df['goal_y'] = -df['goal_y']
            df['lane1'] = -df['lane1']
            df['lane2'] = -df['lane2']
            df['lane3'] = -df['lane3']
            df['lane4'] = -df['lane4']
            for j in range(1, (len(df.columns) - 9 - 7 - 2 * (time_num - 1)) // 7 + 1):

                df[f'x_{j}'] = -df[f'x_{j}']
                df[f'y_{j}'] = -df[f'y_{j}']

            for i in range(0, len(df) - 2 - 5):
                states = []
                # 主车的状态
                states.append(df.loc[i, f'v_ego'])
                states.append((df.loc[i, f'yaw_ego'] + np.pi) % (2 * np.pi) - np.pi)

                # 目标区域中点的坐标
                states.append(df.loc[i, f'goal_x'] - df.loc[i, f'x_ego'])
                states.append(df.loc[i, f'goal_y'] - df.loc[i, f'y_ego'])

                # # 偏移量，先判断在哪个车道，然后计算偏移量
                # offset = 0
                # if df.loc[i, 'lane2'] <= df.loc[i, 'y_ego']:
                #     offset = (df.loc[i, 'lane2'] + df.loc[i, 'lane1']) / 2 - df.loc[i, 'y_ego']
                # elif df.loc[i, 'lane3'] < df.loc[i, 'y_ego'] < df.loc[i, 'lane2']:
                #     offset = (df.loc[i, 'lane2'] + df.loc[i, 'lane3']) / 2 - df.loc[i, 'y_ego']
                # elif df.loc[i, 'y_ego'] <= df.loc[i, 'lane3']:
                #     offset = (df.loc[i, 'lane3'] + df.loc[i, 'lane4']) / 2 - df.loc[i, 'y_ego']
                # states.append(offset)
                # 为适配RL改成上下车道边界
                states.append(df.loc[i, 'lane1'] - df.loc[i, 'y_ego'])
                states.append(df.loc[i, 'lane4'] - df.loc[i, 'y_ego'])

                # 他车的状态，最多八辆车，不足的用(200,0,0,0)补充，且按距离从近到远的顺序排列
                distances = []
                # 依次计算每辆车的相对距离
                for j in range(1, (len(df.columns) - 9 - 7 - 2 * (time_num - 1)) // 7 + 1):
                    if abs(df.loc[i, f'y_ego'] - df.loc[i, f'y_{j}']) > 6:
                        continue
                    if df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) > 0:
                        distance = np.sqrt((df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego'])) ** 2 +
                                           (df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego']) ** 2)
                    elif df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) < 0:
                        distance = np.sqrt((df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego'])) ** 2 +
                                           (df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego']) ** 2)
                    else:
                        distance = np.sqrt((df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego']) ** 2 + (df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego']) ** 2)
                    if distance < 200:
                        distances.append((j, distance))

                # 按照距离从近到远排序，最多考虑8辆车
                distances.sort(key=lambda x: x[1])
                distances = distances[:car_num]
                for j, _ in distances:
                    if df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) > 0:
                        states.append(df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] - 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']))
                    elif df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']) < 0:
                        states.append(df.loc[i, f'x_{j}'] - df.loc[i, f'x_ego'] + 1/2 * (df.loc[i, f'length_{j}'] + df.loc[i, f'length_ego']))
                    else:
                        states.append(0)

                    states.append(df.loc[i, f'y_{j}'] - df.loc[i, f'y_ego'])
                    states.append(df.loc[i, f'v_{j}'] - df.loc[i, f'v_ego'])
                    states.append((df.loc[i, f'yaw_{j}'] + np.pi) % (2 * np.pi) - np.pi)
                    # 加入当前车辆所属车道的两条车道线
                    if df.loc[i, 'lane2'] <= df.loc[i, f'y_{j}']:
                        states.append(df.loc[i, 'lane1'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, 'lane3'] < df.loc[i, f'y_{j}'] < df.loc[i, 'lane2']:
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, f'y_{j}'] <= df.loc[i, 'lane3']:
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane4'] - df.loc[i, 'y_ego'])
                for _ in range(car_num - len(distances)):
                    states.extend([200, 0, 0, 0])
                    # 加入主车所属车道的两条车道线
                    if df.loc[i, 'lane2'] <= df.loc[i, 'y_ego']:
                        states.append(df.loc[i, 'lane1'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, 'lane3'] < df.loc[i, 'y_ego'] < df.loc[i, 'lane2']:
                        states.append(df.loc[i, 'lane2'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                    elif df.loc[i, 'y_ego'] <= df.loc[i, 'lane3']:
                        states.append(df.loc[i, 'lane3'] - df.loc[i, 'y_ego'])
                        states.append(df.loc[i, 'lane4'] - df.loc[i, 'y_ego'])

                # 加入之后5个时刻的加速度和转角
                for j in range(1, 6):
                    states.append(df.loc[i + j, f'a_ego'])
                    states.append(df.loc[i + j, f'steer_ego'])

                # 将状态信息添加到np列表中
                states = np.array(states)
                state_list.append(states)

                # if len(states) != 51:
                #     print(csv_file)

    step += 1
    if step % 1000 == 0:
        print(f"step: {step}")

state_list = np.array(state_list)
# np.save('GRU_choose_closest8_properRL_0.npy', state_list)
# np.save('GRU_choose_closest8_properRL_1.npy', state_list)
# np.save('GRU_choose_closest8_properRL_out.npy', state_list).
np.save('GRU_choose_closest8_properRL_guass_0.npy', state_list)
# np.save('GRU_choose_closest8_properRL_guass_1.npy', state_list)
# np.save('GRU_choose_closest8_properRL_guass_out.npy', state_list)

