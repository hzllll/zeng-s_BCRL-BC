import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.append('..')
from onsite import scenarioOrganizer, env
import pandas as pd
import concurrent.futures
from Lattice_Planner import backcar
import joblib
import matplotlib.pyplot as plt
import time

# 随机种子
seed = 1
torch.manual_seed(seed)

class NetWithEncoderDecoderGRU_properRL(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, seq_length):
        super(NetWithEncoderDecoderGRU_properRL, self).__init__()

        # 定义嵌入层
        self.embedding_main_target = nn.Linear(6, embed_dim)  # 主车及目标点
        self.embedding_vehicle = nn.Linear(6, embed_dim)  # 每辆车

        # GRU 层
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 前向全连接层，带激活函数
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.seq_length = seq_length
        self.car_num = 8

    def forward(self, x):
        # 检查输入是否为单个样本
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # 拆分输入为多个子序列
        main_target = x[:, 0:6]  # 主车及目标点信息
        vehicles = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)  # (batch_size, self.car_num, 6)

        # 生成随机排列的索引
        random_indices = torch.randperm(self.car_num)
        vehicles = vehicles[:, random_indices, :]

        # 对每个子序列进行嵌入
        main_target_emb = self.embedding_main_target(main_target).unsqueeze(1)  # (batch_size, 1, embed_dim)
        vehicles_emb = self.embedding_vehicle(vehicles)  # (batch_size, 6, embed_dim)

        # 将所有嵌入拼接成一个序列
        sequence = torch.cat([vehicles_emb, main_target_emb], dim=1)  # (batch_size, 7, embed_dim)
        # 激活函数
        sequence = torch.tanh(sequence)

        # 使用 GRU 编码器进行处理
        encoder_output, hidden_state = self.encoder_gru(sequence)  # (batch_size, 7, hidden_dim)

        # 解码器的输入是编码器的最后一个输出
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)

        out = torch.zeros(x.size(0), self.seq_length, 2)
        # 解码器的输出
        for i in range(self.seq_length):
            decoder_output, hidden_state = self.decoder_gru(decoder_input, hidden_state)
            decoder_input = decoder_output

            # 通过全连接层并使用激活函数
            output = self.fc1(decoder_output)
            output = self.relu(output)
            output = torch.tanh(self.fc2(output))
            out[:, i, :] = output.squeeze(1)

        return out


# —————— 7维：wcy；lwmi；hzl_test(hzl测试wcy代码在zrl的验证代码下结果)——————
class GRUTrajectoryPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, seq_length, car_num):
        super(GRUTrajectoryPredictor, self).__init__()
        self.car_num = car_num
        self.seq_length = seq_length

        # 嵌入层：将高维特征映射到低维嵌入空间
        self.embedding_main_target = nn.Linear(7, embed_dim)  # 主车+目标点特征（7维）
        self.embedding_vehicle = nn.Linear(7, embed_dim)  # 单个障碍车特征（7维）

        self.dropout_embed = nn.Dropout(0.2)  # 嵌入层后dropout
        self.dropout_encoder = nn.Dropout(0.15) # GRU编码器后dropout
        self.dropout_fc = nn.Dropout(0.2) # 全连接层后dropout

        # GRU编码器：处理序列特征
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim,num_layers=2, batch_first=True, dropout=0.2)
        # GRU解码器：生成预测序列
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim,num_layers=2, batch_first=True, dropout=0.2)

        # 输出层：将GRU输出映射到预测值（加速度+转角）
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # 全连接层1
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 全连接层2（输出层）

    def forward(self, x):
        """
        前向传播逻辑：
        x: 输入特征，shape=(batch_size, 6 + car_num*6)
           - 前6维：主车+目标点特征
           - 后car_num*6维：障碍车特征（每个障碍车6维）
        """
        # 1. 拆分输入特征
        main_target_feat = x[:, 0:7]  # 主车+目标点特征 (batch_size, 7)
        vehicle_feats = x[:, 7:7 + self.car_num*7].reshape(-1, self.car_num, 7)  # 障碍车特征 (batch_size, car_num, 7)

        # 2. 障碍车特征随机打乱（增强模型鲁棒性，避免依赖固定顺序）
        random_indices = torch.randperm(self.car_num).to(device)
        vehicle_feats = vehicle_feats[:, random_indices, :]

        # 3. 特征嵌入（将低维特征映射到高维嵌入空间）
        main_target_emb = self.embedding_main_target(main_target_feat).unsqueeze(1)  # (batch_size, 1, embed_dim)
        vehicle_emb = self.embedding_vehicle(vehicle_feats)  # (batch_size, car_num, embed_dim)
        main_target_emb = self.dropout_embed(main_target_emb) # 主车嵌入dropout
        vehicle_emb = self.dropout_embed(vehicle_emb) # 障碍车嵌入dropout

        # 4. 拼接序列（障碍车嵌入 + 主车嵌入）
        seq_input = torch.cat([vehicle_emb, main_target_emb], dim=1)  # (batch_size, car_num+1, embed_dim)
        seq_input = torch.tanh(seq_input)  # 激活函数增强非线性

        # 5. GRU编码器处理序列
        encoder_output, hidden_state = self.encoder_gru(seq_input)  # encoder_output: (batch_size, car_num+1, hidden_dim)
        encoder_output = self.dropout_encoder(encoder_output) # GRU输出dropout
        # 同步更新隐藏状态（仅取最后一层，不影响解码器）
        hidden_state = self.dropout_encoder(hidden_state)

        # 6. GRU解码器生成预测序列
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)  # 解码器初始输入：编码器最后一个输出 (batch_size, 1, hidden_dim)
        pred_seq = torch.zeros(x.size(0), self.seq_length, OUTPUT_DIM).to(device)  # 初始化预测序列

        for t in range(self.seq_length):
            decoder_output, hidden_state = self.decoder_gru(decoder_input, hidden_state)  # 解码器前向传播
            decoder_input = decoder_output  # 下一时间步输入为当前输出

            # 输出层映射（预测加速度+转角）
            # fc_out = self.fc1(decoder_output)
            # fc_out = self.relu(fc_out)
            # fc_out = torch.tanh(self.fc2(fc_out))  # tanh归一化输出到[-1,1]
            # pred_seq[:, t, :] = fc_out.squeeze(1)  # 保存当前时间步预测结果
            fc_out = self.fc1(decoder_output)
            fc_out = self.relu(fc_out)
            fc_out = self.dropout_fc(fc_out) # 全连接层dropout（防过拟合核心）
            fc_out = torch.tanh(self.fc2(fc_out))
            pred_seq[:, t, :] = fc_out.squeeze(1)

        return pred_seq

# ————————end——————————


class NetWithEncoderDecoderGRU_no_goal_sim(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, seq_length):
        super(NetWithEncoderDecoderGRU_no_goal_sim, self).__init__()

        # 定义嵌入层
        self.embedding_main_target = nn.Linear(5, embed_dim)  # 主车及目标点
        self.embedding_vehicle = nn.Linear(6, embed_dim)  # 每辆车

        # GRU 层
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 前向全连接层，带激活函数
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.seq_length = seq_length
        self.car_num = 8

    def forward(self, x):
        # 检查输入是否为单个样本
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # 拆分输入为多个子序列
        main_target = x[:, 0:5]  # 主车及目标点信息
        vehicles = x[:, 5:5 + self.car_num * 6].reshape(-1, self.car_num, 6)  # (batch_size, self.car_num, 6)

        # 生成随机排列的索引
        random_indices = torch.randperm(self.car_num)
        vehicles = vehicles[:, random_indices, :]

        # 对每个子序列进行嵌入
        main_target_emb = self.embedding_main_target(main_target).unsqueeze(1)  # (batch_size, 1, embed_dim)
        vehicles_emb = self.embedding_vehicle(vehicles)  # (batch_size, 6, embed_dim)

        # 将所有嵌入拼接成一个序列
        sequence = torch.cat([vehicles_emb, main_target_emb], dim=1)  # (batch_size, 7, embed_dim)

        # 使用 GRU 编码器进行处理
        encoder_output, hidden_state = self.encoder_gru(sequence)  # (batch_size, 7, hidden_dim)

        # 解码器的输入是编码器的最后一个输出
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)

        out = torch.zeros(x.size(0), self.seq_length, 2)
        # 解码器的输出
        for i in range(self.seq_length):
            decoder_output, hidden_state = self.decoder_gru(decoder_input, hidden_state)
            decoder_input = decoder_output

            # 通过全连接层并使用激活函数
            output = self.fc1(decoder_output)
            output = self.relu(output)
            output = torch.tanh(self.fc2(output))
            out[:, i, :] = output.squeeze(1)

        return out

def observation_to_state1_sim(observation, frame_n, action, goal, map_info):
    car_num = 8
    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()

    # 加入主车的状态v yaw
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    # 目标点的坐标
    states.extend([goal[0] - state[0, 0], goal[1] - state[0, 1]])
    # 加入偏移量
    offset = 0
    if state[0, 1] >= map_info[0]['right_bound']:
        offset = map_info[0]['center'] - state[0, 1]
    elif map_info[1]['right_bound'] < state[0, 1] < map_info[1]['left_bound']:
        offset = map_info[1]['center'] - state[0, 1]
    elif state[0, 1] <= map_info[2]['left_bound']:
        offset = map_info[2]['center'] - state[0, 1]
    states.append(offset)

    # 他车的状态，最多六辆车，不足的用(200,0,0,0)补充，且按距离从近到远的顺序排列
    distances = []
    # 依次计算每辆车的相对距离
    for j in range(1, len(state)):
        distance = np.sqrt((state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5])) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        if distance < 200:
            distances.append((j, distance))
    # 按照距离从近到远排序，最多考虑6辆车
    distances.sort(key=lambda x: x[1])
    distances = distances[:car_num]
    for j, _ in distances:
        states.append(state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5]))
        states.append(state[j, 1] - state[0, 1])
        states.append(state[j, 2] - state[0, 2])
        states.append((state[j, 4] + np.pi) % (2 * np.pi) - np.pi)
        # 加入当前车辆所属车道的两条车道线
        if map_info[0]['right_bound'] <= state[j, 1]:
            states.append(map_info[0]['left_bound'] - state[0, 1])
            states.append(map_info[0]['right_bound'] - state[0, 1])
        elif map_info[1]['right_bound'] < state[j, 1] < map_info[1]['left_bound']:
            states.append(map_info[1]['left_bound'] - state[0, 1])
            states.append(map_info[1]['right_bound'] - state[0, 1])
        elif state[j, 1] <= map_info[2]['left_bound']:
            states.append(map_info[2]['left_bound'] - state[0, 1])
            states.append(map_info[2]['right_bound'] - state[0, 1])
    for _ in range(car_num - len(distances)):
        states.extend([200, 0, 0, 0])
        # 加入主车所属车道的两条车道线
        if map_info[0]['right_bound'] <= state[0, 1]:
            states.append(map_info[0]['left_bound'] - state[0, 1])
            states.append(map_info[0]['right_bound'] - state[0, 1])
        elif map_info[1]['right_bound'] < state[0, 1] < map_info[1]['left_bound']:
            states.append(map_info[1]['left_bound'] - state[0, 1])
            states.append(map_info[1]['right_bound'] - state[0, 1])
        elif state[0, 1] <= map_info[2]['left_bound']:
            states.append(map_info[2]['left_bound'] - state[0, 1])
            states.append(map_info[2]['right_bound'] - state[0, 1])

    return states
def observation_to_state2_sim(observation, frame_n, action, goal, map_info):
    car_num = 8
    goal_neg = [-goal[0], -goal[1]]
    map_info_neg = {}
    for j in range(3):
        map_info_neg[j] = {
            'left_bound': -map_info[j]['left_bound'],
            'center': -map_info[j]['center'],
            'right_bound': -map_info[j]['right_bound']
        }

    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()
    state[:, 0] = -state[:, 0]
    state[:, 1] = -state[:, 1]
    # 0802 修正yaw: -state[:, 4] -> state[:, 4]
    state[:, 4] = state[:, 4] - np.pi

    # 加入主车的状态v yaw
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    # 加入目标区域中点的坐标
    states.extend([goal_neg[0] - state[0, 0], goal_neg[1] - state[0, 1]])
    # 加入偏移量
    offset = 0
    if state[0, 1] >= map_info_neg[0]['right_bound']:
        offset = map_info_neg[0]['center'] - state[0, 1]
    elif map_info_neg[1]['right_bound'] < state[0, 1] < map_info_neg[1]['left_bound']:
        offset = map_info_neg[1]['center'] - state[0, 1]
    elif state[0, 1] <= map_info_neg[2]['left_bound']:
        offset = map_info_neg[2]['center'] - state[0, 1]
    states.append(offset)

    # 他车的状态，最多六辆车，不足的用(200,0,0,0)补充，且按距离从近到远的顺序排列
    distances = []
    # 依次计算每辆车的相对距离
    for j in range(1, len(state)):
        distance = np.sqrt((state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5])) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        if distance < 200:
            distances.append((j, distance))
    # 按照距离从近到远排序，最多考虑6辆车
    distances.sort(key=lambda x: x[1])
    distances = distances[:car_num]
    for j, _ in distances:
        states.append(state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5]))
        states.append(state[j, 1] - state[0, 1])
        states.append(state[j, 2] - state[0, 2])
        states.append((state[j, 4] + np.pi) % (2 * np.pi) - np.pi)
        # 加入当前车辆所属车道的两条车道线
        if map_info_neg[0]['right_bound'] <= state[j, 1]:
            states.append(map_info_neg[0]['left_bound'] - state[0, 1])
            states.append(map_info_neg[0]['right_bound'] - state[0, 1])
        elif map_info_neg[1]['right_bound'] < state[j, 1] < map_info_neg[1]['left_bound']:
            states.append(map_info_neg[1]['left_bound'] - state[0, 1])
            states.append(map_info_neg[1]['right_bound'] - state[0, 1])
        elif state[j, 1] <= map_info_neg[2]['left_bound']:
            states.append(map_info_neg[2]['left_bound'] - state[0, 1])
            states.append(map_info_neg[2]['right_bound'] - state[0, 1])
    for _ in range(car_num - len(distances)):
        states.extend([200, 0, 0, 0])
        # 加入主车所属车道的两条车道线
        if map_info_neg[0]['right_bound'] <= state[0, 1]:
            states.append(map_info_neg[0]['left_bound'] - state[0, 1])
            states.append(map_info_neg[0]['right_bound'] - state[0, 1])
        elif map_info_neg[1]['right_bound'] < state[0, 1] < map_info_neg[1]['left_bound']:
            states.append(map_info_neg[1]['left_bound'] - state[0, 1])
            states.append(map_info_neg[1]['right_bound'] - state[0, 1])
        elif state[0, 1] <= map_info_neg[2]['left_bound']:
            states.append(map_info_neg[2]['left_bound'] - state[0, 1])
            states.append(map_info_neg[2]['right_bound'] - state[0, 1])

    return states

def observation_to_state1_simRL_proper(observation, goal, map_info):
    car_num = 8
    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()

    # 加入主车的状态v yaw
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    # 目标点的坐标
    states.extend([goal[0] - state[0, 0], goal[1] - state[0, 1]])
    # # 加入偏移量
    # offset = 0
    # if state[0, 1] >= map_info[0]['right_bound']:
    #     offset = map_info[0]['center'] - state[0, 1]
    # elif map_info[1]['right_bound'] < state[0, 1] < map_info[1]['left_bound']:
    #     offset = map_info[1]['center'] - state[0, 1]
    # elif state[0, 1] <= map_info[2]['left_bound']:
    #     offset = map_info[2]['center'] - state[0, 1]
    # states.append(offset)
    # 加入上下车道边界线
    states.append(map_info[0]['left_bound'] - state[0, 1])
    states.append(map_info[2]['right_bound'] - state[0, 1])

    # 他车的状态，最多六辆车，不足的用(200,0,0,0)补充，且按距离从近到远的顺序排列
    distances = []
    # 依次计算每辆车的相对距离
    for j in range(1, len(state)):
        if abs(state[j, 1] - state[0, 1]) > 6:
            continue
        if state[j, 0] - state[0, 0] - 0.5 * (state[j, 5] + state[0, 5]) > 0:
            distance = np.sqrt((state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5])) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        elif state[j, 0] - state[0, 0] + 0.5 * (state[j, 5] + state[0, 5]) < 0:
            distance = np.sqrt((state[j, 0] - state[0, 0] + 1 / 2 * (state[j, 5] + state[0, 5])) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        else:
            distance = np.sqrt((state[j, 0] - state[0, 0]) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        if distance < 200:
            distances.append((j, distance))

    # 按照距离从近到远排序，最多考虑6辆车
    distances.sort(key=lambda x: x[1])
    distances = distances[:car_num]
    for j, _ in distances:
        if state[j, 0] - state[0, 0] - 0.5 * (state[j, 5] + state[0, 5]) > 0:
            states.append(state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5]))
        elif state[j, 0] - state[0, 0] + 0.5 * (state[j, 5] + state[0, 5]) < 0:
            states.append(state[j, 0] - state[0, 0] + 1 / 2 * (state[j, 5] + state[0, 5]))
        else:
            states.append(0)

        states.append(state[j, 1] - state[0, 1])
        states.append(state[j, 2] - state[0, 2])
        states.append((state[j, 4] + np.pi) % (2 * np.pi) - np.pi)
        # 加入当前车辆所属车道的两条车道线
        if map_info[0]['right_bound'] <= state[j, 1]:
            states.append(map_info[0]['left_bound'] - state[0, 1])
            states.append(map_info[0]['right_bound'] - state[0, 1])
        elif map_info[1]['right_bound'] < state[j, 1] < map_info[1]['left_bound']:
            states.append(map_info[1]['left_bound'] - state[0, 1])
            states.append(map_info[1]['right_bound'] - state[0, 1])
        elif state[j, 1] <= map_info[2]['left_bound']:
            states.append(map_info[2]['left_bound'] - state[0, 1])
            states.append(map_info[2]['right_bound'] - state[0, 1])
    for _ in range(car_num - len(distances)):
        states.extend([200, 0, 0, 0])
        # 加入主车所属车道的两条车道线
        if map_info[0]['right_bound'] <= state[0, 1]:
            states.append(map_info[0]['left_bound'] - state[0, 1])
            states.append(map_info[0]['right_bound'] - state[0, 1])
        elif map_info[1]['right_bound'] < state[0, 1] < map_info[1]['left_bound']:
            states.append(map_info[1]['left_bound'] - state[0, 1])
            states.append(map_info[1]['right_bound'] - state[0, 1])
        elif state[0, 1] <= map_info[2]['left_bound']:
            states.append(map_info[2]['left_bound'] - state[0, 1])
            states.append(map_info[2]['right_bound'] - state[0, 1])

    # states = torch.tensor(states, dtype=torch.float32).to(device)

    return states
def observation_to_state2_simRL_proper(observation, goal, map_info):
    car_num = 8
    goal_neg = [-goal[0], -goal[1]]
    map_info_neg = {}
    for j in range(3):
        map_info_neg[j] = {
            'left_bound': -map_info[j]['left_bound'],
            'center': -map_info[j]['center'],
            'right_bound': -map_info[j]['right_bound']
        }

    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()
    state[:, 0] = -state[:, 0]
    state[:, 1] = -state[:, 1]
    # 0802 修正yaw: -state[:, 4] -> state[:, 4]
    state[:, 4] = state[:, 4] - np.pi

    # 加入主车的状态v yaw
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    # 目标点的坐标
    states.extend([goal_neg[0] - state[0, 0], goal_neg[1] - state[0, 1]])
    # # 加入偏移量
    # offset = 0
    # if state[0, 1] >= map_info[0]['right_bound']:
    #     offset = map_info[0]['center'] - state[0, 1]
    # elif map_info[1]['right_bound'] < state[0, 1] < map_info[1]['left_bound']:
    #     offset = map_info[1]['center'] - state[0, 1]
    # elif state[0, 1] <= map_info[2]['left_bound']:
    #     offset = map_info[2]['center'] - state[0, 1]
    # states.append(offset)
    # 加入上下车道边界线
    states.append(map_info_neg[0]['left_bound'] - state[0, 1])
    states.append(map_info_neg[2]['right_bound'] - state[0, 1])

    # 他车的状态，最多六辆车，不足的用(200,0,0,0)补充，且按距离从近到远的顺序排列
    distances = []
    # 依次计算每辆车的相对距离
    for j in range(1, len(state)):
        if abs(state[j, 1] - state[0, 1]) > 6:
            continue
        if state[j, 0] - state[0, 0] - 0.5 * (state[j, 5] + state[0, 5]) > 0:
            distance = np.sqrt((state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5])) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        elif state[j, 0] - state[0, 0] + 0.5 * (state[j, 5] + state[0, 5]) < 0:
            distance = np.sqrt((state[j, 0] - state[0, 0] + 1 / 2 * (state[j, 5] + state[0, 5])) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        else:
            distance = np.sqrt((state[j, 0] - state[0, 0]) ** 2 + (state[j, 1] - state[0, 1]) ** 2)
        if distance < 200:
            distances.append((j, distance))

    # 按照距离从近到远排序，最多考虑6辆车
    distances.sort(key=lambda x: x[1])
    distances = distances[:car_num]
    for j, _ in distances:
        if state[j, 0] - state[0, 0] - 0.5 * (state[j, 5] + state[0, 5]) > 0:
            states.append(state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5]))
        elif state[j, 0] - state[0, 0] + 0.5 * (state[j, 5] + state[0, 5]) < 0:
            states.append(state[j, 0] - state[0, 0] + 1 / 2 * (state[j, 5] + state[0, 5]))
        else:
            states.append(0)

        states.append(state[j, 1] - state[0, 1])
        states.append(state[j, 2] - state[0, 2])
        states.append((state[j, 4] + np.pi) % (2 * np.pi) - np.pi)
        # 加入当前车辆所属车道的两条车道线
        if map_info_neg[0]['right_bound'] <= state[j, 1]:
            states.append(map_info_neg[0]['left_bound'] - state[0, 1])
            states.append(map_info_neg[0]['right_bound'] - state[0, 1])
        elif map_info_neg[1]['right_bound'] < state[j, 1] < map_info_neg[1]['left_bound']:
            states.append(map_info_neg[1]['left_bound'] - state[0, 1])
            states.append(map_info_neg[1]['right_bound'] - state[0, 1])
        elif state[j, 1] <= map_info_neg[2]['left_bound']:
            states.append(map_info_neg[2]['left_bound'] - state[0, 1])
            states.append(map_info_neg[2]['right_bound'] - state[0, 1])
    for _ in range(car_num - len(distances)):
        states.extend([200, 0, 0, 0])
        # 加入主车所属车道的两条车道线
        if map_info_neg[0]['right_bound'] <= state[0, 1]:
            states.append(map_info_neg[0]['left_bound'] - state[0, 1])
            states.append(map_info_neg[0]['right_bound'] - state[0, 1])
        elif map_info_neg[1]['right_bound'] < state[0, 1] < map_info_neg[1]['left_bound']:
            states.append(map_info_neg[1]['left_bound'] - state[0, 1])
            states.append(map_info_neg[1]['right_bound'] - state[0, 1])
        elif state[0, 1] <= map_info_neg[2]['left_bound']:
            states.append(map_info_neg[2]['left_bound'] - state[0, 1])
            states.append(map_info_neg[2]['right_bound'] - state[0, 1])

    # states = torch.tensor(states, dtype=torch.float32).to(device)

    return states

def get_lane_info1(road_info):
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

def process_input_directory(input_dir, output_dir, model):
    times = []
    # 实例化场景管理模块（ScenairoOrganizer）和场景测试模块（Env）
    so = scenarioOrganizer.ScenarioOrganizer()
    envs = env.Env()
    so.load(input_dir, output_dir)
    frame_n = 0
    action = [0, 0]
    ky = 0
    while True:
        scenario_to_test = so.next()
        if scenario_to_test is None:
            break  # 如果场景管理模块给出None，意味着所有场景已测试完毕。
        scenario_to_test['test_settings']['visualize'] = False
        observation, traj = envs.make(scenario=scenario_to_test)
        road_info = envs.controller.observation.road_info
        goal = [np.mean(observation['test_setting']['goal']['x']), np.mean(observation['test_setting']['goal']['y'])]
        init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']]
        lane_num = len(road_info.discretelanes)
        if lane_num != 6:
            continue
        if goal[0] > init_pos[0]:
            # 从左往右
            map_info = get_lane_info1(road_info)
        else:
            # 从右往左
            # continue
            map_info = get_lane_info2(road_info)
        yaw0 = observation['vehicle_info']['ego']['yaw']
        try:
            while observation['test_setting']['end'] == -1:
                if goal[0] > init_pos[0]:
                    next_obs = observation_to_state1_simRL_proper(observation, goal, map_info)
                else:
                    next_obs = observation_to_state2_simRL_proper(observation, goal, map_info)
                    # next_obs = observation_to_state2_closest6(observation, frame_n, action, goal, map_info)
                    # next_obs = observation_to_state2_sim(observation, frame_n, action, goal, map_info)

                next_obs = torch.tensor(next_obs, dtype=torch.float32)

                # tic = time.perf_counter()
                action = model(next_obs)
                # toc = time.perf_counter()
                # time_ = (toc - tic) * 1000
                # # times.append(time_)
                # print(f"Time: {time_:.2f}ms")

                # # times超过2000次，打印平均时间，并保存矢量图
                # if len(times) == 2000:
                #     average_time = sum(times) / 2000
                #     plt.plot(times)
                #     plt.axhline(y=average_time, color='blue', linestyle='-', label='Average Time')
                #     plt.legend()
                #     # 保存svg矢量图
                #     plt.savefig('RunTimes_G.svg', format='svg')
                #     # 保存times
                #     times = np.array(times)
                #     np.save('RunTimes_G.npy', times)

                # # 画出五个时间步的主车位置点，用二自由度车辆模型进行仿真
                # x_sim = []
                # y_sim = []
                # x_sim = np.append(x_sim, observation['vehicle_info']['ego']['x'])
                # y_sim = np.append(y_sim, observation['vehicle_info']['ego']['y'])
                # v_sim = observation['vehicle_info']['ego']['v']
                # heading_sim = observation['vehicle_info']['ego']['yaw']
                # action1 = action.clone()
                # for i in range(5):
                #     action_timestep = action1[:, i, :]
                #     action_timestep = action_timestep.detach().cpu().numpy().reshape(-1)
                #     action_timestep[0] = action_timestep[0] * 10
                #     action_timestep[1] = action_timestep[1] * 0.15
                #     x_sim = np.append(x_sim, x_sim[-1] + v_sim * 0.04 * np.cos(heading_sim))
                #     y_sim = np.append(y_sim, y_sim[-1] + v_sim * 0.04 * np.sin(heading_sim))
                #     v_sim = v_sim + action_timestep[0] * 0.04
                #     heading_sim = heading_sim + v_sim / (observation['vehicle_info']['ego']['length'] / 1.7) * np.tan(action_timestep[1]) * 0.04
                # # 画出仿真的主车位置
                # plt.plot(x_sim, y_sim, 'ro')
                # plt.pause(0.1)

                # # 获取第一个时间步的动作
                # first_timestep_action = action[:, 0, :]
                # action = first_timestep_action.detach().cpu().numpy().reshape(-1)
                # action[0] = action[0] * 10
                # action[1] = action[1] * 0.15
                # observation = envs.step(action)
                # frame_n += 1

                # 依次获取每个时间步的动作
                for i in range(5):
                    if observation['test_setting']['end'] != -1:
                        break
                    action_timestep = action[:, i, :]
                    action_timestep = action_timestep.detach().cpu().numpy().reshape(-1)
                    action_timestep[0] = action_timestep[0] * 10
                    action_timestep[1] = action_timestep[1] * 0.15
                    # action_timestep[1] = action_timestep[1] * 0.6
                    observation = envs.step(action_timestep)
                    frame_n += 1

        finally:
            so.add_result(scenario_to_test, observation['test_setting']['end'])

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetWithEncoderDecoderGRU_properRL(32, 64, 2, 5)

    # model.load_state_dict(torch.load('model_GRU_Multi1_choose_closest8_sim_01o.pth'))
    model.load_state_dict(torch.load('model_GRU_RL_guass_23.pth'))

    # input_dirs = [f"E:\\python_program\\Onsite\\planner\\inputs\\inputs_all_multi\\inputs{i}" for i in range(5)]
    input_dirs = [f"E:\\python_program\\Onsite\\planner\\inputs\\inputs_test_multi\\inputs{i}" for i in range(5)]
    output_dir = r"E:\python_program\Onsite\planner\outputs\model_GRU_RL_guass_23_test"

    # 使用进程池并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_input_directory, input_dir, output_dir, model) for input_dir in input_dirs]

        for future in concurrent.futures.as_completed(futures):
            future.result()
            # try:
            #     future.result()
            # except Exception as e:
            #     print(f"进程执行时发生异常: {e}")

    # # 单线程处理
    # input_dir = r"E:\python_program\Onsite\planner\inputs\inputs_test"
    # # output_dir = r"E:\python_program\Onsite\planner\outputs\Lattice_choose_GRU_test2"
    # # input_dir = r"E:\python_program\Onsite\planner\inputs\inputs3"
    # output_dir = r"E:\python_program\Onsite\planner\outputs\outputs_test_gift"
    # # input_dir = r"E:\python_program\Onsite\planner\inputs\multi_lane_test"
    # # output_dir = r"E:\python_program\Onsite\planner\outputs\outputs_multi_lane_test_gru"
    # process_input_directory(input_dir, output_dir)

    # # model = NetWithEncoderDecoderGRU_no_goal_sim(32, 64, 2, 5)
    # model = NetWithEncoderDecoderGRU_properRL(32, 64, 2, 5)
    # for i in range(15, 23):
    #     model.load_state_dict(torch.load(f'model_GRU_Multi1_choose_seed4_{i}.pth'))
    #     input_dir = f"E:\\python_program\\Onsite\\planner\\inputs\\inputs_test"
    #     output_dir = f"E:\\python_program\\Onsite\\planner\\outputs\\Lattice_GRU_choose_seed4_{i}"
    #     process_input_directory(input_dir, output_dir)
    #     print(f"inputs{i}处理完毕！")

