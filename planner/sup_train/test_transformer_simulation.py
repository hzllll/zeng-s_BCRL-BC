import numpy as np
import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
import time

# 添加上级目录以导入 onsite 包
sys.path.append('..')
from onsite import scenarioOrganizer, env

# ================= 配置区域 =================
# 请确保这些参数与训练时的配置完全一致！
D_MODEL = 256
FFN_DIM = 4 * D_MODEL
num_encoder_layers = 3
num_decoder_layers = 3
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径 (请修改为你训练出来的pth文件路径)
MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Transformer_trajectory_model_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_zDATASET.pth"
# ===========================================

# 1. 定义 Transformer 模型 (必须与训练代码完全一致)
class TransformerTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        seq_length: int,
        car_num: int,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.car_num = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim

        # 这里的维度是 6，与你确认的数据集一致
        self.embedding_main_target = nn.Linear(6, d_model)
        self.embedding_vehicle = nn.Linear(6, d_model)

        self.dropout_embed = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.future_queries = nn.Parameter(torch.randn(seq_length, d_model))

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # x shape: (Batch_Size, 6 + car_num * 6)
        # 如果输入是单个样本 (54,)，需要升维成 (1, 54)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        main_target_feat = x[:, 0:6]
        vehicle_feats = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)

        main_emb = self.embedding_main_target(main_target_feat).unsqueeze(1)
        veh_emb = self.embedding_vehicle(vehicle_feats)
        
        tokens = torch.cat([veh_emb, main_emb], dim=1)
        tokens = torch.tanh(tokens)
        tokens = self.dropout_embed(tokens)

        memory = self.encoder(tokens)

        tgt = self.future_queries.unsqueeze(0).expand(x.size(0), -1, -1)
        dec_out = self.decoder(tgt=tgt, memory=memory)

        pred = self.head(dec_out)
        return pred

# 2. 定义特征提取函数 (直接复用原测试脚本的逻辑，确保特征顺序一致)
def observation_to_state1_simRL_proper(observation, goal, map_info):
    car_num = 8
    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()

    # --- 主车特征 (2维) ---
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    # --- 目标点特征 (2维) ---
    states.extend([goal[0] - state[0, 0], goal[1] - state[0, 1]])
    # --- 车道边界特征 (2维) ---
    states.extend([map_info[0]['left_bound'] - state[0, 1], map_info[2]['right_bound'] - state[0, 1]])
    
    # 目前共 6 维，与 Transformer 输入匹配

    # --- 障碍车处理 ---
    distances = []
    for j in range(1, len(state)):
        # 简单的筛选逻辑
        if abs(state[j, 1] - state[0, 1]) > 6: continue # 横向太远
        
        # 计算距离
        dx = state[j, 0] - state[0, 0]
        dy = state[j, 1] - state[0, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 200:
            distances.append((j, dist))

    distances.sort(key=lambda x: x[1])
    distances = distances[:car_num]

    for j, _ in distances:
        # 相对位置 x
        if state[j, 0] - state[0, 0] - 0.5 * (state[j, 5] + state[0, 5]) > 0:
            states.append(state[j, 0] - state[0, 0] - 1 / 2 * (state[j, 5] + state[0, 5]))
        elif state[j, 0] - state[0, 0] + 0.5 * (state[j, 5] + state[0, 5]) < 0:
            states.append(state[j, 0] - state[0, 0] + 1 / 2 * (state[j, 5] + state[0, 5]))
        else:
            states.append(0)

        # 相对位置 y, 相对速度 v, 相对航向 yaw
        states.append(state[j, 1] - state[0, 1])
        states.append(state[j, 2] - state[0, 2])
        states.append((state[j, 4] + np.pi) % (2 * np.pi) - np.pi)
        
        # 车道线信息 (2维)
        if map_info[0]['right_bound'] <= state[j, 1]:
            states.append(map_info[0]['left_bound'] - state[0, 1])
            states.append(map_info[0]['right_bound'] - state[0, 1])
        elif map_info[1]['right_bound'] < state[j, 1] < map_info[1]['left_bound']:
            states.append(map_info[1]['left_bound'] - state[0, 1])
            states.append(map_info[1]['right_bound'] - state[0, 1])
        elif state[j, 1] <= map_info[2]['left_bound']:
            states.append(map_info[2]['left_bound'] - state[0, 1])
            states.append(map_info[2]['right_bound'] - state[0, 1])
            
    # 补全不足的车辆
    for _ in range(car_num - len(distances)):
        states.extend([200, 0, 0, 0]) # 4维
        # 补全车道线 (2维) - 使用主车所在车道线代替
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

# 辅助函数：获取车道信息

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
    

# 3. 核心处理流程
def process_input_directory(input_dir, output_dir, model):
    so = scenarioOrganizer.ScenarioOrganizer()
    envs = env.Env()
    so.load(input_dir, output_dir)
    
    while True:
        scenario_to_test = so.next()
        if scenario_to_test is None:
            break  # 如果场景管理模块给出None，意味着所有场景已测试完毕。
                
        scenario_to_test['test_settings']['visualize'] = False
        observation, traj = envs.make(scenario=scenario_to_test)
        road_info = envs.controller.observation.road_info
        
        # ... 初始化 goal, map_info 等 (参考原代码) ...
        goal = [np.mean(observation['test_setting']['goal']['x']), np.mean(observation['test_setting']['goal']['y'])]
        init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']]
        
        # 简化版车道判断逻辑
        map_info = get_lane_info1(road_info) # 假设都是从左往右，实际需保留原代码的判断逻辑
        
        try:
            while observation['test_setting']['end'] == -1:
                # 1. 获取状态特征
                next_obs = observation_to_state1_simRL_proper(observation, goal, map_info)
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(DEVICE)
                
                # 2. 模型推理
                with torch.no_grad():
                    action_seq = model(next_obs_tensor) # 输出 (1, 5, 2)
                
                # 3. 执行动作 (开环执行5步，或者只执行第1步)
                # 这里我们采取“滚动优化”策略：只执行预测序列的第1步，然后重新规划
                # 或者按照原代码逻辑，一次预测执行5步
                
                # 方案A：只执行第1步 (更稳健，推荐)
                # action_step = action_seq[0, 0, :].cpu().numpy()
                # action_step[0] *= 10   # 反归一化 加速度
                # action_step[1] *= 0.15 # 反归一化 转角
                # observation = envs.step(action_step)
                
                # 方案B：执行所有5步 (原代码逻辑)
                for i in range(5):
                    if observation['test_setting']['end'] != -1: break
                    action_step = action_seq[0, i, :].cpu().numpy()
                    action_step[0] *= 6    # 注意：你训练代码里是除以6，这里要乘6！
                    action_step[1] *= 0.15 # 训练代码里是除以0.15
                    observation = envs.step(action_step)

        finally:
            so.add_result(scenario_to_test, observation['test_setting']['end'])

if __name__ == "__main__":
    # 初始化模型
    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
        nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        dim_feedforward=FFN_DIM, dropout=0.0
    ).to(DEVICE)
    
    # 加载权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"成功加载 Transformer 模型: {MODEL_PATH}")
    else:
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        exit()
        
    model.eval() # 开启验证模式

    # 设置输入输出目录 (请修改为你的实际路径)
    # input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_test_multi/inputs{i}" for i in range(5)]
    # output_dir = "/root/autodl-tmp/BCRL/bc/planner/outputs/Transformer_test_result"
    
    input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_B"]
    output_dir = "/root/autodl-tmp/BCRL/bc/planner/outputs/Transformer_test_result_B"
    # 并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_input_directory, input_dir, output_dir, model) for input_dir in input_dirs]
        for future in concurrent.futures.as_completed(futures):
            future.result()