import numpy as np
import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
import time
import glob  # 用于自动查找对应实验的 .pth 模型文件
from datetime import datetime

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (bc/) 的路径：当前目录 -> 上一级(planner) -> 上一级(bc)
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from onsite import scenarioOrganizer, env

# ================= 配置区域 =================
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===========================================

# 1. 定义 Vanilla RNN 模型 (必须与训练代码完全一致)
class VanillaRNNTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_length: int,
        car_num: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        nonlinearity: str = 'tanh',
    ):
        super().__init__()
        self.car_num    = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding_main_target = nn.Linear(6, embed_dim)
        self.embedding_vehicle     = nn.Linear(6, embed_dim)
        self.dropout_embed         = nn.Dropout(dropout)

        self.encoder_rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity=nonlinearity,
        )
        self.dropout_encoder = nn.Dropout(dropout)

        self.decoder_rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity=nonlinearity,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.size(0)

        main_target_feat = x[:, 0:6]
        vehicle_feats    = x[:, 6:6 + self.car_num * 6].reshape(B, self.car_num, 6)

        rand_idx     = torch.randperm(self.car_num, device=x.device)
        vehicle_feats = vehicle_feats[:, rand_idx, :]

        main_emb = self.dropout_embed(self.embedding_main_target(main_target_feat).unsqueeze(1))
        veh_emb  = self.dropout_embed(self.embedding_vehicle(vehicle_feats))

        seq_input = torch.cat([veh_emb, main_emb], dim=1)
        seq_input = torch.tanh(seq_input)

        enc_out, hidden = self.encoder_rnn(seq_input)
        enc_out = self.dropout_encoder(enc_out)
        hidden  = self.dropout_encoder(hidden)

        decoder_input = enc_out[:, -1, :].unsqueeze(1)
        pred_seq = torch.zeros(B, self.seq_length, self.output_dim, device=x.device)

        for t in range(self.seq_length):
            dec_out, hidden = self.decoder_rnn(decoder_input, hidden)
            decoder_input   = dec_out
            pred_seq[:, t, :] = self.head(dec_out).squeeze(1)

        return pred_seq

# 2. 定义特征提取函数 (与原测试脚本完全一致)
def observation_to_state1_simRL_proper(observation, goal, map_info):
    car_num = 8
    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()

    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    states.extend([goal[0] - state[0, 0], goal[1] - state[0, 1]])
    states.extend([map_info[0]['left_bound'] - state[0, 1], map_info[2]['right_bound'] - state[0, 1]])
    
    distances = []
    for j in range(1, len(state)):
        if abs(state[j, 1] - state[0, 1]) > 6: continue
        dx = state[j, 0] - state[0, 0]
        dy = state[j, 1] - state[0, 1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 200:
            distances.append((j, dist))

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
    state[:, 4] = state[:, 4] - np.pi

    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])
    states.extend([goal_neg[0] - state[0, 0], goal_neg[1] - state[0, 1]])
    states.append(map_info_neg[0]['left_bound'] - state[0, 1])
    states.append(map_info_neg[2]['right_bound'] - state[0, 1])

    distances = []
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
            break

        scenario_to_test['test_settings']['visualize'] = False
        observation, traj = envs.make(scenario=scenario_to_test)
        road_info = envs.controller.observation.road_info
        
        goal = [np.mean(observation['test_setting']['goal']['x']), np.mean(observation['test_setting']['goal']['y'])]
        init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']]
        
        lane_num = len(road_info.discretelanes)
        if lane_num != 6:
            so.add_result(scenario_to_test, observation['test_setting']['end'])
            continue
        
        if goal[0] > init_pos[0]:
            map_info = get_lane_info1(road_info)
        else:
            map_info = get_lane_info2(road_info)

        try:
            step_count = 0
            max_steps = 5000
            while observation['test_setting']['end'] == -1:
                step_count += 1
                if step_count > max_steps:
                    print(f"超时保护：场景已执行 {max_steps} 步，强制跳过")
                    break

                if goal[0] > init_pos[0]:
                    next_obs = observation_to_state1_simRL_proper(observation, goal, map_info)
                else:
                    next_obs = observation_to_state2_simRL_proper(observation, goal, map_info)

                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    action_seq = model(next_obs_tensor)

                for i in range(5):
                    if observation['test_setting']['end'] != -1:
                        break
                    action_step = action_seq[0, i, :].cpu().numpy()
                    action_step[0] *= 6
                    action_step[1] *= 0.15
                    observation = envs.step(action_step)

        except Exception as e:
            print(f"场景处理出错: {e}")

        finally:
            so.add_result(scenario_to_test, observation['test_setting']['end'])

if __name__ == "__main__":
    # 1. 定义所有实验的超参数配置 (根据你的 Vanilla RNN 训练配置修改)
    experiments = [
        {"name": "Exp-RNN", "embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "bs": 1024},
    ]

    datasets = {
        "B": "/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_B",
        "C": "/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_C"
    }

    checkpoints_dir = "/root/autodl-tmp/BCRL/bc/RNN_checkpoints" # 请根据实际路径修改

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"🚀 开始处理实验: {exp['name']}")
            
        # model_pattern = os.path.join(checkpoints_dir, rnn_trajectory_model_0422_1024BSIZE_256HDdim_2layers_150es_CoAnWarmRest_zDATASET.pth)
        # model_files = glob.glob(model_pattern)
            
        # if not model_files:
        #     print(f"⚠️ 未找到 {exp['name']} 的模型文件，跳过...")
        #     continue
                
        # MODEL_PATH = model_files[0]
        # print(f"📂 找到模型文件: {MODEL_PATH}")

        model_filename = "rnn_trajectory_model_0422_1024BSIZE_256HDdim_2layers_150es_CoAnWarmRest_zDATASET.pth"
        MODEL_PATH = os.path.join(checkpoints_dir, model_filename)
            
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ 未找到模型文件: {MODEL_PATH}，跳过...")
            continue
                
        print(f"📂 找到模型文件: {MODEL_PATH}")

        model = VanillaRNNTrajectoryPredictor(
            embed_dim=exp['embed_dim'], hidden_dim=exp['hidden_dim'], 
            output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
            num_layers=exp['num_layers'], dropout=0.0, nonlinearity='tanh'
        ).to(DEVICE)
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() 
        print(f"✅ 成功加载 {exp['name']} 模型权重")

        for data_name, input_dir in datasets.items():
            print(f"  -> 正在测试数据集: inputs_{data_name}")
            output_dir = f"/root/autodl-tmp/BCRL/bc/planner/outputs/{exp['name']}_r{data_name}_{RUN_DATE_TIME}_{exp['bs']}BS_{exp['embed_dim']}ed_{exp['hidden_dim']}hd_{exp['num_layers']}nl"
            os.makedirs(output_dir, exist_ok=True)

            try:
                process_input_directory(input_dir, output_dir, model)
            except Exception as e:
                print(f"❌ 测试 {exp['name']} 在 inputs_{data_name} 时发生错误: {e}")

        print(f"🎉 实验 {exp['name']} 测试完成！")