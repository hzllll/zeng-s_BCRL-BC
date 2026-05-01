# ... [导入包、配置区域和特征提取函数与上方完全一致] ...
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


# 1. 定义 AR_TF 模型
class AutoregressiveTransformerPredictor(nn.Module):
    def __init__(self, d_model, output_dim, seq_length, car_num, nhead=8, 
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.car_num = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim

        self.embedding_main_target = nn.Linear(6, d_model)
        self.embedding_vehicle = nn.Linear(6, d_model)
        self.dropout_embed = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.action_embedding = nn.Linear(output_dim, d_model)
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh(),
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, target_actions=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.size(0)

        main_target_feat = x[:, 0:6]
        vehicle_feats = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)

        main_emb = self.embedding_main_target(main_target_feat).unsqueeze(1)
        veh_emb = self.embedding_vehicle(vehicle_feats)
        
        tokens = torch.cat([veh_emb, main_emb], dim=1)
        tokens = torch.tanh(tokens)
        tokens = self.dropout_embed(tokens)

        memory = self.encoder(tokens)

        if self.training and target_actions is not None:
            action_embs = self.action_embedding(target_actions)
            sos = self.sos_token.expand(B, -1, -1)
            tgt = torch.cat([sos, action_embs[:, :-1, :]], dim=1)
            tgt_mask = self.generate_square_subsequent_mask(self.seq_length).to(x.device)
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            pred = self.head(dec_out)
            return pred
        else:
            # 【推理/验证模式：逐步自回归生成】
            sos = self.sos_token.expand(B, -1, -1)
            tgt = sos
            preds = []

            for i in range(self.seq_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(x.device)
                dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                
                step_out = dec_out[:, -1, :]
                step_pred = self.head(step_out)
                preds.append(step_pred.unsqueeze(1))
                
                if i < self.seq_length - 1:
                    step_pred_emb = self.action_embedding(step_pred).unsqueeze(1)
                    tgt = torch.cat([tgt, step_pred_emb], dim=1)

            return torch.cat(preds, dim=1)

# ... [特征提取和 process_input_directory 流程与上方完全一致] ...
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
    # 1. 定义所有实验的超参数配置 (根据你的 AR_TF 训练配置修改)
    experiments = [
        {"name": "Exp-ARTF", "d_model": 256, "ffn_dim": 1024, "enc": 3, "dec": 3, "bs": 1024},
    ]

    datasets = {
        "B": "/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_B",
        "C": "/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_C"
    }

    checkpoints_dir = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints" # 请根据实际路径修改

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"🚀 开始处理实验: {exp['name']}")
            
        # model_pattern = os.path.join(checkpoints_dir, Exp-AR-Baseline_AR_model_0422_19_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_250es.pth)
        # model_files = glob.glob(model_pattern)
            
        # if not model_files:
        #     print(f"⚠️ 未找到 {exp['name']} 的模型文件，跳过...")
        #     continue
                
        # MODEL_PATH = model_files[0]
        # print(f"📂 找到模型文件: {MODEL_PATH}")

        model_filename = "Exp-AR-Baseline_AR_model_0422_19_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_250es.pth"
        MODEL_PATH = os.path.join(checkpoints_dir, model_filename)
            
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ 未找到模型文件: {MODEL_PATH}，跳过...")
            continue
                
        print(f"📂 找到模型文件: {MODEL_PATH}")

        model = AutoregressiveTransformerPredictor(
            d_model=exp['d_model'], output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
            nhead=8, num_encoder_layers=exp['enc'], num_decoder_layers=exp['dec'],
            dim_feedforward=exp['ffn_dim'], dropout=0.0
        ).to(DEVICE)
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() 
        print(f"✅ 成功加载 {exp['name']} 模型权重")

        for data_name, input_dir in datasets.items():
            print(f"  -> 正在测试数据集: inputs_{data_name}")
            output_dir = f"/root/autodl-tmp/BCRL/bc/planner/outputs/{exp['name']}_r{data_name}_{RUN_DATE_TIME}_{exp['bs']}BS_{exp['d_model']}dm_{exp['ffn_dim']}FFN_ednc{exp['enc']}"
            os.makedirs(output_dir, exist_ok=True)

            try:
                process_input_directory(input_dir, output_dir, model)
            except Exception as e:
                print(f"❌ 测试 {exp['name']} 在 inputs_{data_name} 时发生错误: {e}")

        print(f"🎉 实验 {exp['name']} 测试完成！")