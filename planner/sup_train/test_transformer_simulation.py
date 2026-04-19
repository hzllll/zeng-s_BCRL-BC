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

# 添加上级目录以导入 onsite 包
# sys.path.append('../..')

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (bc/) 的路径：当前目录 -> 上一级(planner) -> 上一级(bc)
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
# 新增代码：确保能找到同目录下的 test_conf.py
# sys.path.append(current_dir) 

from onsite import scenarioOrganizer, env

# ================= 配置区域 =================
# 请确保这些参数与训练时的配置完全一致！
# EXP_NAME = "Exp-1"  # 每次跑之前改一下这里和下面的参数
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")
# D_MODEL = 128
# FFN_DIM = 4 * D_MODEL
# num_encoder_layers = 2
# num_decoder_layers = 2
# BATCH_SIZE = 1024
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径 (请修改为你训练出来的pth文件路径)
# MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Transformer_trajectory_model_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_zDATASET.pth"
# MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Transformer_trajectory_model_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_40es_zDATASET.pth"
# MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Transformer_trajectory_model_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_40es_zDATASET_0.05wucos_275epo.pth"
# MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Tf_trajectory_model_0328_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_100es_CoAnWarmRest_zDATASET.pth"
# MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth"
MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Exp-1_Tf_trajectory_model_0414_1024BSIZE_128dmodel_512FFNdim_enc2_dec2_300es_CoAnWarmRest_zDATASET.pth"
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


if __name__ == "__main__":
    # 1. 定义所有实验的超参数配置
    experiments = [
        {"name": "Exp-1", "d_model": 128, "ffn_dim": 512,  "enc": 2, "dec": 2, "bs": 1024},
        {"name": "Exp-2", "d_model": 512, "ffn_dim": 2048, "enc": 4, "dec": 4, "bs": 1024}, 
        {"name": "Exp-3", "d_model": 256, "ffn_dim": 1024, "enc": 3, "dec": 3, "bs": 256},
        {"name": "Exp-4", "d_model": 256, "ffn_dim": 1024, "enc": 3, "dec": 3, "bs": 1024},
        {"name": "Exp-5", "d_model": 256, "ffn_dim": 1024, "enc": 3, "dec": 3, "bs": 1024},
    ]

    datasets = {
        "B": "/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_B",
        "C": "/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_C"
    }

    checkpoints_dir = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints"

    # 提前开启进程池，避免在循环中反复创建销毁引发 CUDA 报错
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for exp in experiments:
            print(f"\n{'='*50}")
            print(f"🚀 开始处理实验: {exp['name']}")
            
            # 【修复问题2】：强制匹配 _Tf 开头，完美避开 Exp-2_v2，精准定位 Exp-2 第一版
            model_pattern = os.path.join(checkpoints_dir, f"{exp['name']}_Tf*.pth")
            model_files = glob.glob(model_pattern)
            
            if not model_files:
                print(f"⚠️ 未找到 {exp['name']} 的模型文件，跳过...")
                continue
                
            MODEL_PATH = model_files[0]
            print(f"📂 找到模型文件: {MODEL_PATH}")

            model = TransformerTrajectoryPredictor(
                d_model=exp['d_model'], output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
                nhead=8, num_encoder_layers=exp['enc'], num_decoder_layers=exp['dec'],
                dim_feedforward=exp['ffn_dim'], dropout=0.0
            ).to(DEVICE)
            
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval() 
            print(f"✅ 成功加载 {exp['name']} 模型权重")

            for data_name, input_dir in datasets.items():
                print(f"  -> 正在测试数据集: inputs_{data_name}")
                output_dir = f"/root/autodl-tmp/BCRL/bc/planner/outputs/{exp['name']}_r{data_name}_{RUN_DATE_TIME}_{exp['bs']}BS_{exp['d_model']}dm_{exp['ffn_dim']}FFN_ednc{exp['enc']}_CoAnWarmRest_zDATASET"
                os.makedirs(output_dir, exist_ok=True)
                
                # 【修复问题3】：复用外层的 executor，直接提交任务
                future = executor.submit(process_input_directory, input_dir, output_dir, model)
                try:
                    future.result() # 等待当前数据集跑完再跑下一个
                except Exception as e:
                    print(f"❌ 测试 {exp['name']} 在 inputs_{data_name} 时发生错误: {e}")
                            
            print(f"🎉 实验 {exp['name']} 测试完成！")



# if __name__ == "__main__":
#     # 初始化模型
#     model = TransformerTrajectoryPredictor(
#         d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
#         nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
#         dim_feedforward=FFN_DIM, dropout=0.0
#     ).to(DEVICE)
    
#     # 加载权重
#     if os.path.exists(MODEL_PATH):
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#         print(f"成功加载 Transformer 模型: {MODEL_PATH}")
#     else:
#         print(f"错误：找不到模型文件 {MODEL_PATH}")
#         exit()
        
#     model.eval() # 开启验证模式

#     # 设置输入输出目录 (请修改为你的实际路径)
#     # input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_test_multi/inputs{i}" for i in range(5)]
#     # output_dir = "/root/autodl-tmp/BCRL/bc/planner/outputs/Transformer_test_result"
    
#     input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_C"]
#     output_dir = f"/root/autodl-tmp/BCRL/bc/planner/outputs/{EXP_NAME}_rC_{RUN_DATE_TIME}_{str(BATCH_SIZE)}BS_{str(D_MODEL)}dm_{str(FFN_DIM)}FFN_ednc{num_encoder_layers}_CoAnWarmRest_zDATASET"

#     # input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_B"]
#     # output_dir = "/root/autodl-tmp/BCRL/bc/planner/outputs/{EXP_NAME}_rB_{RUN_DATE_TIME}_{str(BATCH_SIZE)}BS_{str(D_MODEL)}dm_{str(FFN_DIM)}FFN_ednc{num_encoder_layers}_CoAnWarmRest_zDATASET"

#     # input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_B"]
#     # output_dir = "/root/autodl-tmp/BCRL/bc/planner/outputs/TF_resultB_1024BSIZE_256DMODEL_1024FFNdim_500ESTOP_512epo_CoAnWarmRest_zDATASET"
   
#     # 并行执行
#     with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
#         futures = [executor.submit(process_input_directory, input_dir, output_dir, model) for input_dir in input_dirs]
#         for future in concurrent.futures.as_completed(futures):
#             future.result()