import numpy as np
import torch
import torch.nn as nn
import os
import sys
import concurrent.futures
import math

# 获取项目根目录 (bc/) 的路径：当前目录 -> 上一级(planner) -> 上一级(bc)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from onsite import scenarioOrganizer, env

# ================= 配置区域 =================
# 请确保这些参数与你最新训练时的配置完全一致！
# D_MODEL = 128            # 建议与训练脚本对齐 (如果你训练用的是256，请改回256)
D_MODEL = 256
FFN_DIM = 4 * D_MODEL
num_encoder_layers = 3
num_decoder_layers = 3
OUTPUT_DIM = 2
HIST_SEQ_LENGTH = 5      # 历史帧数
PRED_SEQ_LENGTH = 5      # 预测帧数
FRAME_DIM = 74           # 单帧特征维度 (10 + 8*8)
CAR_NUM = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径 (请修改为你最新训练出来的带有历史信息的 pth 文件路径)
MODEL_PATH = "/root/autodl-tmp/BCRL/bc/Transformer_checkpoints_his5/TF_Hist5_1024B_256d_enc3.pth"
# ===========================================

# 1. 定义位置编码与 Transformer 模型 (必须与最新训练代码完全一致)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, d_model, output_dim, hist_seq_len, pred_seq_len, frame_dim,
                 nhead=8, num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.0): # 验证时 dropout 设为 0
        super().__init__()
        self.hist_seq_len = hist_seq_len
        self.pred_seq_len = pred_seq_len
        
        self.frame_embedding = nn.Linear(frame_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=hist_seq_len)
        self.dropout_embed = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.future_queries = nn.Parameter(torch.randn(pred_seq_len, d_model))

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # x shape: (Batch_Size, 370)
        B = x.size(0)
        x_seq = x.view(B, self.hist_seq_len, -1)  # (B, 5, 74)
        
        tokens = self.frame_embedding(x_seq)
        tokens = torch.tanh(tokens)
        tokens = self.pos_encoder(tokens)
        tokens = self.dropout_embed(tokens)

        memory = self.encoder(tokens)

        tgt = self.future_queries.unsqueeze(0).expand(B, -1, -1)
        dec_out = self.decoder(tgt=tgt, memory=memory)

        pred = self.head(dec_out)
        return pred

# 2. 特征提取函数 (完全适配 74 维单帧特征)
def extract_frame_features(observation, goal, map_info, is_negative_dir=False):
    """
    提取单帧 74 维特征 (自车10维 + 8辆障碍车*8维)
    """
    car_num = 8
    states = []
    
    # 解析所有车辆信息
    frame_data = []
    keys = list(observation['vehicle_info'].keys())
    for key in keys:
        val = observation['vehicle_info'][key]
        # 如果环境中没有 width，默认给 1.8
        frame_data.append([
            val['x'], val['y'], val['v'], val['a'], 
            val['yaw'], val['length'], val.get('width', 1.8)
        ])
    state = np.array(frame_data)

    # 处理坐标系反转 (针对反向行驶场景)
    if is_negative_dir:
        state[:, 0] = -state[:, 0]
        state[:, 1] = -state[:, 1]
        state[:, 4] = state[:, 4] - np.pi
        goal = [-goal[0], -goal[1]]
        map_info_used = {}
        for j in range(3):
            map_info_used[j] = {
                'left_bound': -map_info[j]['left_bound'],
                'center': -map_info[j]['center'],
                'right_bound': -map_info[j]['right_bound']
            }
    else:
        map_info_used = map_info

    # --- 1. 提取主车特征 (10维) ---
    ego_x, ego_y, ego_v, _, ego_yaw, ego_len, ego_width = state[0]
    ego_yaw = (ego_yaw + np.pi) % (2 * np.pi) - np.pi
    
    # 判断主车所在车道边界
    if map_info_used[0]['right_bound'] <= ego_y:
        ego_upper = map_info_used[0]['left_bound']
        ego_lower = map_info_used[0]['right_bound']
    elif map_info_used[1]['right_bound'] < ego_y < map_info_used[1]['left_bound']:
        ego_upper = map_info_used[1]['left_bound']
        ego_lower = map_info_used[1]['right_bound']
    else:
        ego_upper = map_info_used[2]['left_bound']
        ego_lower = map_info_used[2]['right_bound']

    states.extend([
        ego_v, ego_yaw, ego_x, ego_y, 
        goal[0] - ego_x, goal[1] - ego_y, 
        ego_len, ego_width, 
        ego_upper - ego_y, ego_lower - ego_y
    ])

    # --- 2. 提取障碍车特征 (8维 * 8辆) ---
    distances = []
    for j in range(1, len(state)):
        if abs(state[j, 1] - ego_y) > 6: continue # 横向太远忽略
        
        obs_x, obs_y, obs_len = state[j, 0], state[j, 1], state[j, 5]
        # 计算纵向距离 (考虑车长)
        if obs_x - ego_x - 0.5 * (obs_len + ego_len) > 0:
            dist_x = obs_x - ego_x - 0.5 * (obs_len + ego_len)
        elif obs_x - ego_x + 0.5 * (obs_len + ego_len) < 0:
            dist_x = obs_x - ego_x + 0.5 * (obs_len + ego_len)
        else:
            dist_x = 0.0
            
        dist = np.sqrt(dist_x**2 + (obs_y - ego_y)**2)
        if dist < 200:
            distances.append((j, dist, dist_x))

    distances.sort(key=lambda x: x[1])
    distances = distances[:car_num]

    for j, _, lon_dist in distances:
        obs_x, obs_y, obs_v, _, obs_yaw, obs_len, obs_width = state[j]
        obs_yaw = (obs_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 判断障碍车所在车道边界
        if map_info_used[0]['right_bound'] <= obs_y:
            obs_upper = map_info_used[0]['left_bound']
            obs_lower = map_info_used[0]['right_bound']
        elif map_info_used[1]['right_bound'] < obs_y < map_info_used[1]['left_bound']:
            obs_upper = map_info_used[1]['left_bound']
            obs_lower = map_info_used[1]['right_bound']
        else:
            obs_upper = map_info_used[2]['left_bound']
            obs_lower = map_info_used[2]['right_bound']

        states.extend([
            lon_dist,             # longitudinal_dist
            obs_y - ego_y,        # y_j - ego_y
            obs_v - ego_v,        # v_j - ego_v
            obs_yaw,              # yaw_j
            obs_len,              # len_j
            obs_width,            # width_j
            obs_upper - obs_y,    # veh_upper - y_j
            obs_lower - obs_y     # veh_lower - y_j
        ])

    # 补全不足的车辆 (每辆车补 8 维 0/默认值)
    for _ in range(car_num - len(distances)):
        # 默认距离200，其余为0
        states.extend([200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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


def init_history_buffer(observation, goal, map_info, is_negative_dir, hist_seq_len=5, dt=0.1):
    """
    用速度向历史反推，初始化历史 Buffer，
    解决复制同一帧导致的绝对坐标矛盾问题。
    """
    ego_v_raw = observation['vehicle_info']['ego']['v']
    ego_yaw_raw = observation['vehicle_info']['ego']['yaw']
    
    # 归一化 yaw
    ego_yaw_norm = (ego_yaw_raw + np.pi) % (2 * np.pi) - np.pi
    
    # 坐标系翻转时，yaw 也需对应处理（与 extract_frame_features 一致）
    if is_negative_dir:
        yaw_used = ego_yaw_norm - np.pi
        yaw_used = (yaw_used + np.pi) % (2 * np.pi) - np.pi
    else:
        yaw_used = ego_yaw_norm
    
    # 提取当前帧（= 最新帧，对应 buffer 末尾）
    current_frame = np.array(
        extract_frame_features(observation, goal, map_info, is_negative_dir),
        dtype=np.float32
    )
    
    history_buffer = []
    # i=hist_seq_len-1 对应最老帧，i=0 对应当前帧
    for i in range(hist_seq_len - 1, -1, -1):
        past_frame = current_frame.copy()
        if i > 0:
            # 反推 i 步前的位置偏移量（假设速度恒定）
            offset_x = ego_v_raw * i * dt * np.cos(yaw_used)
            offset_y = ego_v_raw * i * dt * np.sin(yaw_used)
            
            # 修正 ego 特征中受绝对坐标影响的各维度
            past_frame[2] -= offset_x          # ego_x：过去的 x 更小（还没走到这）
            past_frame[3] -= offset_y          # ego_y：过去的 y 更小
            past_frame[4] += offset_x          # rel_goal_x：过去离目标更远
            past_frame[5] += offset_y          # rel_goal_y
            past_frame[8] += offset_y          # ego_upper - ego_y：过去 ego_y 更小，距上边界更大
            past_frame[9] += offset_y          # ego_lower - ego_y
            
            # 修正每辆障碍车的 y_j - ego_y（ego_y 过去更小，相对距离更大）
            # 障碍车特征起始位置：index 10，每辆8维，第1维（offset=1）是 y_j - ego_y
            for k in range(8):
                obs_feat_start = 10 + k * 8
                past_frame[obs_feat_start + 1] += offset_y  # y_j - ego_y
                # veh_upper - y_j, veh_lower - y_j 不受 ego 移动影响，无需修正
        
        history_buffer.append(past_frame)
    
    # history_buffer[0] = 最老帧(t-4), history_buffer[-1] = 当前帧(t)
    return [frame.tolist() for frame in history_buffer]

# 3. 核心处理流程 (引入历史轨迹 Buffer)
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
        
        is_negative_dir = goal[0] < init_pos[0]
        map_info = get_lane_info2(road_info) if is_negative_dir else get_lane_info1(road_info)

        try:
            step_count = 0
            max_steps = 5000
            
            # --- 初始化历史轨迹 Buffer ---
            # 初始时刻，用第 0 帧的状态复制 5 份，填满历史队列
            # init_state = extract_frame_features(observation, goal, map_info, is_negative_dir)
            # history_buffer = [init_state for _ in range(HIST_SEQ_LENGTH)]

            history_buffer = init_history_buffer(observation, goal, map_info, is_negative_dir, 
                                     hist_seq_len=HIST_SEQ_LENGTH)
            
            while observation['test_setting']['end'] == -1:
                step_count += 1
                if step_count > max_steps:
                    print(f"超时保护：场景已执行 {max_steps} 步，强制跳过")
                    break

                # 拼接 5 帧历史特征 -> 370 维
                model_input = np.concatenate(history_buffer)
                model_input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # 模型预测未来 5 帧
                with torch.no_grad():
                    action_seq = model(model_input_tensor) # shape: (1, 5, 2)

                # 依次执行预测出的 5 帧动作
                # for i in range(5):
                for i in range(1):  # RHC：只执行第一帧，下一步重新规划
                    if observation['test_setting']['end'] != -1:
                        break
                    
                    # 提取动作并反归一化
                    action_step = action_seq[0, i, :].cpu().numpy()
                    action_step[0] *= 6     # 加速度反归一化
                    action_step[1] *= 0.15  # 转角反归一化
                    
                    # 环境执行一步
                    observation = envs.step(action_step)
                    
                    # --- 关键：更新历史轨迹 Buffer ---
                    # 每执行一步，获取最新环境状态，推入队列，并弹出最老的一帧
                    if observation['test_setting']['end'] == -1:
                        new_state = extract_frame_features(observation, goal, map_info, is_negative_dir)
                        history_buffer.pop(0)
                        history_buffer.append(new_state)

        except Exception as e:
            print(f"场景处理出错: {e}")

        finally:
            so.add_result(scenario_to_test, observation['test_setting']['end'])

if __name__ == "__main__":
    # 初始化模型
    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, 
        hist_seq_len=HIST_SEQ_LENGTH, pred_seq_len=PRED_SEQ_LENGTH, frame_dim=FRAME_DIM,
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
    # input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_C"]
    # output_dir = f"/root/autodl-tmp/BCRL/bc/planner/outputs/Transformer_result_C_Hist5_{D_MODEL}d_enc{num_encoder_layers}"
    input_dirs = [f"/root/autodl-tmp/BCRL/bc/planner/inputs/inputs_C"]
    output_dir = f"/root/autodl-tmp/BCRL/bc/planner/outputs/Transformer_result_C_Hist5_{D_MODEL}d_enc{num_encoder_layers}"
    
    
    # 并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_input_directory, input_dir, output_dir, model) for input_dir in input_dirs]
        for future in concurrent.futures.as_completed(futures):
            future.result()