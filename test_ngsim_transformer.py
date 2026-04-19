"""
NGSIM 公开数据集测试脚本：在真实高速公路轨迹数据上，开环 (open-loop) 评估 Transformer BC 模型

=========================================================================================
|                         什么是 NGSIM 数据集？                                          |
=========================================================================================
| NGSIM (Next Generation Simulation) 是美国联邦公路管理局 (FHWA) 采集的 **真实车辆轨迹   |
| 数据集**。他们在高速公路旁的高楼上架设 8 台同步摄像机，录下了 45 分钟内该路段上 **所有 |
| 车辆** 的运动轨迹，每 0.1 秒记录一次位置、速度、加速度等信息，整理成 CSV 文件。        |
|                                                                                         |
| 包含两个主要路段：                                                                      |
|   - US-101 (好莱坞高速，洛杉矶): 5 条主车道 + 1 条辅助车道, ~640m 路段                |
|   - I-80  (Emeryville, 旧金山):   6 条车道 (含 HOV 车道),   ~500m 路段                |
|                                                                                         |
| NGSIM **不是仿真平台**，没有 env.step()，不需要 onsite 包，也不需要 inputs 文件夹。     |
| 测试方式是 **开环回放 (log-replay)**：回放真实数据 → 模型预测 → 与真实轨迹对比。        |
=========================================================================================

=========================================================================================
|                     与 Onsite / highway-env 测试的异同                                  |
=========================================================================================
| 相同点：                                                                                |
|   1. 加载同一个 .pth 模型权重                                                           |
|   2. 使用同一个 TransformerTrajectoryPredictor 模型定义                                 |
|   3. 构造同样的 54 维输入向量（6维主车 + 8×6维障碍车）                                  |
|   4. 模型输出 (5, 2) = 5 步的 (加速度, 转角) 预测                                      |
|                                                                                         |
| 不同点：                                                                                |
|   - Onsite: 闭环仿真，用 onsite 自带场景 (inputs/*.xosc)，评分用 onsite 评分软件       |
|   - highway-env: 闭环仿真，gymnasium 随机生成场景，评价碰撞率/奖励                     |
|   - NGSIM: **开环回放**，用真实高速公路 CSV 数据，评价轨迹预测误差 (ADE/FDE/RMSE)      |
=========================================================================================

=========================================================================================
|                              NGSIM CSV 字段说明                                        |
=========================================================================================
| Vehicle_ID     : 车辆编号（注意：不同车辆可能复用同一编号！需结合 Frame_ID 区分）       |
| Frame_ID       : 帧编号（每帧间隔 0.1s）                                               |
| Total_Frames   : 该车辆在数据集中出现的总帧数                                          |
| Global_Time    : 全局时间戳（单位：毫秒 ms）                                           |
| Local_X        : **横向位置** (垂直于行驶方向, 从路边量起, 单位：英尺 ft)              |
| Local_Y        : **纵向位置** (沿行驶方向, 单位：英尺 ft)                              |
| Global_X/Y     : 全局坐标 (单位：英尺 ft)                                              |
| v_Length        : 车辆长度 (单位：英尺 ft)                                              |
| v_Width         : 车辆宽度 (单位：英尺 ft)                                              |
| v_Class         : 车辆类型 (1=摩托车, 2=小汽车, 3=卡车)                                 |
| v_Vel           : 瞬时速度 (单位：英尺/秒 ft/s)                                        |
| v_Acc           : 瞬时加速度 (单位：英尺/秒² ft/s²)                                    |
| Lane_ID         : 车道编号 (1=最左侧快车道, 往右递增; 6/7/8=匝道/辅助车道)             |
| Preceding       : 前方车辆的 Vehicle_ID (0=无前车)                                     |
| Following       : 后方车辆的 Vehicle_ID (0=无后车)                                     |
| Space_Headway   : 与前车的车头间距 (单位：英尺 ft)                                      |
| Time_Headway    : 与前车的车头时距 (单位：秒 s)                                         |
=========================================================================================

=========================================================================================
|                     坐标映射 (NGSIM → 模型坐标系)                                      |
=========================================================================================
| NGSIM:  Local_Y = 纵向(沿路), Local_X = 横向(跨路)      单位: 英尺(ft)                |
| 模型:   x = 纵向(行驶方向),   y = 横向(跨路方向)        单位: 米(m)                   |
|                                                                                         |
| 映射关系:  模型 x = NGSIM Local_Y × 0.3048                                             |
|            模型 y = NGSIM Local_X × 0.3048                                             |
|            速度  = NGSIM v_Vel   × 0.3048  (ft/s → m/s)                                |
|            加速度 = NGSIM v_Acc  × 0.3048  (ft/s² → m/s²)                              |
|            长度  = NGSIM v_Length × 0.3048  (ft → m)                                   |
=========================================================================================

用法：
  1. 从 Kaggle 或美国交通部下载 NGSIM CSV 文件:
     - Kaggle:  https://www.kaggle.com/datasets/nigelwilliams/ngsim-vehicle-trajectory-data-us-101
     - 官方:    https://data.transportation.gov/d/8ect-6jqj
  2. 将 CSV 文件放到 ngsim_data/ 文件夹下
  3. 修改下方 NGSIM_CSV_PATH 指向你的 CSV 文件
  4. 运行: python test_ngsim_transformer.py
        - cd /root/autodl-tmp/BCRL/bc
        - python test_ngsim_transformer.py
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        全局配置区域                             ║
# ╚══════════════════════════════════════════════════════════════════╝

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- NGSIM 数据路径 ----------
# 请修改为你下载的 NGSIM CSV 文件路径
# US-101 数据集示例：从 Kaggle 下载后通常名为 "US-101-LosAngeles-CA.csv" 或类似名称
NGSIM_CSV_PATH = os.path.join(BASE_DIR, "ngsim_data", "Next_Generation_Simulation_NGSIM_Vehicle_Trajectories_and_Supporting_Data_20260418.csv")

# ---------- NGSIM 数据集参数 ----------
# 以下参数描述了 NGSIM 数据集的基本属性，一般不需要修改
FT_TO_M = 0.3048            # 英尺 → 米 的转换系数（NGSIM 原始数据全部以英尺为单位）
NGSIM_DT = 0.1              # NGSIM 采样间隔 = 0.1 秒 (10Hz)，与 Onsite 的仿真步长一致
WHEELBASE = 2.5              # 自行车模型的轴距 (m)，用于从航向变化率反推转向角

# ---------- 车道选择参数 ----------
# NGSIM US-101 有 5 条主车道 (Lane_ID 1~5)，我们的模型只支持 3 条车道
# 从中选取 3 条相邻的主车道进行测试
# Lane_ID 含义: 1=最左侧(快车道), 2/3/4=中间车道, 5=最右侧(慢车道), 6/7/8=匝道
TARGET_LANES = [2, 3, 4]     # 选择 Lane 2/3/4 (3条相邻中间车道)
LANE_WIDTH = 3.66            # 标准美国高速车道宽度: 12 英尺 ≈ 3.66 米

# ---------- 测试参数 ----------
# NGSIM 有数万帧数据，我们可以选取大量样本进行评估
MAX_EVAL_SAMPLES = 5000      # 最大评估样本数 (设为 -1 表示全部评估)
PREDICTION_HORIZON = 5       # 预测步长 = 5 步 × 0.1s = 0.5 秒（与模型输出一致）
MIN_TRAJECTORY_LENGTH = 50   # 车辆最少存在帧数，太短的轨迹不适合评估
SMOOTHING_WINDOW = 11        # Savitzky-Golay 滤波窗口长度 (必须为奇数)，用于平滑位置/速度噪声
SMOOTHING_POLY = 3           # Savitzky-Golay 滤波多项式阶数

# ---------- 模型路径 ----------
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth"
)

# ---------- Transformer 模型超参数（必须与训练时完全一致）----------
D_MODEL = 256
FFN_DIM = 4 * D_MODEL        # = 1024
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
OUTPUT_DIM = 2                # (加速度, 转向角)
SEQ_LENGTH = 5                # 预测 5 个时间步
CAR_NUM = 8                   # 最多考虑 8 辆周围车辆

# ---------- 输出路径 ----------
SAVE_DIR = os.path.join(BASE_DIR, "ngsim_results")
os.makedirs(SAVE_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")
LOG_FILE = os.path.join(SAVE_DIR, f"ngsim_test_{TIMESTAMP}.log")
RESULT_FILE = os.path.join(SAVE_DIR, f"ngsim_test_{TIMESTAMP}.txt")
PLOT_FILE = os.path.join(SAVE_DIR, f"ngsim_test_{TIMESTAMP}.svg")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        日志配置                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def setup_logger():
    logger = logging.getLogger("NGSIM_TEST")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '[%(asctime)s] [%(levelname)-7s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

log = setup_logger()


# ╔══════════════════════════════════════════════════════════════════╗
# ║          Transformer 模型定义（与训练代码完全一致）              ║
# ╚══════════════════════════════════════════════════════════════════╝

class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, d_model, output_dim, seq_length, car_num,
                 nhead=8, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.car_num = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim

        self.embedding_main_target = nn.Linear(6, d_model)
        self.embedding_vehicle = nn.Linear(6, d_model)
        self.dropout_embed = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     NGSIM 数据加载与预处理                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_ngsim_csv(csv_path):
    """
    加载 NGSIM CSV 文件并执行单位转换。

    NGSIM 原始数据的单位全是英制: 时间(ms), 距离(ft), 速度(ft/s), 加速度(ft/s²)。
    本函数将其全部转换为国际单位制: 时间(s), 距离(m), 速度(m/s), 加速度(m/s²)。

    返回: 预处理后的 DataFrame
    """
    log.info(f"正在加载 NGSIM 数据: {csv_path}")

    if not os.path.exists(csv_path):
        log.error(f"找不到 NGSIM 数据文件: {csv_path}")
        log.error("请先下载 NGSIM 数据集:")
        log.error("  Kaggle:  https://www.kaggle.com/datasets/nigelwilliams/ngsim-vehicle-trajectory-data-us-101")
        log.error("  官方:    https://data.transportation.gov/d/8ect-6jqj")
        log.error(f"下载后将 CSV 文件放到: {os.path.dirname(csv_path)}")
        sys.exit(1)

    usecols = [
        'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
        'Local_X', 'Local_Y', 'v_Length', 'v_Width',
        'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID',
        'Preceding', 'Following', 'Space_Headway',
    ]

    try:
        df = pd.read_csv(csv_path, usecols=usecols)
    except ValueError:
        log.warning("部分列名不存在，尝试加载全部列...")
        df = pd.read_csv(csv_path)
        log.info(f"CSV 文件中的所有列名: {list(df.columns)}")
        # 尝试对列名做 strip
        df.columns = df.columns.str.strip()

    log.info(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    log.info(f"CSV 列名: {list(df.columns)}")

    # ---------- 单位转换 ----------
    # 距离: ft → m
    for col in ['Local_X', 'Local_Y', 'v_Length', 'v_Width']:
        if col in df.columns:
            df[col] = df[col] * FT_TO_M

    if 'Space_Headway' in df.columns:
        df['Space_Headway'] = df['Space_Headway'] * FT_TO_M

    # 速度: ft/s → m/s
    if 'v_Vel' in df.columns:
        df['v_Vel'] = df['v_Vel'] * FT_TO_M

    # 加速度: ft/s² → m/s²
    if 'v_Acc' in df.columns:
        df['v_Acc'] = df['v_Acc'] * FT_TO_M

    # 时间: ms → s
    if 'Global_Time' in df.columns:
        df['Global_Time'] = df['Global_Time'] / 1000.0

    df = df.sort_values(by=['Vehicle_ID', 'Frame_ID']).reset_index(drop=True)

    log.info(f"单位转换完成: 距离(m), 速度(m/s), 加速度(m/s²), 时间(s)")
    log.info(f"Lane_ID 分布: {dict(df['Lane_ID'].value_counts().sort_index())}")

    if 'v_Class' in df.columns:
        log.info(f"车辆类型分布: {dict(df['v_Class'].value_counts().sort_index())}")

    return df


def compute_derived_fields(df):
    """
    计算 NGSIM 数据中不直接提供但模型需要的字段：航向角 (yaw) 和转向角 (steer)。

    航向角 yaw：
      NGSIM 不提供航向角，需要从相邻两帧的位置差分计算：
      yaw = arctan2(dy/dt, dx/dt)
      其中 dx = Local_Y 的变化 (纵向), dy = Local_X 的变化 (横向)

    转向角 steer：
      根据自行车运动学模型从航向变化率反推：
      steer ≈ arctan(wheelbase × d_yaw / (v × dt))

    注意：原始 NGSIM 数据存在噪声，计算前先用 Savitzky-Golay 滤波器平滑位置数据。
    """
    log.info("正在计算航向角(yaw)和转向角(steer)...")

    yaw_all = np.full(len(df), np.nan)
    steer_all = np.full(len(df), np.nan)

    vehicle_ids = df['Vehicle_ID'].unique()
    processed = 0

    for vid in vehicle_ids:
        mask = df['Vehicle_ID'] == vid
        idx = df.index[mask]

        if len(idx) < 5:
            continue

        # 模型坐标: x = Local_Y(纵向), y = Local_X(横向)
        x_raw = df.loc[idx, 'Local_Y'].values
        y_raw = df.loc[idx, 'Local_X'].values
        v_raw = df.loc[idx, 'v_Vel'].values

        # Savitzky-Golay 平滑（减少摄像头提取的位置噪声）
        win = min(SMOOTHING_WINDOW, len(x_raw))
        if win % 2 == 0:
            win -= 1
        if win >= SMOOTHING_POLY + 2:
            x_smooth = savgol_filter(x_raw, win, SMOOTHING_POLY)
            y_smooth = savgol_filter(y_raw, win, SMOOTHING_POLY)
        else:
            x_smooth = x_raw
            y_smooth = y_raw

        dx = np.gradient(x_smooth, NGSIM_DT)
        dy = np.gradient(y_smooth, NGSIM_DT)
        yaw = np.arctan2(dy, dx)

        dyaw = np.gradient(yaw, NGSIM_DT)
        dyaw = (dyaw + np.pi / NGSIM_DT) % (2 * np.pi / NGSIM_DT) - np.pi / NGSIM_DT
        # 对 dyaw 再做一次限幅以减少噪声
        dyaw = np.clip(dyaw, -2.0 / NGSIM_DT, 2.0 / NGSIM_DT)

        v_safe = np.maximum(np.abs(v_raw), 0.5)
        steer = np.arctan(WHEELBASE * dyaw / v_safe)
        steer = np.clip(steer, -0.5, 0.5)

        yaw_all[idx] = yaw
        steer_all[idx] = steer
        processed += 1

    df['yaw'] = yaw_all
    df['steer'] = steer_all

    valid_count = np.sum(~np.isnan(yaw_all))
    log.info(f"航向角/转向角计算完成: 处理 {processed} 辆车, {valid_count}/{len(df)} 行有效")

    return df


def compute_lane_boundaries(df, target_lanes):
    """
    从 NGSIM 数据中统计目标车道的边界位置。

    原理：对每条车道的 Local_X（横向位置）取中位数作为车道中心，
    然后 ± 半个车道宽度得到车道边界。

    返回格式与 Onsite 的 map_info 完全一致：
      map_info[0] = {'left_bound': ..., 'center': ..., 'right_bound': ...}  # y 最大的车道
      map_info[1] = ...
      map_info[2] = ...                                                     # y 最小的车道

    以及 lane1~lane4 (与 Get_Dataset_Closest_Proper_RL.py 一致)：
      lane1 = 最大 y 边界 (上边界)
      lane2 = 上中分界
      lane3 = 中下分界
      lane4 = 最小 y 边界 (下边界)
    """
    log.info(f"正在计算车道边界 (目标车道: {target_lanes})...")

    lane_centers = {}
    for lane_id in target_lanes:
        lane_data = df[df['Lane_ID'] == lane_id]
        if len(lane_data) == 0:
            log.warning(f"  Lane_ID={lane_id} 无数据!")
            continue
        center = lane_data['Local_X'].median()
        lane_centers[lane_id] = center
        log.info(f"  Lane_ID={lane_id}: 中心 Local_X = {center:.2f} m, 数据量 = {len(lane_data)}")

    if len(lane_centers) < 3:
        log.error("有效车道不足 3 条，请检查 TARGET_LANES 设置或 CSV 文件内容")
        sys.exit(1)

    sorted_lanes = sorted(lane_centers.items(), key=lambda x: x[1], reverse=True)

    map_info = {}
    for i, (lane_id, center) in enumerate(sorted_lanes[:3]):
        map_info[i] = {
            'left_bound': center + LANE_WIDTH / 2,
            'center': center,
            'right_bound': center - LANE_WIDTH / 2,
        }
        log.info(
            f"  map_info[{i}] (Lane_ID={lane_id}): "
            f"left={map_info[i]['left_bound']:.2f}, "
            f"center={map_info[i]['center']:.2f}, "
            f"right={map_info[i]['right_bound']:.2f}"
        )

    lane1 = map_info[0]['left_bound']
    lane2 = map_info[0]['right_bound']
    lane3 = map_info[1]['right_bound']
    lane4 = map_info[2]['right_bound']

    log.info(f"  lane1(上)={lane1:.2f}, lane2={lane2:.2f}, lane3={lane3:.2f}, lane4(下)={lane4:.2f}")

    return map_info, lane1, lane2, lane3, lane4


# ╔══════════════════════════════════════════════════════════════════╗
# ║          构造模型输入（54维，与 Onsite 训练数据完全一致）        ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_model_input(ego_row, same_frame_others, goal_x, goal_y,
                      map_info, lane1, lane2, lane3, lane4):
    """
    从 NGSIM 单帧数据构造 Transformer 模型的 54 维输入向量。

    输入格式与 Onsite 训练时 (Get_Dataset_Closest_Proper_RL.py) 完全一致：
      [0:6]   主车+目标点: [v, yaw, goal_dx, goal_dy, lane_upper-ego_y, lane_lower-ego_y]
      [6:54]  8辆障碍车 × 6维: [rel_x, rel_y, rel_v, yaw, lane_left-ego_y, lane_right-ego_y]

    参数:
      ego_row: 主车当前帧的 DataFrame 行
      same_frame_others: 同一帧中其他车辆的 DataFrame
      goal_x, goal_y: 目标点坐标 (模型坐标系)
      map_info, lane1~4: 车道边界信息
    """
    car_num = CAR_NUM

    # 模型坐标: x=Local_Y(纵向), y=Local_X(横向)
    ego_x = ego_row['Local_Y']
    ego_y = ego_row['Local_X']
    ego_v = ego_row['v_Vel']
    ego_yaw = ego_row['yaw']
    ego_length = ego_row['v_Length']

    states = []

    # ---- 主车特征 (6维) ----
    states.append(ego_v)
    states.append((ego_yaw + np.pi) % (2 * np.pi) - np.pi)
    states.append(goal_x - ego_x)
    states.append(goal_y - ego_y)
    states.append(lane1 - ego_y)
    states.append(lane4 - ego_y)

    # ---- 周围车辆处理 ----
    distances = []
    for _, other in same_frame_others.iterrows():
        other_y_model = other['Local_X']
        other_x_model = other['Local_Y']

        if abs(other_y_model - ego_y) > 6:
            continue

        other_length = other['v_Length']
        half_len = 0.5 * (other_length + ego_length)
        dx = other_x_model - ego_x

        if dx - half_len > 0:
            rel_x = dx - half_len
        elif dx + half_len < 0:
            rel_x = dx + half_len
        else:
            rel_x = 0.0

        dist = np.sqrt(rel_x ** 2 + (other_y_model - ego_y) ** 2)
        if dist < 200:
            distances.append({
                'rel_x': rel_x,
                'rel_y': other_y_model - ego_y,
                'rel_v': other['v_Vel'] - ego_v,
                'yaw': other['yaw'] if not np.isnan(other['yaw']) else 0.0,
                'y_pos': other_y_model,
                'dist': dist,
            })

    distances.sort(key=lambda d: d['dist'])
    selected = distances[:car_num]

    for d in selected:
        states.append(d['rel_x'])
        states.append(d['rel_y'])
        states.append(d['rel_v'])
        states.append((d['yaw'] + np.pi) % (2 * np.pi) - np.pi)

        vy = d['y_pos']
        if vy >= lane2:
            states.append(lane1 - ego_y)
            states.append(lane2 - ego_y)
        elif lane3 < vy < lane2:
            states.append(lane2 - ego_y)
            states.append(lane3 - ego_y)
        else:
            states.append(lane3 - ego_y)
            states.append(lane4 - ego_y)

    # 补全不足 8 辆的占位
    for _ in range(car_num - len(selected)):
        states.extend([200.0, 0.0, 0.0, 0.0])
        if ego_y >= lane2:
            states.append(lane1 - ego_y)
            states.append(lane2 - ego_y)
        elif lane3 < ego_y < lane2:
            states.append(lane2 - ego_y)
            states.append(lane3 - ego_y)
        else:
            states.append(lane3 - ego_y)
            states.append(lane4 - ego_y)

    return np.array(states, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    开环评估核心逻辑                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_openloop_evaluation(df, model, map_info, lane1, lane2, lane3, lane4):
    """
    NGSIM 开环评估 (Open-loop / Log-replay):

    核心思路：
      1. 筛选目标车道上、轨迹足够长的车辆
      2. 对每辆车的每个有效时刻：
         a. 用当前帧的真实数据构造 54 维输入
         b. 送入模型，得到未来 5 步的预测 (acc, steer)
         c. 用简单运动学模型积分出预测轨迹
         d. 与 NGSIM 中该车未来 5 步的真实轨迹对比
      3. 汇总所有样本的误差指标

    评估指标：
      ADE (Average Displacement Error): 预测轨迹 5 步内与真实轨迹的平均位置偏差 (m)
      FDE (Final Displacement Error):   第 5 步时预测位置与真实位置的偏差 (m)
      RMSE_acc:  预测加速度与真实加速度的均方根误差 (m/s²)
      RMSE_steer: 预测转向角与真实转向角的均方根误差 (rad)
      RMSE_lat:  横向位置误差的均方根 (m)，体现车道保持能力
      RMSE_lon:  纵向位置误差的均方根 (m)，体现速度控制能力
    """
    log.info("=" * 70)
    log.info("开始 NGSIM 开环评估 (Open-loop Evaluation)")
    log.info("=" * 70)

    # 只保留目标车道上的车辆
    df_target = df[df['Lane_ID'].isin(TARGET_LANES)].copy()
    log.info(f"目标车道 {TARGET_LANES} 的数据量: {len(df_target)} 行")

    # 筛选轨迹足够长的车辆
    veh_counts = df_target.groupby('Vehicle_ID').size()
    valid_vehicles = veh_counts[veh_counts >= MIN_TRAJECTORY_LENGTH].index.tolist()
    log.info(f"轨迹长度 >= {MIN_TRAJECTORY_LENGTH} 帧的车辆: {len(valid_vehicles)} 辆")

    # 按帧分组，方便后续查找同一时刻的其他车辆
    frame_groups = df_target.groupby('Frame_ID')

    all_ade = []
    all_fde = []
    all_rmse_acc = []
    all_rmse_steer = []
    all_rmse_lat = []
    all_rmse_lon = []
    all_inference_ms = []

    sample_count = 0
    skip_nan = 0
    skip_short = 0
    skip_lane_change = 0

    eval_start_time = time.time()

    for vid_idx, vid in enumerate(valid_vehicles):
        veh_data = df_target[df_target['Vehicle_ID'] == vid].sort_values('Frame_ID')

        if len(veh_data) < MIN_TRAJECTORY_LENGTH:
            continue

        frames = veh_data['Frame_ID'].values
        indices = veh_data.index.values

        # 目标点: 使用该车轨迹终点（模型坐标系）
        goal_x = veh_data.iloc[-1]['Local_Y']
        goal_y = veh_data.iloc[-1]['Local_X']

        # 每隔 PREDICTION_HORIZON 帧采样一次（避免重叠评估）
        for t_idx in range(0, len(veh_data) - PREDICTION_HORIZON, PREDICTION_HORIZON):
            if 0 < MAX_EVAL_SAMPLES <= sample_count:
                break

            current_row = veh_data.iloc[t_idx]
            current_frame = current_row['Frame_ID']

            # 检查未来 5 帧是否都存在（车辆可能在中途离开视野）
            future_frames_needed = [current_frame + k for k in range(1, PREDICTION_HORIZON + 1)]
            future_data = veh_data[veh_data['Frame_ID'].isin(future_frames_needed)]
            if len(future_data) < PREDICTION_HORIZON:
                skip_short += 1
                continue

            future_data = future_data.sort_values('Frame_ID')

            # 检查未来帧中车辆是否一直在目标车道（排除换道过程中的数据）
            if not future_data['Lane_ID'].isin(TARGET_LANES).all():
                skip_lane_change += 1
                continue

            # 检查 yaw 是否有效
            if np.isnan(current_row['yaw']):
                skip_nan += 1
                continue

            # 获取同帧的其他车辆（作为周围障碍车）
            if current_frame in frame_groups.groups:
                same_frame = frame_groups.get_group(current_frame)
                others = same_frame[same_frame['Vehicle_ID'] != vid]
            else:
                others = pd.DataFrame()

            # ---------- 构造模型输入 ----------
            model_input = build_model_input(
                current_row, others, goal_x, goal_y,
                map_info, lane1, lane2, lane3, lane4
            )

            if len(model_input) != 54:
                log.warning(f"输入维度异常: {len(model_input)}, 跳过")
                continue

            input_tensor = torch.tensor(model_input).unsqueeze(0).to(DEVICE)

            # ---------- 模型推理 ----------
            tic = time.perf_counter()
            with torch.no_grad():
                pred_seq = model(input_tensor)
            toc = time.perf_counter()
            all_inference_ms.append((toc - tic) * 1000)

            pred = pred_seq[0].cpu().numpy()  # (5, 2)
            pred_acc = pred[:, 0] * 6.0       # 反归一化加速度
            pred_steer = pred[:, 1] * 0.15    # 反归一化转向角

            # ---------- 真实值 ----------
            gt_acc = future_data['v_Acc'].values[:PREDICTION_HORIZON]
            gt_steer = future_data['steer'].values[:PREDICTION_HORIZON]
            gt_x = future_data['Local_Y'].values[:PREDICTION_HORIZON]   # 真实纵向 (模型x)
            gt_y = future_data['Local_X'].values[:PREDICTION_HORIZON]   # 真实横向 (模型y)

            if np.any(np.isnan(gt_steer)):
                skip_nan += 1
                continue

            # ---------- 积分预测轨迹 ----------
            ego_x = current_row['Local_Y']
            ego_y_pos = current_row['Local_X']
            cur_v = current_row['v_Vel']
            cur_yaw = current_row['yaw']

            pred_xs = []
            pred_ys = []
            for step in range(PREDICTION_HORIZON):
                cur_v = max(0.0, cur_v + pred_acc[step] * NGSIM_DT)
                cur_yaw = cur_yaw + pred_steer[step] * NGSIM_DT
                ego_x = ego_x + cur_v * np.cos(cur_yaw) * NGSIM_DT
                ego_y_pos = ego_y_pos + cur_v * np.sin(cur_yaw) * NGSIM_DT
                pred_xs.append(ego_x)
                pred_ys.append(ego_y_pos)

            pred_xs = np.array(pred_xs)
            pred_ys = np.array(pred_ys)

            # ---------- 计算指标 ----------
            displacements = np.sqrt((pred_xs - gt_x) ** 2 + (pred_ys - gt_y) ** 2)
            ade = np.mean(displacements)
            fde = displacements[-1]
            rmse_acc = np.sqrt(np.mean((pred_acc - gt_acc) ** 2))
            rmse_steer = np.sqrt(np.mean((pred_steer - gt_steer) ** 2))
            rmse_lat = np.sqrt(np.mean((pred_ys - gt_y) ** 2))
            rmse_lon = np.sqrt(np.mean((pred_xs - gt_x) ** 2))

            all_ade.append(ade)
            all_fde.append(fde)
            all_rmse_acc.append(rmse_acc)
            all_rmse_steer.append(rmse_steer)
            all_rmse_lat.append(rmse_lat)
            all_rmse_lon.append(rmse_lon)

            sample_count += 1

            if sample_count % 500 == 0:
                elapsed = time.time() - eval_start_time
                log.info(
                    f"  已评估 {sample_count} 样本 | "
                    f"ADE={np.mean(all_ade):.4f}m | "
                    f"FDE={np.mean(all_fde):.4f}m | "
                    f"耗时 {elapsed:.1f}s"
                )

        if 0 < MAX_EVAL_SAMPLES <= sample_count:
            log.info(f"已达最大评估样本数 {MAX_EVAL_SAMPLES}，停止评估")
            break

    total_time = time.time() - eval_start_time

    log.info(f"\n评估完成:")
    log.info(f"  有效样本数: {sample_count}")
    log.info(f"  跳过(yaw/steer NaN): {skip_nan}")
    log.info(f"  跳过(未来帧不足): {skip_short}")
    log.info(f"  跳过(换道中): {skip_lane_change}")
    log.info(f"  总耗时: {total_time:.1f}s")

    if sample_count == 0:
        log.error("没有有效的评估样本！请检查数据和参数设置。")
        return None

    results = {
        'sample_count': sample_count,
        'ADE': float(np.mean(all_ade)),
        'ADE_std': float(np.std(all_ade)),
        'FDE': float(np.mean(all_fde)),
        'FDE_std': float(np.std(all_fde)),
        'RMSE_acc': float(np.mean(all_rmse_acc)),
        'RMSE_steer': float(np.mean(all_rmse_steer)),
        'RMSE_lat': float(np.mean(all_rmse_lat)),
        'RMSE_lon': float(np.mean(all_rmse_lon)),
        'avg_inference_ms': float(np.mean(all_inference_ms)) if all_inference_ms else 0,
        'p50_inference_ms': float(np.percentile(all_inference_ms, 50)) if all_inference_ms else 0,
        'p95_inference_ms': float(np.percentile(all_inference_ms, 95)) if all_inference_ms else 0,
        'total_time_s': total_time,
        'all_ade': all_ade,
        'all_fde': all_fde,
        'all_rmse_acc': all_rmse_acc,
        'all_rmse_steer': all_rmse_steer,
        'skip_nan': skip_nan,
        'skip_short': skip_short,
        'skip_lane_change': skip_lane_change,
    }

    return results


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      结果保存与可视化                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def save_results(results):
    """将评估结果写入文本报告文件"""
    log.info(f"保存结果报告: {RESULT_FILE}")

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("      NGSIM 公开数据集 — Transformer BC 模型 开环测试报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("【测试配置】\n")
        f.write(f"  模型文件:       {MODEL_PATH}\n")
        f.write(f"  NGSIM 数据文件: {NGSIM_CSV_PATH}\n")
        f.write(f"  设备:           {DEVICE}\n")
        f.write(f"  目标车道:       Lane {TARGET_LANES}\n")
        f.write(f"  车道宽度:       {LANE_WIDTH} m\n")
        f.write(f"  预测步长:       {PREDICTION_HORIZON} 步 × {NGSIM_DT}s = {PREDICTION_HORIZON * NGSIM_DT}s\n")
        f.write(f"  最大样本数:     {MAX_EVAL_SAMPLES}\n")
        f.write(f"  平滑窗口:       {SMOOTHING_WINDOW} (Savitzky-Golay)\n\n")

        f.write("【模型参数】\n")
        f.write(f"  d_model={D_MODEL}, FFN_dim={FFN_DIM}\n")
        f.write(f"  Encoder={NUM_ENCODER_LAYERS}层, Decoder={NUM_DECODER_LAYERS}层\n")
        f.write(f"  输入维度=54 (6+8×6), 输出=(5,2)\n\n")

        f.write("【评估结果】\n")
        f.write(f"  有效样本数:     {results['sample_count']}\n")
        f.write(f"  跳过(NaN):      {results['skip_nan']}\n")
        f.write(f"  跳过(帧不足):   {results['skip_short']}\n")
        f.write(f"  跳过(换道中):   {results['skip_lane_change']}\n")
        f.write(f"  总评估耗时:     {results['total_time_s']:.1f} s\n\n")

        f.write("┌────────────────────────────────────────────────────┐\n")
        f.write("│                  核心指标                          │\n")
        f.write("├────────────────────────────────────────────────────┤\n")
        f.write(f"│  ADE (平均位移误差):   {results['ADE']:.4f} ± {results['ADE_std']:.4f} m    │\n")
        f.write(f"│  FDE (终点位移误差):   {results['FDE']:.4f} ± {results['FDE_std']:.4f} m    │\n")
        f.write(f"│  RMSE 加速度:          {results['RMSE_acc']:.4f} m/s²              │\n")
        f.write(f"│  RMSE 转向角:          {results['RMSE_steer']:.6f} rad            │\n")
        f.write(f"│  RMSE 横向位置:        {results['RMSE_lat']:.4f} m                │\n")
        f.write(f"│  RMSE 纵向位置:        {results['RMSE_lon']:.4f} m                │\n")
        f.write("├────────────────────────────────────────────────────┤\n")
        f.write(f"│  平均推理耗时:         {results['avg_inference_ms']:.3f} ms              │\n")
        f.write(f"│  P50 推理耗时:         {results['p50_inference_ms']:.3f} ms              │\n")
        f.write(f"│  P95 推理耗时:         {results['p95_inference_ms']:.3f} ms              │\n")
        f.write("└────────────────────────────────────────────────────┘\n\n")

        f.write("【指标说明】\n")
        f.write("  ADE: 模型预测轨迹 5步(0.5s) 内与真实轨迹的平均欧式距离，越小越好\n")
        f.write("  FDE: 模型预测的第5步位置与真实第5步位置的欧式距离，越小越好\n")
        f.write("  RMSE_acc: 模型预测加速度与真实加速度的均方根误差，越小越好\n")
        f.write("  RMSE_steer: 模型预测转向角与真实转向角的均方根误差，越小越好\n")
        f.write("  RMSE_lat: 横向(跨车道方向)位置误差的均方根，反映车道保持能力\n")
        f.write("  RMSE_lon: 纵向(行驶方向)位置误差的均方根，反映速度控制能力\n")

    log.info(f"报告已保存: {RESULT_FILE}")


def plot_results(results):
    """绘制评估结果的可视化图表"""
    log.info(f"正在绘制图表...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上: ADE 分布直方图
    ax = axes[0, 0]
    ade_arr = np.array(results['all_ade'])
    ax.hist(ade_arr, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(results['ADE'], color='red', linestyle='--', linewidth=2,
               label=f"Mean={results['ADE']:.4f}m")
    ax.set_xlabel('ADE (m)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Average Displacement Error Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右上: FDE 分布直方图
    ax = axes[0, 1]
    fde_arr = np.array(results['all_fde'])
    ax.hist(fde_arr, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(results['FDE'], color='red', linestyle='--', linewidth=2,
               label=f"Mean={results['FDE']:.4f}m")
    ax.set_xlabel('FDE (m)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Final Displacement Error Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 左下: 加速度 RMSE 分布
    ax = axes[1, 0]
    rmse_acc_arr = np.array(results['all_rmse_acc'])
    ax.hist(rmse_acc_arr, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.axvline(results['RMSE_acc'], color='red', linestyle='--', linewidth=2,
               label=f"Mean={results['RMSE_acc']:.4f} m/s²")
    ax.set_xlabel('RMSE Acceleration (m/s²)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Acceleration RMSE Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右下: 转向角 RMSE 分布
    ax = axes[1, 1]
    rmse_steer_arr = np.array(results['all_rmse_steer'])
    ax.hist(rmse_steer_arr, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.axvline(results['RMSE_steer'], color='red', linestyle='--', linewidth=2,
               label=f"Mean={results['RMSE_steer']:.6f} rad")
    ax.set_xlabel('RMSE Steering (rad)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Steering RMSE Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('NGSIM Open-loop Evaluation — Transformer BC Model', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200, bbox_inches='tight')
    plt.close()
    log.info(f"图表已保存: {PLOT_FILE}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          主函数                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    log.info("=" * 70)
    log.info("   NGSIM 公开数据集 Transformer BC 模型 开环测试")
    log.info("=" * 70)
    log.info(f"日志文件: {LOG_FILE}")
    log.info(f"设备: {DEVICE}")
    log.info(f"模型: {MODEL_PATH}")
    log.info(f"NGSIM 数据: {NGSIM_CSV_PATH}")
    log.info(f"目标车道: {TARGET_LANES}")
    log.info(f"最大评估样本: {MAX_EVAL_SAMPLES}")
    log.info("")

    # ---- 1. 加载模型 ----
    log.info("[步骤 1/5] 加载 Transformer 模型...")
    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
        nhead=8, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=FFN_DIM, dropout=0.0,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        log.error(f"找不到模型文件: {MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"  模型加载成功, 参数量: {param_count:,}")

    # ---- 2. 加载 NGSIM 数据 ----
    log.info("\n[步骤 2/5] 加载并预处理 NGSIM 数据...")
    df = load_ngsim_csv(NGSIM_CSV_PATH)

    log.info(f"\n  数据概览:")
    log.info(f"    总行数:       {len(df)}")
    log.info(f"    车辆数:       {df['Vehicle_ID'].nunique()}")
    log.info(f"    帧数范围:     {df['Frame_ID'].min()} ~ {df['Frame_ID'].max()}")
    if 'Global_Time' in df.columns:
        log.info(f"    时间范围:     {df['Global_Time'].min():.1f}s ~ {df['Global_Time'].max():.1f}s")
    log.info(f"    Local_Y范围:  {df['Local_Y'].min():.1f} ~ {df['Local_Y'].max():.1f} m (纵向)")
    log.info(f"    Local_X范围:  {df['Local_X'].min():.1f} ~ {df['Local_X'].max():.1f} m (横向)")
    log.info(f"    速度范围:     {df['v_Vel'].min():.1f} ~ {df['v_Vel'].max():.1f} m/s")

    # ---- 3. 计算航向角/转向角 ----
    log.info("\n[步骤 3/5] 计算航向角(yaw)和转向角(steer)...")
    df = compute_derived_fields(df)

    # ---- 4. 计算车道边界 ----
    log.info("\n[步骤 4/5] 计算车道边界...")
    map_info, lane1, lane2, lane3, lane4 = compute_lane_boundaries(df, TARGET_LANES)

    # ---- 5. 开环评估 ----
    log.info(f"\n[步骤 5/5] 执行开环评估...")
    results = run_openloop_evaluation(df, model, map_info, lane1, lane2, lane3, lane4)

    if results is None:
        log.error("评估失败，无有效结果")
        sys.exit(1)

    # ---- 打印最终结果 ----
    log.info("\n" + "=" * 70)
    log.info("                     最终评估结果")
    log.info("=" * 70)
    log.info(f"  有效样本数:         {results['sample_count']}")
    log.info(f"  ADE (平均位移误差): {results['ADE']:.4f} ± {results['ADE_std']:.4f} m")
    log.info(f"  FDE (终点位移误差): {results['FDE']:.4f} ± {results['FDE_std']:.4f} m")
    log.info(f"  RMSE 加速度:        {results['RMSE_acc']:.4f} m/s²")
    log.info(f"  RMSE 转向角:        {results['RMSE_steer']:.6f} rad")
    log.info(f"  RMSE 横向位置:      {results['RMSE_lat']:.4f} m")
    log.info(f"  RMSE 纵向位置:      {results['RMSE_lon']:.4f} m")
    log.info(f"  平均推理耗时:       {results['avg_inference_ms']:.3f} ms")
    log.info(f"  P95 推理耗时:       {results['p95_inference_ms']:.3f} ms")

    # ---- 保存结果 ----
    save_results(results)
    plot_results(results)

    log.info(f"\n所有输出文件:")
    log.info(f"  日志:   {LOG_FILE}")
    log.info(f"  报告:   {RESULT_FILE}")
    log.info(f"  图表:   {PLOT_FILE}")
    log.info("\n测试完成!")


if __name__ == "__main__":
    main()
