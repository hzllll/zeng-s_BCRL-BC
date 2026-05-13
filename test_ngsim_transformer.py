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
import argparse
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
NUM_EVAL_PERIODS = 4         # 分时段报告数量；设为 4 时输出 时段1~时段4 + 总计
PREDICTION_HORIZON = 5       # 预测步长 = 5 步 × 0.1s = 0.5 秒（与模型输出一致）
MIN_TRAJECTORY_LENGTH = 50   # 车辆最少存在帧数，太短的轨迹不适合评估
SMOOTHING_WINDOW = 11        # Savitzky-Golay 滤波窗口长度 (必须为奇数)，用于平滑位置/速度噪声
SMOOTHING_POLY = 3           # Savitzky-Golay 滤波多项式阶数
AUTO_SELECT_PRIMARY_SCENE = True  # 从混合 CSV 中自动选取目标车道样本最多的单一路段/方向
PRIMARY_SCENE_ID = None           # 如需固定场景，可填类似 "us-101|2|1"；None 表示自动选择
CSV_CHUNK_SIZE = 100_000          # 分块读取行数，避免一次性读入千万级 CSV 被系统 killed
MAX_DATA_ROWS = 800_000           # 评估最多 5000 样本，不需要把主场景几百万行全部读入内存

# ---------- 模型路径 ----------
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    # "Tf_trajectory_model_0328_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_100es_CoAnWarmRest_zDATASET.pth"
    # "Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth"
    "Exp-5_Tf_trajectory_model_0418_10_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_300es_CoAnWarmRest_zDATASET.pth"
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
EXP_NAME = "Exp-5"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")
SAVE_DIR = os.path.join(BASE_DIR, "ngsim_results", EXP_NAME + "_" + TIMESTAMP)
# SAVE_DIR = os.path.join(BASE_DIR, "ngsim_results")
os.makedirs(SAVE_DIR, exist_ok=True)

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

def _make_scene_id(df):
    """构造场景键，避免把不同路段/方向/数据源的相同 Frame_ID 混在一起。"""
    scene_cols = [col for col in ['Location', 'Section_ID', 'Direction'] if col in df.columns]
    if not scene_cols:
        df['Scene_ID'] = 'ALL'
        return df

    parts = []
    for col in scene_cols:
        parts.append(df[col].astype('string').fillna('NA').str.strip())
    df['Scene_ID'] = parts[0]
    for part in parts[1:]:
        df['Scene_ID'] = df['Scene_ID'] + '|' + part
    return df


def _select_primary_scene(df):
    """只保留一个真实场景，防止同一 Frame_ID 下混入其他路段车辆。"""
    if 'Scene_ID' not in df.columns:
        return df, 'ALL'

    target_df = df[df['Lane_ID'].isin(TARGET_LANES)]
    if len(target_df) == 0:
        log.warning("目标车道没有数据，暂不进行单场景过滤")
        return df, 'ALL'

    if PRIMARY_SCENE_ID is not None:
        selected_scene = PRIMARY_SCENE_ID
        if selected_scene not in set(df['Scene_ID'].astype(str)):
            log.warning(f"指定 PRIMARY_SCENE_ID={selected_scene} 不在数据中，将改用自动选择")
        else:
            filtered = df[df['Scene_ID'] == selected_scene].copy()
            log.info(f"使用指定场景: {selected_scene}, 行数={len(filtered)}")
            return filtered, selected_scene

    scene_counts = target_df.groupby('Scene_ID').size().sort_values(ascending=False)
    selected_scene = scene_counts.index[0]
    filtered = df[df['Scene_ID'] == selected_scene].copy()

    log.info("检测到混合 NGSIM 数据，按目标车道样本数选择单一场景:")
    for scene_id, count in scene_counts.head(5).items():
        log.info(f"  Scene_ID={scene_id}: 目标车道样本 {count}")
    log.info(f"  最终使用 Scene_ID={selected_scene}, 过滤后行数={len(filtered)}")
    return filtered, selected_scene


def _actual_col(lower_to_actual, col_name):
    return lower_to_actual.get(col_name.lower())


def _to_numeric(series):
    """兼容 CSV 中偶发的千位逗号，例如 '1,156'。"""
    return pd.to_numeric(
        series.astype('string').str.replace(',', '', regex=False).str.strip(),
        errors='coerce',
    )


def _detect_primary_scene_from_csv(csv_path, lower_to_actual):
    """第一遍分块扫描，只统计目标车道最多的场景，不加载完整 CSV。"""
    if PRIMARY_SCENE_ID is not None:
        return PRIMARY_SCENE_ID

    lane_col = _actual_col(lower_to_actual, 'Lane_ID')
    if lane_col is None:
        log.warning("CSV 中没有 Lane_ID，无法自动选择场景")
        return 'ALL'

    scan_cols = [lane_col]
    for col in ['Location', 'Section_ID', 'Direction']:
        actual_col = _actual_col(lower_to_actual, col)
        if actual_col is not None:
            scan_cols.append(actual_col)

    scene_counts = {}
    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=scan_cols, chunksize=CSV_CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)
        chunk['Lane_ID'] = pd.to_numeric(chunk[lane_col.strip()], errors='coerce')
        chunk.dropna(subset=['Lane_ID'], inplace=True)
        if len(chunk) == 0:
            continue

        chunk = _make_scene_id(chunk)
        target_chunk = chunk[chunk['Lane_ID'].isin(TARGET_LANES)]
        counts = target_chunk['Scene_ID'].value_counts()
        for scene_id, count in counts.items():
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + int(count)

    if not scene_counts:
        log.warning("分块扫描没有发现目标车道数据，将使用 ALL")
        return 'ALL'

    selected_scene = max(scene_counts.items(), key=lambda item: item[1])[0]
    log.info(f"分块扫描完成: 共扫描 {total_rows} 行")
    log.info("目标车道样本最多的前5个场景:")
    for scene_id, count in sorted(scene_counts.items(), key=lambda item: item[1], reverse=True)[:5]:
        log.info(f"  Scene_ID={scene_id}: 目标车道样本 {count}")
    log.info(f"自动选择主场景: {selected_scene}")
    return selected_scene


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

    required_cols = [
        'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
        'Local_X', 'Local_Y', 'v_Length', 'v_Width',
        'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID',
        'Section_ID', 'Direction', 'Preceding', 'Following',
        'Space_Headway', 'Time_Headway', 'Location'
    ]

    # 先只读表头，兼容 v_Length/v_length 等列名差异；避免列名不匹配时退回全量读取。
    header = pd.read_csv(csv_path, nrows=0)
    actual_columns = [str(col).strip() for col in header.columns]
    lower_to_actual = {col.lower(): col for col in actual_columns}
    log.info(f"CSV 文件中的所有列名: {actual_columns}")

    usecols = []
    missing_cols = []
    for col in required_cols:
        actual_col = lower_to_actual.get(col.lower())
        if actual_col is None:
            missing_cols.append(col)
        else:
            usecols.append(actual_col)

    if missing_cols:
        log.warning(f"以下可选列不存在，将跳过: {missing_cols}")

    # 第二遍读取先按字符串分块读入，后续统一清洗千位逗号并转数值。
    dtype_map = {col: 'string' for col in usecols}

    selected_scene = 'ALL'
    if AUTO_SELECT_PRIMARY_SCENE:
        selected_scene = _detect_primary_scene_from_csv(csv_path, lower_to_actual)

    log.info(f"按必要列分块读取 CSV: {usecols}")
    chunks = []
    total_rows = 0
    kept_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype_map,
        low_memory=False,
        chunksize=CSV_CHUNK_SIZE,
    ):
        total_rows += len(chunk)
        chunk.columns = chunk.columns.str.strip()
        if 'v_length' in chunk.columns and 'v_Length' not in chunk.columns:
            chunk.rename(columns={'v_length': 'v_Length'}, inplace=True)
        chunk = _make_scene_id(chunk)

        if AUTO_SELECT_PRIMARY_SCENE and selected_scene != 'ALL':
            chunk = chunk[chunk['Scene_ID'] == selected_scene]
        if 'Lane_ID' in chunk.columns:
            chunk['Lane_ID'] = _to_numeric(chunk['Lane_ID'])
            chunk = chunk[chunk['Lane_ID'].isin(TARGET_LANES)]
        if len(chunk) == 0:
            continue

        numeric_cols_in_chunk = [
            'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
            'Local_X', 'Local_Y', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc',
            'Space_Headway', 'Time_Headway', 'Lane_ID', 'Section_ID', 'Direction',
            'Preceding', 'Following', 'v_Class'
        ]
        for col in numeric_cols_in_chunk:
            if col in chunk.columns and not pd.api.types.is_numeric_dtype(chunk[col]):
                chunk[col] = _to_numeric(chunk[col])

        chunk.dropna(subset=['Local_X', 'Local_Y', 'v_Vel', 'Lane_ID'], inplace=True)
        chunk.drop(columns=['Location', 'Section_ID', 'Direction'], errors='ignore', inplace=True)
        if len(chunk) == 0:
            continue

        if 0 < MAX_DATA_ROWS < kept_rows + len(chunk):
            chunk = chunk.iloc[:MAX_DATA_ROWS - kept_rows].copy()

        kept_rows += len(chunk)
        chunks.append(chunk)

        if 0 < MAX_DATA_ROWS <= kept_rows:
            log.info(f"已达到最大加载行数 {MAX_DATA_ROWS}，停止继续读取 CSV")
            break

    if not chunks:
        log.error(f"分块读取后没有保留任何数据，请检查场景过滤: {selected_scene}")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    df.attrs['selected_scene'] = selected_scene
    log.info(f"分块读取完成: 扫描 {total_rows} 行, 保留 {kept_rows} 行")

    log.info(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    log.info(f"CSV 列名: {list(df.columns)}")


    # 修复列名大小写不一致的问题
    if 'v_length' in df.columns and 'v_Length' not in df.columns:
        df.rename(columns={'v_length': 'v_Length'}, inplace=True)

    # 2. 解决根本原因：强制将关键列转换为数值类型 (float)
    # errors='coerce' 的作用是：遇到像空格、字母等无法转换的脏字符时，直接将其变成 NaN（缺失值），而不是报错崩溃
    # 强制将所有需要计算的列转换为数值类型 (float)，遇到脏字符(如空格)强制转为 NaN
    numeric_cols = [
        'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
        'Local_X', 'Local_Y', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc',
        'Space_Headway', 'Time_Headway', 'Lane_ID', 'Section_ID', 'Direction',
        'Preceding', 'Following', 'v_Class'
    ]
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = _to_numeric(df[col])
            
    # 3. 剔除无效数据：把那些因为脏数据被转换成 NaN 的行删掉，防止后续计算（如乘法、求导）报错
    before_drop = len(df)
    df.dropna(subset=['Local_X', 'Local_Y', 'v_Vel', 'Lane_ID'], inplace=True)
    after_drop = len(df)
    if before_drop != after_drop:
        log.warning(f"已剔除 {before_drop - after_drop} 行包含脏数据/缺失值的无效记录")
    # ==================================================================

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

    sort_cols = ['Scene_ID', 'Vehicle_ID', 'Frame_ID']
    if 'Global_Time' in df.columns:
        sort_cols.append('Global_Time')
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # 区分相同 Vehicle_ID 在不同场景或不同连续片段中的轨迹。
    group_cols = ['Scene_ID', 'Vehicle_ID']
    frame_diff = df.groupby(group_cols)['Frame_ID'].diff()
    new_segment = frame_diff.isna() | (frame_diff != 1)
    df['Trajectory_ID'] = new_segment.cumsum().astype('int64')
    
    log.info(f"单位转换完成: 距离(m), 速度(m/s), 加速度(m/s²), 时间(s)")
    log.info(f"当前评估场景: {selected_scene}")
    log.info(f"提取出独立的连续轨迹数量: {df['Trajectory_ID'].nunique()}")
    log.info(f"Lane_ID 分布: {dict(df['Lane_ID'].value_counts().sort_index())}")
    
    if 'v_Class' in df.columns:
        log.info(f"车辆类型分布: {dict(df['v_Class'].value_counts().sort_index())}")

    return df

def compute_derived_fields(df):
    """
    计算 NGSIM 数据中不直接提供但模型需要的字段：航向角 (yaw) 和转向角 (steer)。
    使用纯 NumPy 向量化操作，极低内存占用，几秒钟内处理 150 万条轨迹。
    """
    log.info("正在计算航向角(yaw)和转向角(steer) (已启用纯 NumPy 加速)...")

    # 提取需要的列为 numpy 数组，加速计算
    x_raw = df['Local_Y'].values
    y_raw = df['Local_X'].values
    v_raw = df['v_Vel'].values
    tids = df['Trajectory_ID'].values

    # 初始化结果数组
    yaw_all = np.full(len(df), np.nan)
    steer_all = np.full(len(df), np.nan)

    # 找到每条轨迹的起始和结束索引
    # tids 是递增的，所以变化的地方就是新轨迹的起点
    change_indices = np.where(tids[:-1] != tids[1:])[0] + 1
    start_indices = np.concatenate(([0], change_indices))
    end_indices = np.concatenate((change_indices, [len(df)]))

    processed = 0
    valid_count = 0

    # 遍历每条轨迹的索引范围（只传递索引，不复制数据，内存极小）
    for start_idx, end_idx in zip(start_indices, end_indices):
        length = end_idx - start_idx
        if length < 5:
            continue

        # 提取当前轨迹的切片（视图，不占内存）
        x_traj = x_raw[start_idx:end_idx]
        y_traj = y_raw[start_idx:end_idx]
        v_traj = v_raw[start_idx:end_idx]

        # Savitzky-Golay 平滑
        win = min(SMOOTHING_WINDOW, length)
        if win % 2 == 0:
            win -= 1
        if win >= SMOOTHING_POLY + 2:
            x_smooth = savgol_filter(x_traj, win, SMOOTHING_POLY)
            y_smooth = savgol_filter(y_traj, win, SMOOTHING_POLY)
        else:
            x_smooth = x_traj
            y_smooth = y_traj

        # 计算 yaw
        dx = np.gradient(x_smooth, NGSIM_DT)
        dy = np.gradient(y_smooth, NGSIM_DT)
        yaw = np.arctan2(dy, dx)

        # 计算 steer
        dyaw = np.gradient(yaw, NGSIM_DT)
        dyaw = (dyaw + np.pi / NGSIM_DT) % (2 * np.pi / NGSIM_DT) - np.pi / NGSIM_DT
        dyaw = np.clip(dyaw, -2.0 / NGSIM_DT, 2.0 / NGSIM_DT)

        v_safe = np.maximum(np.abs(v_traj), 0.5)
        steer = np.arctan(WHEELBASE * dyaw / v_safe)
        steer = np.clip(steer, -0.5, 0.5)

        # 赋值回结果数组
        yaw_all[start_idx:end_idx] = yaw
        steer_all[start_idx:end_idx] = steer
        
        processed += 1
        valid_count += length

    # 将结果写回 DataFrame
    df['yaw'] = yaw_all
    df['steer'] = steer_all

    log.info(f"航向角/转向角计算完成: 处理 {processed} 条有效轨迹, {valid_count}/{len(df)} 行有效")

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


def _score_efficiency(mean_pred_v, mean_ref_v):
    """效率分：以同一 NGSIM 片段真实交通流速度为参考，满分 30。"""
    efficiency_score = 30.0
    if mean_pred_v < mean_ref_v * 0.8:
        efficiency_score -= (mean_ref_v - mean_pred_v) * 2.0
    elif mean_pred_v > mean_ref_v * 1.2:
        efficiency_score -= (mean_pred_v - mean_ref_v) * 2.0
    return float(np.clip(efficiency_score, 0.0, 30.0))


def _score_comfort(acc_seq, steer_seq):
    """
    舒适分：从“动作幅值硬扣分”改为以 jerk / steer-rate 为主。
    NGSIM 是 0.5s 短时开环评估，动作变化率比单步最大值更能反映舒适性。
    """
    acc_seq = np.asarray(acc_seq, dtype=float)
    steer_seq = np.asarray(steer_seq, dtype=float)

    comfort_score = 20.0
    max_abs_acc = float(np.max(np.abs(acc_seq))) if len(acc_seq) else 0.0
    max_abs_steer = float(np.max(np.abs(steer_seq))) if len(steer_seq) else 0.0

    if len(acc_seq) > 1:
        mean_jerk = float(np.mean(np.abs(np.diff(acc_seq))) / NGSIM_DT)
        mean_steer_rate = float(np.mean(np.abs(np.diff(steer_seq))) / NGSIM_DT)
    else:
        mean_jerk = 0.0
        mean_steer_rate = 0.0

    if max_abs_acc > 4.0:
        comfort_score -= min(3.0, (max_abs_acc - 4.0) * 0.8)
    if max_abs_steer > 0.12:
        comfort_score -= min(2.0, (max_abs_steer - 0.12) * 20.0)
    if mean_jerk > 4.0:
        comfort_score -= min(5.0, (mean_jerk - 4.0) * 0.6)
    if mean_steer_rate > 0.25:
        comfort_score -= min(4.0, (mean_steer_rate - 0.25) * 8.0)

    return float(np.clip(comfort_score, 0.0, 20.0)), mean_jerk, mean_steer_rate


def _safety_risk_for_path(path_x, path_y, speed_seq, ego_row, close_others, lane1, lane4):
    """
    计算短时轨迹风险。这里使用道路坐标下的近似矩形占用，而不是单纯欧氏距离。
    返回的风险会同时用于 Expert 和 Agent，Agent 只对相对 Expert 的新增风险重扣。
    """
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    speed_seq = np.asarray(speed_seq, dtype=float)

    out_of_lane = bool(np.any(path_y > lane1) or np.any(path_y < lane4))
    hard_collision = False
    min_gap = float("inf")
    min_ttc = float("inf")

    ego_len = float(ego_row['v_Length']) if not np.isnan(ego_row['v_Length']) else 4.8
    ego_width = float(ego_row['v_Width']) if not np.isnan(ego_row['v_Width']) else 1.8

    for other in close_others:
        other_len = float(other['v_Length']) if not np.isnan(other['v_Length']) else 4.8
        other_width = float(other['v_Width']) if not np.isnan(other['v_Width']) else 1.8
        half_len = 0.5 * (ego_len + other_len)
        half_width = 0.5 * (ego_width + other_width)

        other_x0 = float(other['Local_Y'])
        other_y0 = float(other['Local_X'])
        other_v = float(other['v_Vel'])
        other_yaw = float(other['yaw']) if not np.isnan(other['yaw']) else 0.0

        steps = np.arange(1, len(path_x) + 1)
        other_x = other_x0 + other_v * np.cos(other_yaw) * NGSIM_DT * steps
        other_y = other_y0 + other_v * np.sin(other_yaw) * NGSIM_DT * steps

        lon_sep = np.abs(path_x - other_x)
        lat_sep = np.abs(path_y - other_y)
        signed_lon_gap = lon_sep - half_len
        signed_lat_gap = lat_sep - half_width
        rect_gap = np.maximum(signed_lon_gap, 0.0) + np.maximum(signed_lat_gap, 0.0)
        min_gap = min(min_gap, float(np.min(rect_gap)))

        # 允许少量 NGSIM 标注噪声/矩形近似误差，避免把密集跟车都判为碰撞。
        if np.any((lon_sep < half_len * 0.70) & (lat_sep < half_width * 0.75)):
            hard_collision = True

        same_lane = lat_sep < max(2.2, half_width + 0.4)
        ahead = other_x > path_x
        rel_speed = speed_seq - other_v
        valid_ttc = same_lane & ahead & (rel_speed > 0.1)
        if np.any(valid_ttc):
            ttc = (other_x[valid_ttc] - path_x[valid_ttc] - half_len) / rel_speed[valid_ttc]
            ttc = ttc[ttc > 0]
            if len(ttc):
                min_ttc = min(min_ttc, float(np.min(ttc)))

    return {
        'out_of_lane': out_of_lane,
        'hard_collision': hard_collision,
        'min_gap': min_gap,
        'min_ttc': min_ttc,
    }


def _score_safety(agent_risk, expert_risk):
    """
    安全分：满分 50。
    - 越界最多扣 5 分；
    - 碰撞只对 Agent 相比 Expert 的新增碰撞重扣；
    - 近距和 TTC 作为软风险扣分。
    """
    safety_score = 50.0

    if agent_risk['out_of_lane']:
        safety_score -= 5.0

    new_collision = agent_risk['hard_collision'] and not expert_risk['hard_collision']
    inherited_collision = agent_risk['hard_collision'] and expert_risk['hard_collision']
    if new_collision:
        safety_score -= 15.0
    elif inherited_collision:
        safety_score -= 3.0

    min_gap = agent_risk['min_gap']
    expert_gap = expert_risk['min_gap']
    if np.isfinite(min_gap):
        # 只在 Agent 比 Expert 明显更近时扣重分；否则视为真实交通流密集带来的背景风险。
        added_gap_risk = np.isfinite(expert_gap) and min_gap + 0.3 < expert_gap
        if min_gap < 0.2 and added_gap_risk:
            safety_score -= 6.0
        elif min_gap < 0.6 and added_gap_risk:
            safety_score -= 3.0
        elif min_gap < 0.3:
            safety_score -= 1.0

    min_ttc = agent_risk['min_ttc']
    expert_ttc = expert_risk['min_ttc']
    if np.isfinite(min_ttc):
        added_ttc_risk = (not np.isfinite(expert_ttc)) or (min_ttc + 0.5 < expert_ttc)
        if min_ttc < 1.0 and added_ttc_risk:
            safety_score -= 5.0
        elif min_ttc < 2.0 and added_ttc_risk:
            safety_score -= 3.0
        elif min_ttc < 3.0:
            safety_score -= 1.0

    return float(np.clip(safety_score, 0.0, 50.0)), new_collision


def _period_name(frame_id, frame_min, frame_max):
    """按 Frame_ID 均匀切分生成论文式时段标签。"""
    if frame_max <= frame_min:
        return "时段1"
    ratio = (frame_id - frame_min) / (frame_max - frame_min)
    period_idx = min(int(ratio * NUM_EVAL_PERIODS), NUM_EVAL_PERIODS - 1)
    return f"时段{period_idx + 1}"


def _select_evenly(records, limit):
    """从已按时间排序的候选样本中均匀取样，避免只取到某个时段开头。"""
    if limit < 0 or len(records) <= limit:
        return records
    if limit <= 0:
        return []
    indices = np.linspace(0, len(records) - 1, limit, dtype=int)
    return [records[int(i)] for i in indices]


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
    frame_min = float(df_target['Frame_ID'].min())
    frame_max = float(df_target['Frame_ID'].max())

    # 筛选轨迹足够长的车辆
    # veh_counts = df_target.groupby('Vehicle_ID').size()
    veh_counts = df_target.groupby('Trajectory_ID').size()
    valid_trajectories = veh_counts[veh_counts >= MIN_TRAJECTORY_LENGTH].index.tolist()
    log.info(f"轨迹长度 >= {MIN_TRAJECTORY_LENGTH} 帧的轨迹: {len(valid_trajectories)} 条")

    # 按场景+帧分组，避免不同路段/方向中相同 Frame_ID 的车辆互相污染。
    frame_groups = df_target.groupby(['Scene_ID', 'Frame_ID'])

    all_ade = []
    all_fde = []
    all_rmse_acc = []
    all_rmse_steer = []
    all_rmse_lat = []
    all_rmse_lon = []
    all_inference_ms = []

    all_safety_scores = []
    all_efficiency_scores = []
    all_comfort_scores = []
    all_expert_safety_scores = []
    all_expert_efficiency_scores = []
    all_expert_comfort_scores = []
    collision_count = 0
    expert_collision_count = 0
    out_of_lane_count = 0
    expert_out_of_lane_count = 0
    all_gt_step_dist = []
    all_const_ade = []
    all_same_frame_counts = []
    all_collision_candidates = []
    period_records = []

    sample_count = 0
    skip_nan = 0
    skip_short = 0
    skip_lane_change = 0

    eval_start_time = time.time()

    periods = [f"时段{i + 1}" for i in range(NUM_EVAL_PERIODS)]
    candidate_by_period = {period: [] for period in periods}

    log.info("正在按时段预扫描有效候选样本，用于分层均匀采样...")
    for tid in valid_trajectories:
        veh_data = df_target[df_target['Trajectory_ID'] == tid].sort_values('Frame_ID')
        if len(veh_data) < MIN_TRAJECTORY_LENGTH:
            continue

        for t_idx in range(0, len(veh_data) - PREDICTION_HORIZON, PREDICTION_HORIZON):
            current_row = veh_data.iloc[t_idx]
            if np.isnan(current_row['yaw']):
                continue

            current_frame = current_row['Frame_ID']
            future_frames_needed = [current_frame + k for k in range(1, PREDICTION_HORIZON + 1)]
            future_data = veh_data[veh_data['Frame_ID'].isin(future_frames_needed)]
            if len(future_data) < PREDICTION_HORIZON:
                continue
            if not future_data['Lane_ID'].isin(TARGET_LANES).all():
                continue
            if np.any(np.isnan(future_data['steer'].values[:PREDICTION_HORIZON])):
                continue

            period = _period_name(current_frame, frame_min, frame_max)
            candidate_by_period[period].append({
                'tid': tid,
                't_idx': t_idx,
                'frame_id': float(current_frame),
            })

    selected_candidates = []
    selected_keys = set()

    if MAX_EVAL_SAMPLES < 0:
        for period in periods:
            group = sorted(candidate_by_period[period], key=lambda item: item['frame_id'])
            selected_candidates.extend(group)
    else:
        base_quota = MAX_EVAL_SAMPLES // NUM_EVAL_PERIODS
        remainder = MAX_EVAL_SAMPLES % NUM_EVAL_PERIODS

        for idx, period in enumerate(periods):
            quota = base_quota + (1 if idx < remainder else 0)
            group = sorted(candidate_by_period[period], key=lambda item: item['frame_id'])
            selected_candidates.extend(_select_evenly(group, quota))

        # 如果某些时段候选不足，用其余时段的未选样本补足总样本数。
        if len(selected_candidates) < MAX_EVAL_SAMPLES:
            selected_keys = {(item['tid'], item['t_idx']) for item in selected_candidates}
            all_candidates = []
            for period in periods:
                all_candidates.extend(candidate_by_period[period])
            all_candidates = sorted(all_candidates, key=lambda item: item['frame_id'])
            remaining = [
                item for item in all_candidates
                if (item['tid'], item['t_idx']) not in selected_keys
            ]
            need = MAX_EVAL_SAMPLES - len(selected_candidates)
            selected_candidates.extend(_select_evenly(remaining, need))

    selected_keys = {(item['tid'], item['t_idx']) for item in selected_candidates}
    selected_counts = {period: 0 for period in periods}
    for item in selected_candidates:
        period = _period_name(item['frame_id'], frame_min, frame_max)
        selected_counts[period] += 1

    for period in periods:
        log.info(
            f"  {period}: 候选 {len(candidate_by_period[period])} 个, "
            f"计划评估 {selected_counts[period]} 个"
        )
    log.info(f"  分层采样总候选: {sum(len(v) for v in candidate_by_period.values())}, 计划评估: {len(selected_candidates)}")

    for tid in valid_trajectories:
        veh_data = df_target[df_target['Trajectory_ID'] == tid].sort_values('Frame_ID')

        if len(veh_data) < MIN_TRAJECTORY_LENGTH:
            continue

        # 目标点: 使用该车轨迹终点（模型坐标系）
        # goal_x = veh_data.iloc[-1]['Local_Y']
        # goal_y = veh_data.iloc[-1]['Local_X']

        # 每隔 PREDICTION_HORIZON 帧采样一次（避免重叠评估）
        for t_idx in range(0, len(veh_data) - PREDICTION_HORIZON, PREDICTION_HORIZON):
            if 0 < MAX_EVAL_SAMPLES <= sample_count:
                break

            if (tid, t_idx) not in selected_keys:
                continue

            current_row = veh_data.iloc[t_idx]
            current_frame = current_row['Frame_ID']
            current_scene = current_row['Scene_ID']

            # === 新增代码：动态构造目标点 ===
            # 1. 纵向目标点：始终保持在主车前方 500 米 (与 highway-env 的 GOAL_LOOKAHEAD 一致)
            goal_x = current_row['Local_Y'] + 500.0
            
            # 2. 横向目标点：保持在当前车道的中心线
            # 寻找距离当前车辆最近的车道中心作为目标 y
            centers = [map_info[i]['center'] for i in range(3)]
            goal_y = min(centers, key=lambda c: abs(c - current_row['Local_X']))
            # ==============================

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
            frame_key = (current_scene, current_frame)
            if frame_key in frame_groups.groups:
                same_frame = frame_groups.get_group(frame_key)
                others = same_frame[same_frame['Trajectory_ID'] != tid]
            else:
                same_frame = pd.DataFrame()
                others = pd.DataFrame()
            all_same_frame_counts.append(len(same_frame))

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
            gt_v = future_data['v_Vel'].values[:PREDICTION_HORIZON]

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
            pred_vs = []
            for step in range(PREDICTION_HORIZON):
                cur_v = max(0.0, cur_v + pred_acc[step] * NGSIM_DT)
                cur_yaw = cur_yaw + pred_steer[step] * NGSIM_DT
                ego_x = ego_x + cur_v * np.cos(cur_yaw) * NGSIM_DT
                ego_y_pos = ego_y_pos + cur_v * np.sin(cur_yaw) * NGSIM_DT
                pred_xs.append(ego_x)
                pred_ys.append(ego_y_pos)
                pred_vs.append(cur_v)

            pred_xs = np.array(pred_xs)
            pred_ys = np.array(pred_ys)
            pred_vs = np.array(pred_vs)

            # ---------- 评估诊断：常速基线和真实单步位移 ----------
            step_dist = np.sqrt(np.diff(np.r_[current_row['Local_Y'], gt_x]) ** 2 +
                                np.diff(np.r_[current_row['Local_X'], gt_y]) ** 2)
            all_gt_step_dist.extend(step_dist.tolist())

            base_xs = current_row['Local_Y'] + current_row['v_Vel'] * np.cos(current_row['yaw']) * NGSIM_DT * np.arange(1, PREDICTION_HORIZON + 1)
            base_ys = current_row['Local_X'] + current_row['v_Vel'] * np.sin(current_row['yaw']) * NGSIM_DT * np.arange(1, PREDICTION_HORIZON + 1)
            const_displacements = np.sqrt((base_xs - gt_x) ** 2 + (base_ys - gt_y) ** 2)
            all_const_ade.append(float(np.mean(const_displacements)))

            # ---------- 构造碰撞候选车辆 ----------
            close_others = []
            for _, other in others.iterrows():
                if abs(other['Local_X'] - current_row['Local_X']) > 6.0:
                    continue
                if abs(other['Local_Y'] - current_row['Local_Y']) > 80.0:
                    continue
                close_others.append(other)
            all_collision_candidates.append(len(close_others))

            # =================================================================
            # 论文式 Expert / Agent 同场景评分
            # =================================================================
            expert_risk = _safety_risk_for_path(gt_x, gt_y, gt_v, current_row, close_others, lane1, lane4)
            agent_risk = _safety_risk_for_path(pred_xs, pred_ys, pred_vs, current_row, close_others, lane1, lane4)

            safety_score, is_new_collision = _score_safety(agent_risk, expert_risk)
            expert_safety_score, _ = _score_safety(expert_risk, expert_risk)

            if is_new_collision:
                collision_count += 1
            if expert_risk['hard_collision']:
                expert_collision_count += 1
            if agent_risk['out_of_lane']:
                out_of_lane_count += 1
            if expert_risk['out_of_lane']:
                expert_out_of_lane_count += 1

            mean_gt_v = np.mean(gt_v)
            mean_pred_v = np.mean(pred_vs)
            efficiency_score = _score_efficiency(mean_pred_v, mean_gt_v)
            expert_efficiency_score = _score_efficiency(mean_gt_v, mean_gt_v)

            comfort_score, agent_jerk, agent_steer_rate = _score_comfort(pred_acc, pred_steer)
            expert_comfort_score, expert_jerk, expert_steer_rate = _score_comfort(gt_acc, gt_steer)

            all_safety_scores.append(safety_score)
            all_efficiency_scores.append(efficiency_score)
            all_comfort_scores.append(comfort_score)
            all_expert_safety_scores.append(expert_safety_score)
            all_expert_efficiency_scores.append(expert_efficiency_score)
            all_expert_comfort_scores.append(expert_comfort_score)

            period = _period_name(current_frame, frame_min, frame_max)
            agent_displacements = np.sqrt((pred_xs - gt_x) ** 2 + (pred_ys - gt_y) ** 2)
            goal_success = (
                np.mean(agent_displacements) < 1.5 and
                agent_displacements[-1] < 2.5 and
                not agent_risk['out_of_lane'] and
                not is_new_collision
            )
            period_records.append({
                'period': period,
                'expert_safety': expert_safety_score,
                'expert_efficiency': expert_efficiency_score,
                'expert_comfort': expert_comfort_score,
                'expert_total': expert_safety_score + expert_efficiency_score + expert_comfort_score,
                'expert_goal': not expert_risk['out_of_lane'],
                'agent_safety': safety_score,
                'agent_efficiency': efficiency_score,
                'agent_comfort': comfort_score,
                'agent_total': safety_score + efficiency_score + comfort_score,
                'agent_goal': goal_success,
                'agent_jerk': agent_jerk,
                'agent_steer_rate': agent_steer_rate,
                'expert_jerk': expert_jerk,
                'expert_steer_rate': expert_steer_rate,
            })
            # =================================================================

            # ---------- 计算指标 ----------
            displacements = agent_displacements
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

    if period_records:
        period_df = pd.DataFrame(period_records)
        period_summary = []
        for period in [f"时段{i + 1}" for i in range(NUM_EVAL_PERIODS)]:
            group = period_df[period_df['period'] == period]
            if len(group) == 0:
                continue
            period_summary.append({
                'period': period,
                'count': int(len(group)),
                'expert_safety': float(group['expert_safety'].mean()),
                'expert_efficiency': float(group['expert_efficiency'].mean()),
                'expert_comfort': float(group['expert_comfort'].mean()),
                'expert_total': float(group['expert_total'].mean()),
                'expert_goal_rate': float(group['expert_goal'].mean()),
                'agent_safety': float(group['agent_safety'].mean()),
                'agent_efficiency': float(group['agent_efficiency'].mean()),
                'agent_comfort': float(group['agent_comfort'].mean()),
                'agent_total': float(group['agent_total'].mean()),
                'agent_goal_rate': float(group['agent_goal'].mean()),
            })
        period_summary.append({
            'period': '总计',
            'count': int(len(period_df)),
            'expert_safety': float(period_df['expert_safety'].mean()),
            'expert_efficiency': float(period_df['expert_efficiency'].mean()),
            'expert_comfort': float(period_df['expert_comfort'].mean()),
            'expert_total': float(period_df['expert_total'].mean()),
            'expert_goal_rate': float(period_df['expert_goal'].mean()),
            'agent_safety': float(period_df['agent_safety'].mean()),
            'agent_efficiency': float(period_df['agent_efficiency'].mean()),
            'agent_comfort': float(period_df['agent_comfort'].mean()),
            'agent_total': float(period_df['agent_total'].mean()),
            'agent_goal_rate': float(period_df['agent_goal'].mean()),
        })
    else:
        period_summary = []

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
        'avg_expert_safety': float(np.mean(all_expert_safety_scores)),
        'avg_expert_efficiency': float(np.mean(all_expert_efficiency_scores)),
        'avg_expert_comfort': float(np.mean(all_expert_comfort_scores)),
        'avg_expert_total_score': float(np.mean(all_expert_safety_scores) + np.mean(all_expert_efficiency_scores) + np.mean(all_expert_comfort_scores)),
        'avg_safety': float(np.mean(all_safety_scores)),
        'avg_efficiency': float(np.mean(all_efficiency_scores)),
        'avg_comfort': float(np.mean(all_comfort_scores)),
        'avg_total_score': float(np.mean(all_safety_scores) + np.mean(all_efficiency_scores) + np.mean(all_comfort_scores)),
        'collision_rate': float(collision_count / sample_count * 100) if sample_count > 0 else 0.0,
        'expert_collision_rate': float(expert_collision_count / sample_count * 100) if sample_count > 0 else 0.0,
        'out_of_lane_rate': float(out_of_lane_count / sample_count * 100) if sample_count > 0 else 0.0,
        'expert_out_of_lane_rate': float(expert_out_of_lane_count / sample_count * 100) if sample_count > 0 else 0.0,
        'period_summary': period_summary,
        'selected_scene': df.attrs.get('selected_scene', 'UNKNOWN'),
        'mean_gt_step_dist': float(np.mean(all_gt_step_dist)) if all_gt_step_dist else 0.0,
        'p95_gt_step_dist': float(np.percentile(all_gt_step_dist, 95)) if all_gt_step_dist else 0.0,
        'const_velocity_ADE': float(np.mean(all_const_ade)) if all_const_ade else 0.0,
        'avg_same_frame_count': float(np.mean(all_same_frame_counts)) if all_same_frame_counts else 0.0,
        'avg_collision_candidates': float(np.mean(all_collision_candidates)) if all_collision_candidates else 0.0,
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
        f.write(f"  使用场景:       {results['selected_scene']}\n")
        f.write(f"  目标车道:       Lane {TARGET_LANES}\n")
        f.write(f"  车道宽度:       {LANE_WIDTH} m\n")
        f.write(f"  预测步长:       {PREDICTION_HORIZON} 步 × {NGSIM_DT}s = {PREDICTION_HORIZON * NGSIM_DT}s\n")
        f.write(f"  最大样本数:     {MAX_EVAL_SAMPLES}\n")
        f.write(f"  分时段数量:     {NUM_EVAL_PERIODS}\n")
        f.write(f"  最大加载行数:   {MAX_DATA_ROWS}\n")
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

        f.write("【对齐诊断】\n")
        f.write(f"  真实单步位移均值:     {results['mean_gt_step_dist']:.4f} m / 0.1s\n")
        f.write(f"  真实单步位移P95:      {results['p95_gt_step_dist']:.4f} m / 0.1s\n")
        f.write(f"  常速基线ADE:          {results['const_velocity_ADE']:.4f} m\n")
        f.write(f"  平均同帧车辆数:       {results['avg_same_frame_count']:.2f}\n")
        f.write(f"  平均碰撞候选车辆数:   {results['avg_collision_candidates']:.2f}\n\n")

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

        f.write("【综合评分评估 (对标 OnSite 体系)】\n")
        f.write(f"  Expert 平均总分: {results['avg_expert_total_score']:.2f} / 100\n")
        f.write(f"  Expert 安全分:   {results['avg_expert_safety']:.2f} / 50\n")
        f.write(f"  Expert 效率分:   {results['avg_expert_efficiency']:.2f} / 30\n")
        f.write(f"  Expert 舒适分:   {results['avg_expert_comfort']:.2f} / 20\n")
        f.write(f"  Agent  平均总分: {results['avg_total_score']:.2f} / 100\n")
        f.write(f"  Agent  安全分:   {results['avg_safety']:.2f} / 50\n")
        f.write(f"  Agent  效率分:   {results['avg_efficiency']:.2f} / 30\n")
        f.write(f"  Agent  舒适分:   {results['avg_comfort']:.2f} / 20\n\n")

        f.write("【分时段 Expert / Agent 对照表】\n")
        f.write("时段    场景数   Expert安全  Expert效率  Expert舒适  Expert总分  Expert达成率   Agent安全  Agent效率  Agent舒适  Agent总分  Agent达成率\n")
        for row in results['period_summary']:
            f.write(
                f"{row['period']:<6}"
                f"{row['count']:>7d}"
                f"{row['expert_safety']:>11.2f}"
                f"{row['expert_efficiency']:>11.2f}"
                f"{row['expert_comfort']:>11.2f}"
                f"{row['expert_total']:>11.2f}"
                f"{row['expert_goal_rate']:>13.4f}"
                f"{row['agent_safety']:>11.2f}"
                f"{row['agent_efficiency']:>11.2f}"
                f"{row['agent_comfort']:>11.2f}"
                f"{row['agent_total']:>11.2f}"
                f"{row['agent_goal_rate']:>12.4f}\n"
            )
        f.write("\n")
        
        f.write("【安全违规统计】\n")
        f.write(f"  Expert 背景近距/重叠率: {results['expert_collision_rate']:.2f} %\n")
        f.write(f"  Agent 新增碰撞风险率:   {results['collision_rate']:.2f} %\n")
        f.write(f"  Expert 越界率:          {results['expert_out_of_lane_rate']:.2f} %\n")
        f.write(f"  Agent 预测越界率:       {results['out_of_lane_rate']:.2f} %\n\n")

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

def parse_args():
    parser = argparse.ArgumentParser(description="NGSIM Transformer 开环评估")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="覆盖 MAX_EVAL_SAMPLES；小样本诊断建议 50 或 100",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速诊断模式，等价于 --max-samples 100",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="固定使用某个 Scene_ID；不传则自动选择目标车道样本最多的场景",
    )
    parser.add_argument(
        "--max-data-rows",
        type=int,
        default=None,
        help="覆盖 CSV 读取后最多保留的数据行数；低内存环境建议 100000~800000",
    )
    return parser.parse_args()


def main():
    global MAX_EVAL_SAMPLES, PRIMARY_SCENE_ID, MAX_DATA_ROWS
    args = parse_args()
    if args.quick:
        MAX_EVAL_SAMPLES = 100
        MAX_DATA_ROWS = 100_000
    if args.max_samples is not None:
        MAX_EVAL_SAMPLES = args.max_samples
    if args.scene_id is not None:
        PRIMARY_SCENE_ID = args.scene_id
    if args.max_data_rows is not None:
        MAX_DATA_ROWS = args.max_data_rows

    log.info("=" * 70)
    log.info("   NGSIM 公开数据集 Transformer BC 模型 开环测试")
    log.info("=" * 70)
    log.info(f"日志文件: {LOG_FILE}")
    log.info(f"设备: {DEVICE}")
    log.info(f"模型: {MODEL_PATH}")
    log.info(f"NGSIM 数据: {NGSIM_CSV_PATH}")
    log.info(f"目标车道: {TARGET_LANES}")
    log.info(f"最大评估样本: {MAX_EVAL_SAMPLES}")
    log.info(f"最大加载行数: {MAX_DATA_ROWS}")
    log.info(f"场景选择: {'自动选择' if PRIMARY_SCENE_ID is None else PRIMARY_SCENE_ID}")
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
    log.info(f"  常速基线ADE:        {results['const_velocity_ADE']:.4f} m")
    log.info(f"  真实单步位移均值:   {results['mean_gt_step_dist']:.4f} m")
    log.info(f"  平均同帧车辆数:     {results['avg_same_frame_count']:.2f}")
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
