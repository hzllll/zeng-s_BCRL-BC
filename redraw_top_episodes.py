#!/usr/bin/env python3
"""
redraw_top_episodes.py — 精选高分回合可视化 + 详细报告

从已有 1000 集测试结果 CSV 中，每种场景选取 Top-20 高分无碰撞回合，
用保存的 seed 复现仿真场景，以改进后的可视化方案绘图，并生成详细 TXT 报告。

特性（相比 v2 的可视化方案改进）：
  ① 全集固定绝对坐标系：所有面板 x/y 轴范围相同，显示完整行驶轨迹
  ② 目标点：红色方块标注于图中实际 goal_x/goal_y 处
  ③ 自车预测轨迹：绿色小圆点显示当前推理的 5 步预测位置
  ④ 障碍车编号：按距自车距离从近到远编号 1,2,3...
  ⑤ 颜色语义：自车=绿，普通障碍=蓝，危险障碍（TTC<2s）=深蓝，目标点=红
  ⑥ 时间标注：面板标题显示 "t = X.Xs"
  ⑦ 不使用 set_aspect('equal')，车辆视觉长度放大至 12m 保证可见

用法：
    python redraw_top_episodes.py
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

try:
    from highway_env.vehicle.behavior import IDMVehicle
except ImportError:
    IDMVehicle = None

# ---- 中文字体 ----
matplotlib.font_manager._load_fontmanager(try_read_cache=False)
plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "Noto Sans CJK SC", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
#                         全 局 配 置
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_"
    "enc3_dec3_500es_CoAnWarmRest_zDATASET.pth",
)

SOURCE_CSV = os.path.join(
    BASE_DIR, "highway_env_results_v2", "0410_104509",
    "episode_details_0410_104509.csv",
)

D_MODEL        = 256
FFN_DIM        = 4 * D_MODEL
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 3
OUTPUT_DIM     = 2
SEQ_LENGTH     = 5
CAR_NUM        = 8

LANES_COUNT      = 3
LANE_WIDTH       = 4.0
VEHICLES_COUNT   = 20
SIM_FREQ         = 10
POLICY_FREQ      = 10
EPISODE_DURATION = 25
STEPS_PER_INFER  = 5
GOAL_LOOKAHEAD   = 500.0
TARGET_SPEED     = 25.0
DT               = 1.0 / POLICY_FREQ

TOP_N            = 5        # 每种场景精选数量
VIS_N_PANELS     = 4        # 每集面板数
VIS_WINDOW_HALF  = 200.0    # 每帧以自车为中心的半窗口宽度 (m), 总显示 400m
FIG_W            = 18.0     # 图幅宽度 (inches)
FIG_H_PER_PANEL  = 3.2      # 每面板高度 (inches)
VIS_TARGET_RATIO = 1.8      # 车辆视觉长宽比目标值 (长/宽)

SCENARIO_TYPES = ["following", "lane_change", "overtaking", "avoidance"]
SCENARIO_CN = {
    "following":   "跟车",
    "lane_change": "合适时机变道",
    "overtaking":  "超车",
    "avoidance":   "避让",
}

# =====================================================================
#              Transformer 模型定义（与训练脚本完全一致）
# =====================================================================
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
        return self.head(dec_out)


# =====================================================================
#                      车道映射 & 特征提取
# =====================================================================
def build_map_info():
    m = {}
    for i in range(LANES_COUNT):
        hw = LANES_COUNT - 1 - i
        cy = hw * LANE_WIDTH
        m[i] = {"left_bound": cy + LANE_WIDTH / 2,
                 "center": cy,
                 "right_bound": cy - LANE_WIDTH / 2}
    return m


def _ego_lane_bounds(ego_y, mi):
    if ego_y >= mi[0]["right_bound"]:
        return mi[0]["left_bound"] - ego_y, mi[0]["right_bound"] - ego_y
    if mi[1]["right_bound"] < ego_y < mi[1]["left_bound"]:
        return mi[1]["left_bound"] - ego_y, mi[1]["right_bound"] - ego_y
    return mi[2]["left_bound"] - ego_y, mi[2]["right_bound"] - ego_y


def _veh_lane_bounds(vy, ego_y, mi):
    if vy >= mi[0]["right_bound"]:
        return mi[0]["left_bound"] - ego_y, mi[0]["right_bound"] - ego_y
    if mi[1]["right_bound"] < vy < mi[1]["left_bound"]:
        return mi[1]["left_bound"] - ego_y, mi[1]["right_bound"] - ego_y
    if vy <= mi[2]["left_bound"]:
        return mi[2]["left_bound"] - ego_y, mi[2]["right_bound"] - ego_y
    return _ego_lane_bounds(ego_y, mi)


def extract_features(env, mi, goal_x, goal_y):
    ego = env.unwrapped.vehicle
    ex, ey = ego.position
    ev = ego.speed
    eyaw = (ego.heading + np.pi) % (2 * np.pi) - np.pi
    elen = ego.LENGTH

    states = [ev, eyaw, goal_x - ex, goal_y - ey,
              mi[0]["left_bound"] - ey, mi[2]["right_bound"] - ey]

    cands = []
    for v in env.unwrapped.road.vehicles[1:]:
        vx, vy = v.position
        if abs(vy - ey) > 6:
            continue
        dx = vx - ex
        hs = 0.5 * (v.LENGTH + elen)
        ld = dx - hs if dx - hs > 0 else (dx + hs if dx + hs < 0 else 0.0)
        d = np.sqrt(ld ** 2 + (vy - ey) ** 2)
        if d < 200:
            cands.append((d, ld, v))

    cands.sort(key=lambda t: t[0])
    sel = cands[:CAR_NUM]

    for _, ld, v in sel:
        vy = v.position[1]
        vyaw = (v.heading + np.pi) % (2 * np.pi) - np.pi
        states.extend([ld, vy - ey, v.speed - ev, vyaw])
        lb, rb = _veh_lane_bounds(vy, ey, mi)
        states.extend([lb, rb])

    elb, erb = _ego_lane_bounds(ey, mi)
    for _ in range(CAR_NUM - len(sel)):
        states.extend([200.0, 0.0, 0.0, 0.0, elb, erb])

    return states


# =====================================================================
#                           TTC 计算
# =====================================================================
def _min_ttc(ego, vehicles):
    best = float("inf")
    for v in vehicles:
        dx = v.position[0] - ego.position[0]
        if dx <= 0 or abs(v.position[1] - ego.position[1]) > 3.0:
            continue
        gap = dx - 0.5 * (ego.LENGTH + v.LENGTH)
        if gap <= 0:
            return 0.0
        cl = ego.speed * np.cos(ego.heading) - v.speed * np.cos(v.heading)
        if cl <= 0:
            continue
        best = min(best, gap / cl)
    return best


def _quick_ttc(ego, v):
    dx = v.position[0] - ego.position[0]
    if dx <= 0:
        return float("inf")
    gap = dx - 0.5 * (ego.LENGTH + v.LENGTH)
    if gap <= 0:
        return 0.0
    cl = ego.speed - v.speed
    return gap / cl if cl > 0 else float("inf")


# =====================================================================
#                          评 分 系 统
# =====================================================================
def compute_scores(d):
    crashed = d["crashed"]
    ttcs = [t for t in d["min_ttc"] if t < float("inf")]
    speeds = d["ego_speed"]
    acts = d["actions"]
    xs = d["ego_x"]
    ys = d["ego_y"]
    dur = d["total_steps"] / POLICY_FREQ

    col_pts = 0.0 if crashed else 35.0
    if ttcs:
        mt = np.mean(ttcs)
        ttc_pts = 10.0 if mt > 3 else (7.0 if mt > 2 else (4.0 if mt > 1 else 1.0))
    else:
        ttc_pts = 10.0
    boundary_max = (LANES_COUNT - 1) * LANE_WIDTH + LANE_WIDTH / 2
    viols = sum(1 for y in ys if y < -LANE_WIDTH / 2 or y > boundary_max)
    vr = viols / max(len(ys), 1)
    lane_pts = max(0.0, 5.0 * (1.0 - vr * 10))
    safety = col_pts + ttc_pts + lane_pts

    avg_spd = float(np.mean(speeds)) if speeds else 0.0
    speed_pts = min(avg_spd / TARGET_SPEED, 1.0) * 20.0
    dist = xs[-1] - xs[0] if len(xs) > 1 else 0.0
    exp_dist = TARGET_SPEED * dur
    dist_pts = min(dist / max(exp_dist, 1), 1.0) * 10.0
    efficiency = speed_pts + dist_pts

    if len(acts) > 1:
        accs = [a[0] for a in acts]
        strs = [a[1] for a in acts]
        jerks = [abs(accs[i+1] - accs[i]) / DT for i in range(len(accs)-1)]
        sr    = [abs(strs[i+1] - strs[i]) / DT for i in range(len(strs)-1)]
        aj  = float(np.mean(jerks))
        asr = float(np.mean(sr))
    else:
        aj = asr = 0.0
    jk_pts = 10.0 if aj < 2 else (7.0 if aj < 5 else (4.0 if aj < 10 else 1.0))
    st_pts = 10.0 if asr < 0.05 else (7.0 if asr < 0.1 else (4.0 if asr < 0.2 else 1.0))
    comfort = jk_pts + st_pts

    total = safety + efficiency + comfort

    n_lc = 0
    if len(ys) > 1:
        lanes = [int(round(y / LANE_WIDTH)) for y in ys]
        n_lc = sum(1 for i in range(1, len(lanes)) if lanes[i] != lanes[i-1])

    return {
        "safety":       round(safety, 2),
        "efficiency":   round(efficiency, 2),
        "comfort":      round(comfort, 2),
        "total":        round(total, 2),
        "avg_speed":    round(avg_spd, 2),
        "distance":     round(dist, 1),
        "mean_ttc":     round(float(np.mean(ttcs)), 2) if ttcs else 999.0,
        "min_ttc_val":  round(float(min(ttcs)), 2) if ttcs else 999.0,
        "avg_jerk":     round(aj, 4),
        "avg_steer_rate": round(asr, 4),
        "n_lane_changes": n_lc,
        "goal_achieved": not crashed and d["total_steps"] >= EPISODE_DURATION * POLICY_FREQ * 0.95,
    }


# =====================================================================
#                       场 景 构 造（可复现版）
# =====================================================================
def setup_scenario_reproducible(env, stype, env_seed):
    """
    使用 env_seed 复现场景。
    env.reset(seed=env_seed) 复现 highway-env 的初始场景布局；
    场景额外布置使用由 env_seed 派生的独立 rng，保证每次调用结果相同。
    """
    env.reset(seed=env_seed)
    rng = np.random.default_rng(env_seed)  # 从相同 seed 派生，保证可复现

    ego   = env.unwrapped.vehicle
    road  = env.unwrapped.road
    ex, ey = ego.position
    elane = int(np.clip(round(ey / LANE_WIDTH), 0, LANES_COUNT - 1))

    if stype == "following":
        placed = False
        for v in road.vehicles[1:]:
            if abs(v.position[1] - ey) < 2.0 and 20 < v.position[0] - ex < 70:
                v.speed = rng.uniform(16, 21)
                if hasattr(v, "target_speed"):
                    v.target_speed = v.speed
                placed = True
                break
        if not placed and IDMVehicle is not None:
            lo = road.network.get_lane(("0", "1", elane))
            lx = ex + rng.uniform(35, 55)
            lv = IDMVehicle(road, lo.position(lx, 0),
                            heading=lo.heading_at(lx),
                            speed=rng.uniform(16, 21))
            road.vehicles.append(lv)
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH

    elif stype == "lane_change":
        avail = [l for l in range(LANES_COUNT) if l != elane]
        tgt = int(rng.choice(avail))
        gx, gy = ex + GOAL_LOOKAHEAD, tgt * LANE_WIDTH

    elif stype == "overtaking":
        if IDMVehicle is not None:
            lo = road.network.get_lane(("0", "1", elane))
            sx = ex + rng.uniform(30, 50)
            sv = IDMVehicle(road, lo.position(sx, 0),
                            heading=lo.heading_at(sx),
                            speed=rng.uniform(10, 15))
            road.vehicles.append(sv)
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH

    elif stype == "avoidance":
        placed = False
        for v in road.vehicles[1:]:
            if abs(v.position[1] - ey) < 2.0 and v.position[0] > ex:
                v.position = np.array([ex + rng.uniform(15, 25), ey])
                v.speed = max(5.0, ego.speed - rng.uniform(3, 10))
                if hasattr(v, "target_speed"):
                    v.target_speed = max(5.0, v.speed - 5)
                try:
                    v.lane_index = road.network.get_closest_lane_index(
                        v.position, v.heading)
                    v.lane = road.network.get_lane(v.lane_index)
                except Exception:
                    pass
                placed = True
                break
        if not placed and IDMVehicle is not None:
            lo = road.network.get_lane(("0", "1", elane))
            cx = ex + rng.uniform(15, 25)
            cv = IDMVehicle(road, lo.position(cx, 0),
                            heading=lo.heading_at(cx),
                            speed=max(5.0, ego.speed - rng.uniform(3, 10)))
            road.vehicles.append(cv)
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH
    else:
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH

    return float(gx), float(gy)


def _make_env():
    return gym.make("highway-v0", config={
        "observation": {
            "type": "Kinematics", "vehicles_count": 9,
            "features": ["x", "y", "vx", "vy", "heading"],
            "absolute": True, "normalize": False,
        },
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": (-6, 6),
            "steering_range": (-0.15, 0.15),
        },
        "lanes_count": LANES_COUNT,
        "vehicles_count": VEHICLES_COUNT,
        "duration": EPISODE_DURATION,
        "simulation_frequency": SIM_FREQ,
        "policy_frequency": POLICY_FREQ,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "collision_reward": -1.0,
        "reward_speed_range": [20, 30],
        "offscreen_rendering": True,
    })


# =====================================================================
#              自车预测轨迹（简化运动学正向推演）
# =====================================================================
def compute_pred_traj(x, y, speed, heading, aseq_np):
    """
    用简化自行车模型推演自车未来 5 步位置。
    aseq_np shape: (5, 2)，值域 [-1, 1]（模型 Tanh 输出）
    highway-env ContinuousAction 映射：
      acc   = action[0] * 6      → [-6, 6] m/s²
      steer = action[1] * 0.15   → [-0.15, 0.15] rad
    """
    WHEELBASE = 2.5
    points = []
    for i in range(len(aseq_np)):
        acc   = float(aseq_np[i, 0]) * 6
        steer = float(aseq_np[i, 1]) * 0.15
        x       = x + speed * np.cos(heading) * DT
        y       = y + speed * np.sin(heading) * DT
        heading = heading + (speed / WHEELBASE) * np.tan(steer) * DT
        speed   = max(0.0, speed + acc * DT)
        points.append((float(x), float(y)))
    return points


# =====================================================================
#                       快 照（改进版 v3）
# =====================================================================
def _snapshot_v3(env, step, aseq_np, goal_x, goal_y):
    ego  = env.unwrapped.vehicle
    ex, ey = ego.position
    ev, eh = ego.speed, ego.heading

    pred_traj = compute_pred_traj(ex, ey, ev, eh, aseq_np)

    # 道路 y 范围（用于过滤道路外车辆）
    road_y_lo = -(LANE_WIDTH / 2 + 0.5)
    road_y_hi = (LANES_COUNT - 1) * LANE_WIDTH + LANE_WIDTH / 2 + 0.5

    # 障碍车：只保留道路 y 范围内的，按距自车距离从近到远排序
    others = []
    for i, v in enumerate(env.unwrapped.road.vehicles):
        if i == 0:
            continue
        vx, vy = v.position
        if vy < road_y_lo or vy > road_y_hi:
            continue  # 过滤道路外车辆
        dist = np.hypot(vx - ex, vy - ey)
        others.append((dist, v))
    others.sort(key=lambda t: t[0])

    vehs = [{
        "x": float(ex), "y": float(ey), "heading": float(eh),
        "speed": float(ev), "width": float(ego.WIDTH),
        "color": "limegreen", "label": None, "is_ego": True,
    }]
    for label_num, (_, v) in enumerate(others[:10], start=1):
        vx, vy = v.position
        ttc = _quick_ttc(ego, v)
        color = "navy" if (ttc < 2.0 and vx > ex) else "royalblue"
        vehs.append({
            "x": float(vx), "y": float(vy), "heading": float(v.heading),
            "speed": float(v.speed), "width": float(v.WIDTH),
            "color": color, "label": str(label_num), "is_ego": False,
        })

    return {
        "step":      step,
        "time_s":    round(step * DT, 1),
        "ego_x":     float(ex),
        "ego_y":     float(ey),
        "goal_x":    float(goal_x),
        "goal_y":    float(goal_y),
        "vehicles":  vehs,
        "pred_traj": pred_traj,
    }


# =====================================================================
#                       单 集 仿 真（v3）
# =====================================================================
def run_episode_v3(env, model, mi, stype, ep_rank, env_seed):
    gx, gy = setup_scenario_reproducible(env, stype, env_seed)
    max_steps = EPISODE_DURATION * POLICY_FREQ

    # 选取快照步：在推理边界（STEPS_PER_INFER 的倍数）中等距选 VIS_N_PANELS 个
    n_infer = max_steps // STEPS_PER_INFER
    vis_steps = set()
    for k in range(VIS_N_PANELS):
        inf_idx = int(round(k * (n_infer - 1) / (VIS_N_PANELS - 1)))
        vis_steps.add(inf_idx * STEPS_PER_INFER)

    d = {
        "scenario_type": stype,
        "episode_rank":  ep_rank,
        "seed":          env_seed,
        "goal_x": gx, "goal_y": gy,
        "crashed": False, "total_steps": 0, "total_reward": 0.0,
        "inference_times": [],
        "ego_x": [], "ego_y": [], "ego_speed": [], "ego_heading": [],
        "actions": [], "rewards": [], "min_ttc": [],
        "vis_frames": [],
    }

    step = 0
    terminated = truncated = False
    info = {}

    while step < max_steps and not terminated and not truncated:
        feats = extract_features(env, mi, gx, gy)
        ft = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        tic = time.perf_counter()
        with torch.no_grad():
            aseq = model(ft)
        d["inference_times"].append((time.perf_counter() - tic) * 1000)

        aseq_np = aseq[0].cpu().numpy()  # (5, 2)

        # 在推理后、执行前拍快照（含预测轨迹）
        if step in vis_steps:
            d["vis_frames"].append(_snapshot_v3(env, step, aseq_np, gx, gy))

        for i in range(STEPS_PER_INFER):
            if step >= max_steps or terminated or truncated:
                break
            act = aseq_np[i]
            obs, rew, terminated, truncated, info = env.step(act)
            ego = env.unwrapped.vehicle
            d["ego_x"].append(float(ego.position[0]))
            d["ego_y"].append(float(ego.position[1]))
            d["ego_speed"].append(float(ego.speed))
            d["ego_heading"].append(float(ego.heading))
            d["actions"].append((float(act[0]), float(act[1])))
            d["rewards"].append(float(rew))
            d["total_reward"] += rew
            d["min_ttc"].append(_min_ttc(ego, env.unwrapped.road.vehicles[1:]))
            step += 1

    # 碰撞时补拍终止帧（保留完整过程记录）
    if terminated and len(d["vis_frames"]) < VIS_N_PANELS:
        aseq_zero = np.zeros((SEQ_LENGTH, OUTPUT_DIM))
        d["vis_frames"].append(_snapshot_v3(env, step, aseq_zero, gx, gy))

    d["crashed"]     = bool(info.get("crashed", terminated)) if terminated else False
    d["total_steps"] = step
    d["scores"]      = compute_scores(d)
    return d


# =====================================================================
#                      改 进 可 视 化（v3）
# =====================================================================
def draw_episode_v3(ep, save_path, cn_name, rank, orig_ep_idx):
    vis = ep["vis_frames"]
    if not vis:
        return

    n_panels = len(vis)

    # ---- 全集固定 y 轴范围（与道路一致）----
    y_lo = -LANE_WIDTH / 2 - 0.8
    y_hi = (LANES_COUNT - 1) * LANE_WIDTH + LANE_WIDTH / 2 + 0.8
    y_range = y_hi - y_lo

    # ---- 全集固定 x 轴范围（基于完整 ego 轨迹 + margin）----
    all_ego_x = ep.get("ego_x", [])
    goal_x    = ep["goal_x"]
    x_lo = (min(all_ego_x) if all_ego_x else vis[0]["ego_x"]) - 80.0
    x_hi = max(
        (max(all_ego_x) if all_ego_x else vis[-1]["ego_x"]) + 80.0,
        goal_x + 20.0,
    )
    x_range = x_hi - x_lo

    # ---- 动态计算车辆绘图长度，使视觉长宽比 ≈ VIS_TARGET_RATIO ----
    # visual_length_in / visual_width_in = VIS_TARGET_RATIO
    # => VIS_VEH_LENGTH = VIS_TARGET_RATIO × veh_width × (x_range/FIG_W) / (y_range/FIG_H_PER_PANEL)
    veh_width_ref = 2.0  # highway-env 标准车宽 (m)
    VIS_VEH_LENGTH = VIS_TARGET_RATIO * veh_width_ref * (x_range / FIG_W) / (y_range / FIG_H_PER_PANEL)
    # 上限：固定 12m，防止避让/跟车等近距场景中矩形严重重叠
    VIS_VEH_LENGTH = min(VIS_VEH_LENGTH, 12.0)
    # 动态绘图车宽：使视觉长宽比恒为 1.3:1（车在 x 方向视觉长度 > y 方向）
    VIS_VEH_WIDTH = min((VIS_VEH_LENGTH / 1.3) * (y_range / FIG_H_PER_PANEL) / (x_range / FIG_W), 2.0)

    fig, axes = plt.subplots(n_panels, 1, figsize=(FIG_W, FIG_H_PER_PANEL * n_panels))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, (ax, frame) in enumerate(zip(axes, vis)):
        gx_f = frame["goal_x"]
        gy_f = frame["goal_y"]

        # ---- 道路背景 ----
        ax.set_facecolor("#f5f5f5")
        for li in range(LANES_COUNT + 1):
            yy = li * LANE_WIDTH - LANE_WIDTH / 2
            lw  = 1.5 if (li == 0 or li == LANES_COUNT) else 0.8
            col = "black" if (li == 0 or li == LANES_COUNT) else "#777777"
            ax.axhline(yy, color=col, lw=lw, ls="-", zorder=1)
        for li in range(LANES_COUNT):
            yc = li * LANE_WIDTH
            ax.axhline(yc, color="#cccccc", lw=0.4, ls=":", zorder=1)

        # ---- 目标点（红色方块，只在显示范围内绘制）----
        if x_lo - VIS_VEH_LENGTH <= gx_f <= x_hi + VIS_VEH_LENGTH:
            ax.add_patch(mpatches.Rectangle(
                (gx_f - VIS_VEH_LENGTH / 2, gy_f - LANE_WIDTH / 2),
                VIS_VEH_LENGTH, LANE_WIDTH,
                fc="red", ec="#990000", lw=0.8, alpha=0.92, zorder=5,
            ))
            ax.text(gx_f, gy_f, "目标", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=6)

        # ---- 车辆（矩形）----
        for veh in frame["vehicles"]:
            vx, vy = veh["x"], veh["y"]
            # 只绘制当前 x 范围内的车辆
            if vx < x_lo - VIS_VEH_LENGTH or vx > x_hi + VIS_VEH_LENGTH:
                continue
            vw = VIS_VEH_WIDTH
            vl = VIS_VEH_LENGTH
            ax.add_patch(mpatches.Rectangle(
                (vx - vl / 2, vy - vw / 2),
                vl, vw,
                fc=veh["color"], ec="black", lw=0.5, alpha=0.88, zorder=4,
            ))
            if not veh["is_ego"] and veh["label"] is not None:
                ax.text(vx, vy + vw / 2 + 0.3, veh["label"],
                        fontsize=7, ha="center", va="bottom",
                        color="#222222", fontweight="bold", zorder=7)

        # ---- 自车预测轨迹点（绿色小圆点）----
        pred = frame.get("pred_traj", [])
        if pred:
            px_list = [p[0] for p in pred]
            py_list = [p[1] for p in pred]
            ax.scatter(px_list, py_list,
                       color="limegreen", s=14, zorder=8,
                       edgecolors="darkgreen", linewidths=0.5, alpha=0.9)

        # 四帧共享同一 x/y 范围
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y/m", fontsize=8)
        ax.set_title(
            f"t = {frame['time_s']:.1f}s（帧: {frame['step']}）",
            fontsize=9, loc="left", pad=3,
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x方向位置/m", fontsize=8)

    sc = ep.get("scores", {})
    status = "碰撞" if ep["crashed"] else "安全"
    fig.suptitle(
        f"{cn_name}场景 | 精选 #{rank}（原始 EP {orig_ep_idx + 1}）| {status} | "
        f"Total={sc.get('total', 0):.1f}  "
        f"Safety={sc.get('safety', 0):.1f}  "
        f"Efficiency={sc.get('efficiency', 0):.1f}  "
        f"Comfort={sc.get('comfort', 0):.1f}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
#                         详 细 TXT 报 告
# =====================================================================
def generate_detail_report(all_results, selected_orig_rows, save_path):
    # 读取基线数据
    baseline = []
    try:
        with open(SOURCE_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                baseline.append(row)
    except Exception:
        pass

    with open(save_path, "w", encoding="utf-8") as f:
        sep = "=" * 90
        sep2 = "-" * 90

        f.write(sep + "\n")
        f.write("Highway-env Transformer BC — 精选高分回合详细报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型    : {MODEL_PATH}\n")
        f.write(f"基线来源: {SOURCE_CSV}\n")
        f.write(f"精选方式: 每种场景 Top-{TOP_N} 高分无碰撞回合（按 Total 降序）\n")
        f.write(f"共计    : {len(all_results)} 集（{len(SCENARIO_TYPES)} 场景 × {TOP_N}）\n")
        f.write(sep + "\n\n")

        # ---- 基线 vs 精选对比 ----
        f.write("一、基线（原始 1000 集）vs 精选对比\n")
        f.write(sep2 + "\n")
        f.write(f"  {'指标':<22} {'原始 1000 集':>14} {'精选 80 集':>14}\n")
        f.write(sep2 + "\n")

        def _base(key):
            vals = [float(r[key]) for r in baseline if r.get(key, "")]
            return float(np.mean(vals)) if vals else 0.0

        def _sel(key):
            vals = [r["scores"][key] for r in all_results]
            return float(np.mean(vals)) if vals else 0.0

        for label, bk, sk in [
            ("平均总分 (/100)", "total",      "total"),
            ("平均安全分 (/50)", "safety",    "safety"),
            ("平均效率分 (/30)", "efficiency","efficiency"),
            ("平均舒适分 (/20)", "comfort",   "comfort"),
            ("平均速度 (m/s)",  "avg_speed", "avg_speed"),
            ("平均TTC (s)",     "mean_ttc",  "mean_ttc"),
        ]:
            f.write(f"  {label:<22} {_base(bk):>14.2f} {_sel(sk):>14.2f}\n")

        base_crash = sum(1 for r in baseline if r.get("crashed", "0") == "1")
        sel_crash  = sum(1 for r in all_results if r["crashed"])
        f.write(f"  {'碰撞率':<22} {100*base_crash/max(len(baseline),1):>13.1f}%"
                f" {100*sel_crash/max(len(all_results),1):>13.1f}%\n")
        f.write(sep2 + "\n\n")

        # ---- 分场景汇总 ----
        f.write("二、分场景精选汇总\n")
        f.write(sep2 + "\n")
        hdr = (f"  {'场景':<12} {'集数':>5} {'碰撞':>4} {'安全率':>7} "
               f"{'达成率':>7} {'Safety':>7} {'Effic':>7} "
               f"{'Comft':>7} {'Total':>7} {'速度':>6} {'TTC':>7}\n")
        f.write(hdr)
        f.write(sep2 + "\n")
        for st in SCENARIO_TYPES:
            sr = [r for r in all_results if r["scenario_type"] == st]
            if not sr:
                continue
            sc_list = [r["scores"] for r in sr]
            cr  = sum(1 for r in sr if r["crashed"])
            ga  = sum(1 for r in sr if r["scores"]["goal_achieved"])
            sn  = len(sr)
            f.write(
                f"  {SCENARIO_CN[st]:<12} {sn:>5} {cr:>4} "
                f"{100*(sn-cr)/sn:>6.1f}% {100*ga/sn:>6.1f}% "
                f"{np.mean([s['safety'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['efficiency'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['comfort'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['total'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['avg_speed'] for s in sc_list]):>6.1f} "
                f"{np.mean([s['mean_ttc'] for s in sc_list]):>7.1f}\n"
            )
        f.write(sep2 + "\n\n")

        # ---- 分场景逐集详情 ----
        f.write("三、分场景逐集详细结果\n")

        for st in SCENARIO_TYPES:
            sr    = [r for r in all_results if r["scenario_type"] == st]
            orig_r = [row for row in selected_orig_rows
                      if row.get("scenario_type") == st]
            if not sr:
                continue

            cn = SCENARIO_CN[st]
            f.write("\n" + sep + "\n")
            f.write(f"【{cn}场景】精选 Top-{TOP_N} 回合\n")
            f.write(sep + "\n")

            sc_list = [r["scores"] for r in sr]
            cr  = sum(1 for r in sr if r["crashed"])
            ga  = sum(1 for r in sr if r["scores"]["goal_achieved"])
            all_inf = [t for r in sr for t in r["inference_times"]]
            f.write(f"  集数={len(sr)}  碰撞={cr}  目标达成={ga}\n")
            f.write(f"  Safety: {np.mean([s['safety'] for s in sc_list]):.2f}/50  "
                    f"Efficiency: {np.mean([s['efficiency'] for s in sc_list]):.2f}/30  "
                    f"Comfort: {np.mean([s['comfort'] for s in sc_list]):.2f}/20  "
                    f"Total: {np.mean([s['total'] for s in sc_list]):.2f}/100\n")
            f.write(f"  平均速度: {np.mean([s['avg_speed'] for s in sc_list]):.1f} m/s  "
                    f"平均TTC: {np.mean([s['mean_ttc'] for s in sc_list]):.1f}s  "
                    f"平均变道: {np.mean([s['n_lane_changes'] for s in sc_list]):.1f}次\n")
            if all_inf:
                f.write(f"  推理耗时: 均值={np.mean(all_inf):.3f}ms  "
                        f"P50={np.percentile(all_inf,50):.3f}ms  "
                        f"P95={np.percentile(all_inf,95):.3f}ms\n")
            f.write("\n")

            # 逐集表格头
            f.write(
                f"  {'#':>4} {'原EP':>6} {'Seed':>10} "
                f"{'Safety':>7} {'Effic':>7} {'Comft':>7} {'Total':>7} "
                f"{'速度':>6} {'TTC均':>7} {'TTC最小':>8} "
                f"{'变道':>5} {'步数':>5} {'推理ms':>8} {'状态'}\n"
            )
            f.write("  " + "-" * 86 + "\n")

            for rank_i, (r, orig) in enumerate(zip(sr, orig_r), start=1):
                sc = r["scores"]
                avg_inf = (float(np.mean(r["inference_times"]))
                           if r["inference_times"] else 0.0)
                orig_ep = int(orig.get("episode_idx", 0))
                status  = "碰撞" if r["crashed"] else "安全"
                f.write(
                    f"  {rank_i:>4} {orig_ep+1:>6} {r['seed']:>10} "
                    f"{sc['safety']:>7.2f} {sc['efficiency']:>7.2f} "
                    f"{sc['comfort']:>7.2f} {sc['total']:>7.2f} "
                    f"{sc['avg_speed']:>6.1f} {sc['mean_ttc']:>7.1f} "
                    f"{sc['min_ttc_val']:>8.1f} "
                    f"{sc['n_lane_changes']:>5} {r['total_steps']:>5} "
                    f"{avg_inf:>8.3f} {status}\n"
                )

            # 该场景最佳/最差/中位简评
            safe_sr = [r for r in sr if not r["crashed"]]
            if safe_sr:
                best_r = max(safe_sr, key=lambda r: r["scores"]["total"])
                worst_r = min(safe_sr, key=lambda r: r["scores"]["total"])
                med_r = sorted(safe_sr, key=lambda r: r["scores"]["total"])[len(safe_sr)//2]
                f.write(f"\n  ★ 最佳: EP {best_r['episode_idx']+1}  "
                        f"Total={best_r['scores']['total']:.1f}  "
                        f"速度={best_r['scores']['avg_speed']:.1f}m/s\n")
                f.write(f"  ★ 中位: EP {med_r['episode_idx']+1}  "
                        f"Total={med_r['scores']['total']:.1f}  "
                        f"速度={med_r['scores']['avg_speed']:.1f}m/s\n")
                f.write(f"  ★ 最难: EP {worst_r['episode_idx']+1}  "
                        f"Total={worst_r['scores']['total']:.1f}  "
                        f"速度={worst_r['scores']['avg_speed']:.1f}m/s\n")

        f.write("\n" + sep + "\n")
        f.write("报告结束\n")
        f.write(sep + "\n")


# =====================================================================
#                            主 函 数
# =====================================================================
def main():
    ts = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join(BASE_DIR, "highway_env_top20_vis", ts)
    for st in SCENARIO_TYPES:
        os.makedirs(os.path.join(save_dir, st), exist_ok=True)

    print("=" * 70)
    print(f"精选高分回合可视化 — Top-{TOP_N}/场景")
    print(f"基线 CSV : {SOURCE_CSV}")
    print(f"输出目录 : {save_dir}")
    print("=" * 70)

    if not os.path.exists(SOURCE_CSV):
        print(f"[错误] 找不到基线 CSV: {SOURCE_CSV}")
        sys.exit(1)

    # ---- 1. 读取基线 CSV，每种场景选 Top-N ----
    rows_by_type = {st: [] for st in SCENARIO_TYPES}
    with open(SOURCE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = row.get("scenario_type", "")
            if st in rows_by_type:
                rows_by_type[st].append(row)

    selected_by_type = {}
    print("\n精选统计:")
    for st in SCENARIO_TYPES:
        safe = [r for r in rows_by_type[st] if r["crashed"] == "0"]
        safe.sort(key=lambda r: float(r["total"]), reverse=True)
        chosen = safe[:TOP_N]
        selected_by_type[st] = chosen
        print(f"  {SCENARIO_CN[st]}: {len(chosen)} 集  "
              f"Total 范围 [{float(chosen[-1]['total']):.1f}, "
              f"{float(chosen[0]['total']):.1f}]")

    # ---- 2. 加载模型 ----
    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH,
        car_num=CAR_NUM, nhead=8,
        num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS,
        dim_feedforward=FFN_DIM, dropout=0.0,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型: {MODEL_PATH}")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"\n模型加载成功  设备: {DEVICE}\n")

    mi  = build_map_info()
    env = _make_env()

    all_results      = []
    all_orig_rows    = []
    total_to_run     = sum(len(v) for v in selected_by_type.values())
    run_count        = 0
    t_start          = time.time()

    for st in SCENARIO_TYPES:
        chosen = selected_by_type[st]
        print(f"\n{'=' * 70}")
        print(f"场景: {SCENARIO_CN[st]} ({st})  —  {len(chosen)} 集")
        print(f"{'=' * 70}")

        for rank_i, orig_row in enumerate(chosen, start=1):
            env_seed    = int(orig_row["seed"])
            orig_ep_idx = int(orig_row["episode_idx"])
            run_count  += 1

            try:
                ep = run_episode_v3(env, model, mi, st, rank_i, env_seed)
            except Exception as exc:
                print(f"  [WARN] 排名#{rank_i} (原EP{orig_ep_idx+1}) 异常: {exc}")
                continue

            ep["episode_idx"] = orig_ep_idx
            sc     = ep["scores"]
            status = "碰撞" if ep["crashed"] else "安全"
            avg_inf = float(np.mean(ep["inference_times"])) if ep["inference_times"] else 0

            print(
                f"  #{rank_i:2d}/{TOP_N}  原EP={orig_ep_idx+1:4d} | {status} | "
                f"T={sc['total']:5.1f} S={sc['safety']:5.1f} "
                f"E={sc['efficiency']:5.1f} C={sc['comfort']:5.1f} | "
                f"spd={sc['avg_speed']:.1f}m/s | "
                f"infer={avg_inf:.2f}ms  [{run_count}/{total_to_run}]"
            )

            # 绘图
            fname = (f"rank{rank_i:02d}_ep{orig_ep_idx+1:04d}"
                     f"_T{sc['total']:.0f}.png")
            fpath = os.path.join(save_dir, st, fname)
            draw_episode_v3(ep, fpath, SCENARIO_CN[st], rank_i, orig_ep_idx)

            all_results.append(ep)
            all_orig_rows.append(orig_row)

    env.close()
    elapsed = time.time() - t_start

    # ---- 3. 生成详细报告 ----
    report_path = os.path.join(save_dir, f"detail_report_{ts}.txt")
    generate_detail_report(all_results, all_orig_rows, report_path)

    print(f"\n{'=' * 70}")
    print(f"完成！共绘制 {len(all_results)} 张可视化图  用时 {elapsed/60:.1f} 分钟")
    print(f"输出目录: {save_dir}")
    print(f"报告    : {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
