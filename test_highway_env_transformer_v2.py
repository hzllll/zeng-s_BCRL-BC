#!/usr/bin/env python3
"""
Highway-env 综合测试脚本 v2
支持多场景类型（跟车/变道/超车/避让）、细化评分（安全/效率/舒适/总分）、
每集可视化、代表性标记、完整详细日志。

用法:
    python test_highway_env_transformer_v2.py             # 完整 1000 集
    python test_highway_env_transformer_v2.py --quick      # 快速验证 (每场景 4 集)
    python test_highway_env_transformer_v2.py -n 50        # 自定义每场景集数
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from datetime import datetime

try:
    from highway_env.vehicle.behavior import IDMVehicle
except ImportError:
    IDMVehicle = None

# ====================== 中文字体配置 ======================
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

D_MODEL        = 256
FFN_DIM        = 4 * D_MODEL  # 1024
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
EPISODE_DURATION = 25          # 秒
STEPS_PER_INFER  = 5
GOAL_LOOKAHEAD   = 500.0
TARGET_SPEED     = 25.0        # 评分参考速度 (m/s)

SCENARIO_TYPES = ["following", "lane_change", "overtaking", "avoidance"]
SCENARIO_CN = {
    "following":   "跟车",
    "lane_change": "合适时机变道",
    "overtaking":  "超车",
    "avoidance":   "避让",
}
EPISODES_PER_SCENARIO = 250    # 默认每场景 250 集 (×4 = 1000)

VIS_PANELS = 4                 # 每集可视化面板数

# =====================================================================
#                      Transformer 模型定义
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
#                        车道映射 & 特征提取
# =====================================================================
def build_map_info(lanes_count=LANES_COUNT, lane_width=LANE_WIDTH):
    """highway-env lane 结构 → Onsite 风格 map_info (索引 0 = 最高 y 车道)"""
    m = {}
    for i in range(lanes_count):
        hw = lanes_count - 1 - i
        cy = hw * lane_width
        m[i] = {"left_bound": cy + lane_width / 2,
                "center": cy,
                "right_bound": cy - lane_width / 2}
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
    """54 维特征向量，与 Onsite 训练数据格式对齐"""
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


# =====================================================================
#                          评 分 系 统
# =====================================================================
def compute_scores(d):
    """返回 Safety(50) + Efficiency(30) + Comfort(20) = Total(100)"""
    crashed = d["crashed"]
    ttcs = [t for t in d["min_ttc"] if t < float("inf")]
    speeds = d["ego_speed"]
    acts = d["actions"]
    xs = d["ego_x"]
    ys = d["ego_y"]
    dur = d["total_steps"] / POLICY_FREQ
    dt = 1.0 / POLICY_FREQ

    # ---- Safety (50) ----
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

    # ---- Efficiency (30) ----
    avg_spd = float(np.mean(speeds)) if speeds else 0.0
    speed_pts = min(avg_spd / TARGET_SPEED, 1.0) * 20.0
    dist = xs[-1] - xs[0] if len(xs) > 1 else 0.0
    exp_dist = TARGET_SPEED * dur
    dist_pts = min(dist / max(exp_dist, 1), 1.0) * 10.0
    efficiency = speed_pts + dist_pts

    # ---- Comfort (20) ----
    if len(acts) > 1:
        accs  = [a[0] for a in acts]
        strs  = [a[1] for a in acts]
        jerks = [abs(accs[i + 1] - accs[i]) / dt for i in range(len(accs) - 1)]
        sr    = [abs(strs[i + 1] - strs[i]) / dt for i in range(len(strs) - 1)]
        aj = float(np.mean(jerks))
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
        n_lc = sum(1 for i in range(1, len(lanes)) if lanes[i] != lanes[i - 1])

    return {
        "safety": round(safety, 2),
        "efficiency": round(efficiency, 2),
        "comfort": round(comfort, 2),
        "total": round(total, 2),
        "avg_speed": round(avg_spd, 2),
        "distance": round(dist, 1),
        "mean_ttc": round(float(np.mean(ttcs)), 2) if ttcs else 999.0,
        "min_ttc_val": round(float(min(ttcs)), 2) if ttcs else 999.0,
        "avg_jerk": round(aj, 4),
        "avg_steer_rate": round(asr, 4),
        "n_lane_changes": n_lc,
        "goal_achieved": not crashed and d["total_steps"] >= EPISODE_DURATION * POLICY_FREQ * 0.95,
    }


# =====================================================================
#                         场 景 构 造
# =====================================================================
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


def setup_scenario(env, stype, rng):
    """重置环境并按场景类型布置交通，返回 (goal_x, goal_y, seed)"""
    seed = int(rng.integers(100000))
    env.reset(seed=seed)
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
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
            lane_obj = road.network.get_lane(("0", "1", elane))
            lx = ex + rng.uniform(35, 55)
            lv = IDMVehicle(road, lane_obj.position(lx, 0),
                            heading=lane_obj.heading_at(lx),
                            speed=rng.uniform(16, 21))
            road.vehicles.append(lv)
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH

    elif stype == "lane_change":
        avail = [l for l in range(LANES_COUNT) if l != elane]
        tgt = int(rng.choice(avail))
        gx, gy = ex + GOAL_LOOKAHEAD, tgt * LANE_WIDTH

    elif stype == "overtaking":
        if IDMVehicle is not None:
            lane_obj = road.network.get_lane(("0", "1", elane))
            sx = ex + rng.uniform(30, 50)
            sv = IDMVehicle(road, lane_obj.position(sx, 0),
                            heading=lane_obj.heading_at(sx),
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
            lane_obj = road.network.get_lane(("0", "1", elane))
            cx = ex + rng.uniform(15, 25)
            cv = IDMVehicle(road, lane_obj.position(cx, 0),
                            heading=lane_obj.heading_at(cx),
                            speed=max(5.0, ego.speed - rng.uniform(3, 10)))
            road.vehicles.append(cv)
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH
    else:
        gx, gy = ex + GOAL_LOOKAHEAD, elane * LANE_WIDTH

    return gx, gy, seed


# =====================================================================
#                       单 集 运 行 & 快 照
# =====================================================================
def _snapshot(env, step):
    """记录当前帧所有车辆位置"""
    ego = env.unwrapped.vehicle
    vehs = []
    for idx, v in enumerate(env.unwrapped.road.vehicles):
        is_ego = (idx == 0)
        color = "green"
        if not is_ego:
            dx = v.position[0] - ego.position[0]
            dy = abs(v.position[1] - ego.position[1])
            color = "red" if (dx > 0 and dy < 3.0 and
                              _quick_ttc(ego, v) < 2.0) else "royalblue"
        vehs.append({
            "x": float(v.position[0]), "y": float(v.position[1]),
            "heading": float(v.heading), "speed": float(v.speed),
            "length": float(v.LENGTH), "width": float(v.WIDTH),
            "color": color,
        })
    return {"step": step, "ego_x": float(ego.position[0]), "vehicles": vehs}


def _quick_ttc(ego, v):
    dx = v.position[0] - ego.position[0]
    if dx <= 0:
        return float("inf")
    gap = dx - 0.5 * (ego.LENGTH + v.LENGTH)
    if gap <= 0:
        return 0.0
    cl = ego.speed - v.speed
    return gap / cl if cl > 0 else float("inf")


def run_episode(env, model, mi, stype, ep_idx, rng):
    gx, gy, seed = setup_scenario(env, stype, rng)
    max_steps = EPISODE_DURATION * POLICY_FREQ

    vis_indices = set()
    if max_steps >= VIS_PANELS:
        for k in range(VIS_PANELS):
            vis_indices.add(int(k * (max_steps - 1) / (VIS_PANELS - 1)))
    else:
        vis_indices = set(range(max_steps))

    d = {
        "scenario_type": stype, "episode_idx": ep_idx, "seed": seed,
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
        if step in vis_indices:
            d["vis_frames"].append(_snapshot(env, step))

        feats = extract_features(env, mi, gx, gy)
        ft = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        tic = time.perf_counter()
        with torch.no_grad():
            aseq = model(ft)
        d["inference_times"].append((time.perf_counter() - tic) * 1000)

        for i in range(STEPS_PER_INFER):
            if step >= max_steps or terminated or truncated:
                break
            act = aseq[0, i, :].cpu().numpy()
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

            if step in vis_indices and not terminated and not truncated:
                d["vis_frames"].append(_snapshot(env, step))

    if terminated and step not in vis_indices:
        d["vis_frames"].append(_snapshot(env, step))

    d["crashed"] = bool(info.get("crashed", terminated)) if terminated else False
    d["total_steps"] = step
    d["scores"] = compute_scores(d)
    return d


# =====================================================================
#                          可 视 化
# =====================================================================
def _vehicle_corners(x, y, heading, length, width):
    ch, sh = np.cos(heading), np.sin(heading)
    hl, hw = length / 2, width / 2
    return [
        (x + hl * ch - hw * sh, y + hl * sh + hw * ch),
        (x + hl * ch + hw * sh, y + hl * sh - hw * ch),
        (x - hl * ch + hw * sh, y - hl * sh - hw * ch),
        (x - hl * ch - hw * sh, y - hl * sh + hw * ch),
    ]


def draw_episode(ep, save_path, cn_name):
    vis = ep["vis_frames"]
    n = len(vis)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n))
    if n == 1:
        axes = [axes]

    for idx, frame in enumerate(vis):
        ax = axes[idx]
        ego_x = frame["ego_x"]

        for li in range(LANES_COUNT + 1):
            yy = li * LANE_WIDTH - LANE_WIDTH / 2
            ax.axhline(y=yy, color="gray", lw=0.8, ls="--")

        for veh in frame["vehicles"]:
            corners = _vehicle_corners(
                veh["x"], veh["y"], veh["heading"],
                veh["length"], veh["width"])
            poly = MplPolygon(corners, closed=True, fc=veh["color"],
                              ec="black", lw=0.5, alpha=0.85)
            ax.add_patch(poly)

        ax.set_xlim(ego_x - 60, ego_x + 260)
        ax.set_ylim(-LANE_WIDTH, LANES_COUNT * LANE_WIDTH)
        ax.set_ylabel("y方向位置/m", fontsize=8)
        ax.set_title(f"当前帧: {frame['step']}", fontsize=9, loc="left")
        ax.set_aspect("equal")

    axes[-1].set_xlabel("x方向位置/m", fontsize=8)

    status = "碰撞" if ep["crashed"] else "安全"
    sc = ep.get("scores", {})
    tag = ""
    if ep.get("representative"):
        tag = f" ★{ep['representative']}"
    fig.suptitle(
        f"{cn_name}场景 | EP {ep['episode_idx']+1} | {status} | "
        f"Total={sc.get('total',0):.1f} S={sc.get('safety',0):.1f} "
        f"E={sc.get('efficiency',0):.1f} C={sc.get('comfort',0):.1f}{tag}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
#                       报 告 生 成
# =====================================================================
def generate_csv(results, path):
    fields = [
        "scenario_type", "scenario_cn", "episode_idx", "seed",
        "crashed", "goal_achieved",
        "total_steps", "duration_s", "total_reward",
        "avg_speed", "distance", "n_lane_changes",
        "mean_ttc", "min_ttc_val", "avg_jerk", "avg_steer_rate",
        "safety", "efficiency", "comfort", "total",
        "avg_inference_ms", "representative",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            sc = r["scores"]
            w.writerow({
                "scenario_type": r["scenario_type"],
                "scenario_cn": SCENARIO_CN[r["scenario_type"]],
                "episode_idx": r["episode_idx"],
                "seed": r["seed"],
                "crashed": int(r["crashed"]),
                "goal_achieved": int(sc["goal_achieved"]),
                "total_steps": r["total_steps"],
                "duration_s": round(r["total_steps"] / POLICY_FREQ, 1),
                "total_reward": round(r["total_reward"], 2),
                "avg_speed": sc["avg_speed"],
                "distance": sc["distance"],
                "n_lane_changes": sc["n_lane_changes"],
                "mean_ttc": sc["mean_ttc"],
                "min_ttc_val": sc["min_ttc_val"],
                "avg_jerk": sc["avg_jerk"],
                "avg_steer_rate": sc["avg_steer_rate"],
                "safety": sc["safety"],
                "efficiency": sc["efficiency"],
                "comfort": sc["comfort"],
                "total": sc["total"],
                "avg_inference_ms": round(
                    float(np.mean(r["inference_times"])), 3
                ) if r["inference_times"] else 0,
                "representative": r.get("representative", ""),
            })


def generate_txt_report(results, path):
    n = len(results)
    crashes = sum(1 for r in results if r["crashed"])
    all_sc = [r["scores"] for r in results]
    all_inf = [t for r in results for t in r["inference_times"]]

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Highway-env Transformer BC 综合测试报告 v2\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"设备: {DEVICE}\n")
        f.write(f"每集时长: {EPISODE_DURATION}s | 每场景集数: "
                f"{n // len(SCENARIO_TYPES)}\n")
        f.write("=" * 80 + "\n\n")

        ga = sum(1 for s in all_sc if s["goal_achieved"])
        f.write(f"总测试集数: {n}\n")
        f.write(f"安全完成 (无碰撞): {n - crashes}/{n} "
                f"({100 * (n - crashes) / n:.1f}%)\n")
        f.write(f"碰撞次数: {crashes}/{n} ({100 * crashes / n:.1f}%)\n")
        f.write(f"目标达成率: {ga}/{n} ({100 * ga / n:.1f}%)\n\n")

        f.write(f"平均总分:   {np.mean([s['total'] for s in all_sc]):.2f} / 100\n")
        f.write(f"平均安全分: {np.mean([s['safety'] for s in all_sc]):.2f} / 50\n")
        f.write(f"平均效率分: {np.mean([s['efficiency'] for s in all_sc]):.2f} / 30\n")
        f.write(f"平均舒适分: {np.mean([s['comfort'] for s in all_sc]):.2f} / 20\n")
        f.write(f"平均速度:   {np.mean([s['avg_speed'] for s in all_sc]):.1f} m/s\n")
        if all_inf:
            f.write(f"\n推理耗时 — 平均: {np.mean(all_inf):.3f} ms | "
                    f"P50: {np.percentile(all_inf, 50):.3f} ms | "
                    f"P95: {np.percentile(all_inf, 95):.3f} ms | "
                    f"P99: {np.percentile(all_inf, 99):.3f} ms\n")

        # ---- 分场景汇总表 ----
        f.write("\n" + "-" * 80 + "\n")
        f.write("分场景汇总:\n")
        hdr = (f"{'场景':<14} {'集数':>4} {'碰撞':>4} {'安全率':>7} "
               f"{'达成率':>7} {'Safety':>7} {'Effic':>7} {'Comft':>7} "
               f"{'Total':>7} {'速度':>6} {'TTC':>6}\n")
        f.write(hdr)
        f.write("-" * 80 + "\n")
        for st in SCENARIO_TYPES:
            sr = [r for r in results if r["scenario_type"] == st]
            sn = len(sr)
            sc_list = [r["scores"] for r in sr]
            cr = sum(1 for r in sr if r["crashed"])
            ga2 = sum(1 for s in sc_list if s["goal_achieved"])
            f.write(
                f"{SCENARIO_CN[st]:<12} {sn:>4} {cr:>4} "
                f"{100 * (sn - cr) / sn:>6.1f}% "
                f"{100 * ga2 / sn:>6.1f}% "
                f"{np.mean([s['safety'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['efficiency'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['comfort'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['total'] for s in sc_list]):>7.2f} "
                f"{np.mean([s['avg_speed'] for s in sc_list]):>5.1f} "
                f"{np.mean([s['mean_ttc'] for s in sc_list]):>6.1f}\n"
            )
        f.write("-" * 80 + "\n\n")

        # ---- 代表性集 ----
        f.write("代表性集标记:\n")
        for r in results:
            if r.get("representative"):
                sc = r["scores"]
                f.write(
                    f"  ★ [{r['representative']}] {SCENARIO_CN[r['scenario_type']]} "
                    f"EP {r['episode_idx'] + 1} | "
                    f"{'碰撞' if r['crashed'] else '安全'} | "
                    f"S={sc['safety']} E={sc['efficiency']} "
                    f"C={sc['comfort']} T={sc['total']}\n"
                )
        f.write("\n")

        # ---- 逐集详情 ----
        f.write("=" * 80 + "\n")
        f.write("逐集详情:\n")
        f.write("=" * 80 + "\n")
        for r in results:
            sc = r["scores"]
            tag = f" ★{r['representative']}" if r.get("representative") else ""
            avg_inf = (float(np.mean(r["inference_times"]))
                       if r["inference_times"] else 0)
            f.write(
                f"  [{SCENARIO_CN[r['scenario_type']]}] "
                f"EP {r['episode_idx'] + 1:4d}{tag} | "
                f"{'碰撞' if r['crashed'] else '安全'} | "
                f"steps={r['total_steps']:4d} "
                f"dur={r['total_steps'] / POLICY_FREQ:.1f}s | "
                f"reward={r['total_reward']:.2f} | "
                f"speed={sc['avg_speed']:.1f} | "
                f"S={sc['safety']} E={sc['efficiency']} "
                f"C={sc['comfort']} T={sc['total']} | "
                f"TTC={sc['mean_ttc']:.1f} | "
                f"lc={sc['n_lane_changes']} | "
                f"infer={avg_inf:.2f}ms\n"
            )


def generate_summary_plots(results, save_dir, ts):
    # 1) 各指标分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("safety", "Safety (50)", 50),
               ("efficiency", "Efficiency (30)", 30),
               ("comfort", "Comfort (20)", 20),
               ("total", "Total (100)", 100)]
    for ax, (key, title, _) in zip(axes.flat, metrics):
        for st in SCENARIO_TYPES:
            vals = [r["scores"][key] for r in results if r["scenario_type"] == st]
            if vals:
                ax.hist(vals, bins=20, alpha=0.55, label=SCENARIO_CN[st])
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"score_distribution_{ts}.png"), dpi=150)
    plt.close(fig)

    # 2) 推理耗时
    all_inf = [t for r in results for t in r["inference_times"]]
    if all_inf:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_inf, bins=50, color="steelblue", ec="black", alpha=0.7)
        ax.axvline(np.mean(all_inf), color="red", ls="--",
                   label=f"Mean={np.mean(all_inf):.3f}ms")
        ax.set_xlabel("Inference Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Transformer Inference Time Distribution")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"inference_time_{ts}.png"), dpi=150)
        plt.close(fig)

    # 3) 场景对比柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SCENARIO_TYPES))
    w = 0.22
    for i, (key, label) in enumerate([("safety", "Safety"),
                                       ("efficiency", "Efficiency"),
                                       ("comfort", "Comfort")]):
        means = []
        for st in SCENARIO_TYPES:
            vals = [r["scores"][key] for r in results if r["scenario_type"] == st]
            means.append(float(np.mean(vals)) if vals else 0)
        ax.bar(x + i * w, means, w, label=label)
    ax.set_xticks(x + w)
    ax.set_xticklabels([SCENARIO_CN[s] for s in SCENARIO_TYPES])
    ax.set_ylabel("Average Score")
    ax.set_title("Score Comparison by Scenario Type")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"scenario_comparison_{ts}.png"), dpi=150)
    plt.close(fig)


# =====================================================================
#                            主 函 数
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Highway-env Transformer 综合测试 v2")
    parser.add_argument("--quick", action="store_true",
                        help="快速验证模式 (每场景 4 集)")
    parser.add_argument("-n", type=int, default=None,
                        help="每场景集数 (默认 250)")
    args = parser.parse_args()

    eps_per = args.n if args.n else (4 if args.quick else EPISODES_PER_SCENARIO)
    total_eps = eps_per * len(SCENARIO_TYPES)

    ts = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join(BASE_DIR, "highway_env_results_v2", ts)
    os.makedirs(save_dir, exist_ok=True)
    for st in SCENARIO_TYPES:
        os.makedirs(os.path.join(save_dir, "episodes", st), exist_ok=True)

    print(f"设备: {DEVICE}")
    print(f"模型: {MODEL_PATH}")
    print(f"场景: {', '.join(SCENARIO_CN[s] for s in SCENARIO_TYPES)}")
    print(f"每场景集数: {eps_per} | 总计: {total_eps}")
    print(f"每集时长: {EPISODE_DURATION}s ({EPISODE_DURATION * POLICY_FREQ} steps)")
    print(f"输出目录: {save_dir}")
    print("=" * 70)

    # ---- 加载模型 ----
    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH,
        car_num=CAR_NUM, nhead=8,
        num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS,
        dim_feedforward=FFN_DIM, dropout=0.0,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功\n")

    mi = build_map_info()
    env = _make_env()
    rng = np.random.default_rng(42)

    # ---- 运行所有场景 ----
    all_results = []
    global_ep = 0
    t0 = time.time()

    for st in SCENARIO_TYPES:
        print(f"\n{'=' * 70}")
        print(f"场景: {SCENARIO_CN[st]} ({st}) — {eps_per} 集")
        print(f"{'=' * 70}")
        for i in range(eps_per):
            try:
                ep = run_episode(env, model, mi, st, i, rng)
            except Exception as exc:
                print(f"  [WARNING] EP {i+1} 异常: {exc}")
                continue
            all_results.append(ep)
            global_ep += 1
            sc = ep["scores"]
            status = "碰撞" if ep["crashed"] else "安全"
            print(
                f"  [{SCENARIO_CN[st]}] EP {i+1:4d}/{eps_per} "
                f"({global_ep}/{total_eps}) | {status} | "
                f"T={sc['total']:5.1f} S={sc['safety']:5.1f} "
                f"E={sc['efficiency']:5.1f} C={sc['comfort']:5.1f} | "
                f"spd={sc['avg_speed']:.1f} | "
                f"infer={np.mean(ep['inference_times']):.2f}ms"
            )

    env.close()
    elapsed = time.time() - t0
    print(f"\n仿真完成，用时 {elapsed / 60:.1f} 分钟")

    # ---- 标记代表性集 ----
    for st in SCENARIO_TYPES:
        sr = [r for r in all_results if r["scenario_type"] == st]
        if not sr:
            continue
        nc = [r for r in sr if not r["crashed"]]
        cr = [r for r in sr if r["crashed"]]
        if nc:
            best = max(nc, key=lambda r: r["scores"]["total"])
            best["representative"] = "最佳"
            worst = min(nc, key=lambda r: r["scores"]["total"])
            if worst is not best:
                worst["representative"] = "最难"
            med = sorted(nc, key=lambda r: r["scores"]["total"])
            m = med[len(med) // 2]
            if not m.get("representative"):
                m["representative"] = "中位"
        for r in cr:
            r["representative"] = "碰撞"

    # ---- 绘制每集可视化 ----
    print(f"\n正在绘制 {len(all_results)} 张可视化图...")
    for idx, r in enumerate(all_results):
        tag = f"_{r['representative']}" if r.get("representative") else ""
        fname = (f"ep_{r['episode_idx']+1:04d}_"
                 f"score{r['scores']['total']:.0f}{tag}.png")
        fpath = os.path.join(save_dir, "episodes", r["scenario_type"], fname)
        draw_episode(r, fpath, SCENARIO_CN[r["scenario_type"]])
        if (idx + 1) % 100 == 0:
            print(f"  已绘制 {idx + 1}/{len(all_results)}")
    print("可视化绘制完成")

    # ---- 生成报告 ----
    csv_path = os.path.join(save_dir, f"episode_details_{ts}.csv")
    txt_path = os.path.join(save_dir, f"test_report_{ts}.txt")
    generate_csv(all_results, csv_path)
    generate_txt_report(all_results, txt_path)
    generate_summary_plots(all_results, save_dir, ts)

    # ---- 最终汇总输出到终端 ----
    print(f"\n{'=' * 70}")
    print("最终汇总")
    print(f"{'=' * 70}")
    n = len(all_results)
    crashes = sum(1 for r in all_results if r["crashed"])
    ga = sum(1 for r in all_results if r["scores"]["goal_achieved"])
    print(f"  总集数:     {n}")
    print(f"  安全完成:   {n - crashes}/{n} ({100 * (n - crashes) / n:.1f}%)")
    print(f"  碰撞:       {crashes}/{n} ({100 * crashes / n:.1f}%)")
    print(f"  目标达成:   {ga}/{n} ({100 * ga / n:.1f}%)")
    print(f"  平均总分:   {np.mean([r['scores']['total'] for r in all_results]):.2f}")
    print(f"  平均安全分: "
          f"{np.mean([r['scores']['safety'] for r in all_results]):.2f}/50")
    print(f"  平均效率分: "
          f"{np.mean([r['scores']['efficiency'] for r in all_results]):.2f}/30")
    print(f"  平均舒适分: "
          f"{np.mean([r['scores']['comfort'] for r in all_results]):.2f}/20")
    all_inf = [t for r in all_results for t in r["inference_times"]]
    if all_inf:
        print(f"  推理耗时:   {np.mean(all_inf):.3f} ms (P95={np.percentile(all_inf, 95):.3f}ms)")
    print(f"\n  结果目录: {save_dir}")
    print(f"  CSV: {csv_path}")
    print(f"  报告: {txt_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
