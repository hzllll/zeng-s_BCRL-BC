"""
Highway-env 测试脚本：加载 Transformer BC 模型在 highway-env 高速公路环境中评测
用法：python test_highway_env_transformer.py
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ======================== 全局配置 ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth"
)

# Transformer 超参数（必须与训练脚本完全一致）
D_MODEL = 256
FFN_DIM = 4 * D_MODEL
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8

# highway-env 环境参数
LANES_COUNT = 3
LANE_WIDTH = 4.0
VEHICLES_COUNT = 30
DURATION = 60          # 单集时长（秒）
SIM_FREQ = 10          # 仿真频率（Hz），dt = 0.1s，匹配 Onsite
POLICY_FREQ = 10       # 策略频率（Hz），每次 env.step() = 0.1s

# 测试参数
NUM_EPISODES = 50
STEPS_PER_INFERENCE = 5   # 每次推理后执行 5 步（与 Onsite 一致）
GOAL_LOOKAHEAD = 500.0    # 虚拟目标点前视距离（m）
RENDER = False             # 是否渲染画面（无头服务器设为 False）

SAVE_DIR = os.path.join(BASE_DIR, "highway_env_results")
os.makedirs(SAVE_DIR, exist_ok=True)


# ======================== 模型定义（与 get_clone_learning_Transformer6_7.py 完全一致）========================
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


# ======================== 车道映射（highway-env → Onsite 约定）========================
def build_map_info(lanes_count=3, lane_width=4.0):
    """
    将 highway-env 的车道结构映射为 Onsite 风格的 map_info。
    
    highway-env 3车道:  lane0 center=0, lane1 center=4, lane2 center=8
    Onsite 约定:        map_info[0] 是最高 y 的车道（left_bound 最大）
    
    因此需要反转顺序：map_info[0] ← highway lane 2, map_info[2] ← highway lane 0
    """
    map_info = {}
    for i in range(lanes_count):
        hw_lane_idx = lanes_count - 1 - i
        center_y = hw_lane_idx * lane_width
        map_info[i] = {
            "left_bound": center_y + lane_width / 2,
            "center": center_y,
            "right_bound": center_y - lane_width / 2,
        }
    return map_info


def get_ego_lane_bounds(ego_y, map_info):
    """根据 ego_y 返回所在车道的 [left_bound, right_bound] 相对值"""
    if ego_y >= map_info[0]['right_bound']:
        return map_info[0]['left_bound'] - ego_y, map_info[0]['right_bound'] - ego_y
    elif map_info[1]['right_bound'] < ego_y < map_info[1]['left_bound']:
        return map_info[1]['left_bound'] - ego_y, map_info[1]['right_bound'] - ego_y
    else:
        return map_info[2]['left_bound'] - ego_y, map_info[2]['right_bound'] - ego_y


def get_veh_lane_bounds(veh_y, ego_y, map_info):
    """根据障碍车 y 返回其所在车道 [left_bound, right_bound] 相对于 ego 的值"""
    if veh_y >= map_info[0]['right_bound']:
        return map_info[0]['left_bound'] - ego_y, map_info[0]['right_bound'] - ego_y
    elif map_info[1]['right_bound'] < veh_y < map_info[1]['left_bound']:
        return map_info[1]['left_bound'] - ego_y, map_info[1]['right_bound'] - ego_y
    elif veh_y <= map_info[2]['left_bound']:
        return map_info[2]['left_bound'] - ego_y, map_info[2]['right_bound'] - ego_y
    else:
        return get_ego_lane_bounds(ego_y, map_info)


# ======================== 特征提取（与 Onsite test_transformer_simulation.py 完全对齐）========================
def extract_features(env, map_info, goal_lookahead=500.0):
    """
    从 highway-env 环境中提取 54 维特征向量，格式与 Onsite 训练数据完全一致：
      [0:6]  = 主车 6 维: v, yaw, goal_dx, goal_dy, upper_bound-ego_y, lower_bound-ego_y
      [6:54] = 8辆障碍车 × 6维: lon_dist, lat_dist, rel_v, yaw, lane_left-ego_y, lane_right-ego_y
    """
    ego = env.unwrapped.vehicle
    ego_x, ego_y = ego.position
    ego_v = ego.speed
    ego_yaw = ego.heading
    ego_len = ego.LENGTH

    ego_yaw_norm = (ego_yaw + np.pi) % (2 * np.pi) - np.pi

    # 虚拟目标点：前方 goal_lookahead 米处，当前车道中心
    hw_lane_idx = int(round(ego_y / LANE_WIDTH))
    hw_lane_idx = max(0, min(hw_lane_idx, LANES_COUNT - 1))
    goal_x = ego_x + goal_lookahead
    goal_y = hw_lane_idx * LANE_WIDTH

    states = []

    # ---- 主车 6 维 ----
    states.append(ego_v)
    states.append(ego_yaw_norm)
    states.append(goal_x - ego_x)
    states.append(goal_y - ego_y)
    states.append(map_info[0]['left_bound'] - ego_y)
    states.append(map_info[2]['right_bound'] - ego_y)

    # ---- 障碍车处理 ----
    other_vehicles = env.unwrapped.road.vehicles[1:]

    candidates = []
    for v in other_vehicles:
        vx, vy = v.position
        if abs(vy - ego_y) > 6:
            continue

        dx = vx - ego_x
        v_len = v.LENGTH
        half_len_sum = 0.5 * (v_len + ego_len)

        if dx - half_len_sum > 0:
            lon_dist = dx - half_len_sum
        elif dx + half_len_sum < 0:
            lon_dist = dx + half_len_sum
        else:
            lon_dist = 0.0

        dist = np.sqrt(lon_dist ** 2 + (vy - ego_y) ** 2)
        if dist < 200:
            candidates.append((dist, lon_dist, v))

    candidates.sort(key=lambda t: t[0])
    selected = candidates[:CAR_NUM]

    for _, lon_dist, v in selected:
        vx, vy = v.position
        v_speed = v.speed
        v_yaw = (v.heading + np.pi) % (2 * np.pi) - np.pi

        states.append(lon_dist)
        states.append(vy - ego_y)
        states.append(v_speed - ego_v)
        states.append(v_yaw)

        lb, rb = get_veh_lane_bounds(vy, ego_y, map_info)
        states.append(lb)
        states.append(rb)

    # 补全不足的车辆（与 Onsite 一致）
    ego_lb, ego_rb = get_ego_lane_bounds(ego_y, map_info)
    for _ in range(CAR_NUM - len(selected)):
        states.extend([200.0, 0.0, 0.0, 0.0])
        states.append(ego_lb)
        states.append(ego_rb)

    return states


# ======================== highway-env 环境工厂 ========================
def make_env(render=False):
    render_mode = "rgb_array" if render else None
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 9,
            "features": ["x", "y", "vx", "vy", "heading"],
            "absolute": True,
            "normalize": False,
        },
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": (-6, 6),
            "steering_range": (-0.15, 0.15),
        },
        "lanes_count": LANES_COUNT,
        "vehicles_count": VEHICLES_COUNT,
        "duration": DURATION,
        "simulation_frequency": SIM_FREQ,
        "policy_frequency": POLICY_FREQ,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "collision_reward": -1.0,
        "reward_speed_range": [20, 30],
        "offscreen_rendering": True,
    }
    env = gym.make("highway-v0", render_mode=render_mode, config=config)
    return env


# ======================== 单集测试 ========================
def run_episode(env, model, map_info, episode_idx=0):
    obs, info = env.reset()

    total_reward = 0.0
    total_steps = 0
    crashed = False
    inference_times = []
    speeds = []

    max_steps = DURATION * POLICY_FREQ  # 最大步数

    while total_steps < max_steps:
        # 1. 特征提取
        features = extract_features(env, map_info, GOAL_LOOKAHEAD)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 2. 模型推理（计时）
        tic = time.perf_counter()
        with torch.no_grad():
            action_seq = model(feat_tensor)  # (1, 5, 2)
        toc = time.perf_counter()
        inference_times.append((toc - tic) * 1000)

        # 3. 依次执行 5 步动作
        for i in range(STEPS_PER_INFERENCE):
            action = action_seq[0, i, :].cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            speeds.append(env.unwrapped.vehicle.speed)

            if terminated or truncated:
                crashed = info.get("crashed", terminated)
                break

        if terminated or truncated:
            break

    avg_speed = np.mean(speeds) if speeds else 0.0
    avg_infer = np.mean(inference_times) if inference_times else 0.0

    return {
        "episode": episode_idx,
        "steps": total_steps,
        "duration_s": total_steps / POLICY_FREQ,
        "reward": total_reward,
        "crashed": crashed,
        "avg_speed": avg_speed,
        "avg_inference_ms": avg_infer,
        "inference_times": inference_times,
    }


# ======================== 主函数 ========================
def main():
    print(f"设备: {DEVICE}")
    print(f"模型: {MODEL_PATH}")
    print(f"测试集数: {NUM_EPISODES}")
    print(f"环境: highway-v0 | {LANES_COUNT} 车道 | {VEHICLES_COUNT} 辆周围车 | {DURATION}s/集")
    print("=" * 70)

    # 1. 加载模型
    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
        nhead=8, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=FFN_DIM, dropout=0.0,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功\n")

    # 2. 构建车道映射
    map_info = build_map_info(LANES_COUNT, LANE_WIDTH)
    print("车道映射 (Onsite 格式):")
    for k, v in map_info.items():
        print(f"  map_info[{k}]: left={v['left_bound']:.1f}, center={v['center']:.1f}, right={v['right_bound']:.1f}")
    print()

    # 3. 创建环境
    env = make_env(render=RENDER)

    # 4. 多集测试
    results = []
    for ep in range(NUM_EPISODES):
        res = run_episode(env, model, map_info, ep)
        results.append(res)

        status = "碰撞" if res["crashed"] else "安全"
        print(
            f"Episode {ep+1:3d}/{NUM_EPISODES} | "
            f"{status} | "
            f"步数: {res['steps']:4d} | "
            f"时长: {res['duration_s']:5.1f}s | "
            f"累计奖励: {res['reward']:7.2f} | "
            f"平均速度: {res['avg_speed']:5.1f} m/s | "
            f"推理: {res['avg_inference_ms']:.2f} ms"
        )

    env.close()

    # 5. 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    n = len(results)
    crash_count = sum(1 for r in results if r["crashed"])
    safe_count = n - crash_count
    avg_reward = np.mean([r["reward"] for r in results])
    avg_speed = np.mean([r["avg_speed"] for r in results])
    avg_duration = np.mean([r["duration_s"] for r in results])
    all_infer = [t for r in results for t in r["inference_times"]]
    avg_infer = np.mean(all_infer) if all_infer else 0

    print(f"  总集数:        {n}")
    print(f"  安全完成:      {safe_count} ({100*safe_count/n:.1f}%)")
    print(f"  碰撞次数:      {crash_count} ({100*crash_count/n:.1f}%)")
    print(f"  平均奖励:      {avg_reward:.2f}")
    print(f"  平均速度:      {avg_speed:.1f} m/s")
    print(f"  平均存活时长:  {avg_duration:.1f} s")
    print(f"  平均推理耗时:  {avg_infer:.3f} ms")
    if all_infer:
        print(f"  推理P50:       {np.percentile(all_infer, 50):.3f} ms")
        print(f"  推理P95:       {np.percentile(all_infer, 95):.3f} ms")
        print(f"  推理P99:       {np.percentile(all_infer, 99):.3f} ms")

    # 6. 保存结果
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    result_file = os.path.join(SAVE_DIR, f"highway_test_{timestamp}.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Highway-env Transformer BC 测试报告\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"环境: {LANES_COUNT} 车道 | {VEHICLES_COUNT} 周围车 | {DURATION}s/集\n")
        f.write(f"测试集数: {n}\n\n")
        f.write(f"安全完成: {safe_count}/{n} ({100*safe_count/n:.1f}%)\n")
        f.write(f"碰撞次数: {crash_count}/{n} ({100*crash_count/n:.1f}%)\n")
        f.write(f"平均奖励: {avg_reward:.2f}\n")
        f.write(f"平均速度: {avg_speed:.1f} m/s\n")
        f.write(f"平均存活时长: {avg_duration:.1f} s\n")
        f.write(f"平均推理耗时: {avg_infer:.3f} ms\n\n")
        f.write("逐集详情:\n")
        for r in results:
            f.write(
                f"  ep={r['episode']:3d} | "
                f"{'碰撞' if r['crashed'] else '安全'} | "
                f"steps={r['steps']:4d} | "
                f"dur={r['duration_s']:.1f}s | "
                f"reward={r['reward']:.2f} | "
                f"speed={r['avg_speed']:.1f}\n"
            )
    print(f"\n结果已保存: {result_file}")

    # 7. 绘制推理时间分布
    if all_infer:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(all_infer, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(avg_infer, color='red', linestyle='--', label=f'Mean={avg_infer:.3f}ms')
        axes[0].set_xlabel('Inference Time (ms)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Transformer Inference Time Distribution')
        axes[0].legend()

        episode_rewards = [r["reward"] for r in results]
        axes[1].bar(range(n), episode_rewards,
                    color=['red' if r["crashed"] else 'steelblue' for r in results],
                    alpha=0.7)
        axes[1].axhline(avg_reward, color='black', linestyle='--', label=f'Mean={avg_reward:.1f}')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Cumulative Reward')
        axes[1].set_title('Episode Rewards (red = crash)')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(SAVE_DIR, f"highway_test_{timestamp}.svg")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
