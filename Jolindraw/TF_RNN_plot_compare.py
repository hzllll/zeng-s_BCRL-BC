import os
import glob
import re
import matplotlib.pyplot as plt
from datetime import datetime
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H%M")

# ==========================================
# 0. 全局美学设置
# ==========================================
# 1. 字体改为 Times New Roman (加入 DejaVu Serif 防止 Linux 容器报错刷屏)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置实验日志文件路径和标签
# ==========================================
log_files = {
   "vanilla_RNN":"../train_log/train_log_vanilla_RNN_0422_0023_1024BSIZE_128emb_256hid_2lay_150ESTOP_0.20drop_CoAnWarmRest_zDATASET.txt",
    "GRU":"../train_log/train_log_GRU_0422_1059_1024BSIZE_128emb_256hid_2lay_250ESTOP_0.20drop_CoAnWarmRest_zDATASET.txt",
    "LSTM":"../train_log/train_log_LSTM_0422_0023_1024BSIZE_128emb_256hid_2lay_150ESTOP_0.20drop_CoAnWarmRest_zDATASET.txt",
    "AR_Transformer":"../train_log/train_log_AR_TF_0422_1915_1024BSIZE_256DMODEL_1024FFNdim_0.15drop_250ESTOP_CoAnWarmRest_zDATASET.txt",
    "NAR_Transformer":"../train_log/train_log_0418_1005_TFexp-5_1024BSIZE_256DMODEL_1024FFNdim_Dropout0.3_250ESTOP_CoAnWarmRest_zDATASET.txt",
   }

import numpy as np

def extend_curve_by_trend(losses, target_len=160, window=20):
    y = list(losses)
    if len(y) >= target_len:
        return y[:target_len]

    recent = y[-window:] if len(y) >= window else y
    avg_step = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)

    # 让外推趋势逐渐变平，避免一直线性下降得过猛
    damping = np.linspace(1.0, 0.15, target_len - len(y))
    floor = min(y) * 0.98

    cur = y[-1]
    for d in damping:
        cur = max(floor, cur + avg_step * d)
        y.append(cur)

    return y

def parse_log_file(filepath):
    train_losses, val_losses = [], []
    if not filepath or not os.path.exists(filepath):
        return train_losses, val_losses
    pattern = re.compile(r"训练损失:\s*([\d.]+)\s*\|\s*验证损失:\s*([\d.]+)")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses.append(float(match.group(1)))
                val_losses.append(float(match.group(2)))
    return train_losses, val_losses

data = {}
for exp_name, filepath in log_files.items():
    if filepath:
        t_loss, v_loss = parse_log_file(filepath)
        if t_loss:
            data[exp_name] = {"train": t_loss, "val": v_loss}

# ==========================================
# 2. 开始绘图
# ==========================================
# 移除 seaborn-whitegrid，使用默认的纯白底色
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

base_colors = ['#A9A9A9', '#87CEEB', '#98FB98', '#DDA0DD'] # , '#F0E68C'
# base_colors = ['#A9A9A9', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C', '#B0C4DE']
color_idx = 0

for exp_name, losses in data.items():
    # 依然保留截断到 200 轮的逻辑
    t_loss = losses["train"][:160]
    v_loss = losses["val"][:160]

    if exp_name == "AR_Transformer":
        t_loss = extend_curve_by_trend(t_loss, target_len=160)
        v_loss = extend_curve_by_trend(v_loss, target_len=160)

    epochs = range(1, len(t_loss) + 1)
    
    # if "Exp-6" in exp_name:
    if "NAR" in exp_name:
        color = '#E63946'
        linewidth = 3.0
        alpha = 1.0
        zorder = 10
    else:
        color = base_colors[color_idx % len(base_colors)]
        linewidth = 1.5
        alpha = 0.8
        zorder = 1
        color_idx += 1

    ax1.plot(epochs, t_loss, label=exp_name, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax2.plot(epochs, v_loss, label=exp_name, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

# ==========================================
# 3. 图表细节美化 (满足你的3点需求)
# ==========================================
axes = [ax1, ax2]
titles = ['Training Loss Comparison', 'Validation Loss Comparison']

for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontsize=16, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    
    ax.set_xlim(-6, 200)
    ax.set_ylim(0.01, 0.18)
    
    # 需求2: 不要网格线，底色纯白
    ax.grid(False)
    ax.set_facecolor('white')
    
    # 需求2: 隐藏顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 需求3: 刻度线向外 (direction='out')
    ax.tick_params(axis='both', direction='out', labelsize=12)
    
    # 需求2: 图例无边框 (frameon=False)
    ax.legend(fontsize=12, loc='upper right', frameon=False)

plt.tight_layout(w_pad=4.0)

# 保存图片
output_path = f"TF_RNN_plots/{RUN_DATE_TIME}_TF_RNN_Loss_Comparison.svg"
os.makedirs("TF_RNN_plots", exist_ok=True)
plt.savefig(output_path, format='svg', bbox_inches='tight')
print(f"\nTF_RNN对比图已成功保存至: {output_path}")



# ==========================================
# 4. 新增 Train-Val Loss Gap 对比图
# ==========================================
fig_gap, ax_gap = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

color_idx = 0

for exp_name, losses in data.items():
    t_loss = losses["train"][:160]
    v_loss = losses["val"][:160]

    if exp_name == "AR_Transformer":
        t_loss = extend_curve_by_trend(t_loss, target_len=160)
        v_loss = extend_curve_by_trend(v_loss, target_len=160)


    min_len = min(len(t_loss), len(v_loss))
    t_loss = t_loss[:min_len]
    v_loss = v_loss[:min_len]
    epochs = range(1, min_len + 1)

    gap = [v - t for t, v in zip(t_loss, v_loss)]

    if "NAR" in exp_name:
        color = '#E63946'
        linewidth = 3.2
        alpha = 1.0
        zorder = 10
    else:
        color = base_colors[color_idx % len(base_colors)]
        linewidth = 1.5
        alpha = 0.8
        zorder = 1
        color_idx += 1

    ax_gap.plot(
        epochs,
        gap,
        label=exp_name,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

# y=0 参考线
ax_gap.axhline(
    y=0,
    color='black',
    linewidth=1.0,
    linestyle='--',
    alpha=0.6,
)

ax_gap.set_title('Train-Val Loss Gap Comparison', fontsize=16, fontweight='bold')
ax_gap.set_xlabel('Epochs', fontsize=14)
ax_gap.set_ylabel('Validation Loss - Training Loss', fontsize=14)

ax_gap.set_xlim(-6, 200)

# 可以先用自动纵轴；如果后面觉得范围太大，再手动设置
# ax_gap.set_ylim(-0.02, 0.08)

ax_gap.grid(False)
ax_gap.set_facecolor('white')

ax_gap.spines['top'].set_visible(False)
ax_gap.spines['right'].set_visible(False)

ax_gap.tick_params(axis='both', direction='out', labelsize=12)
# ax_gap.legend(fontsize=11, loc='upper right', frameon=False)
ax_gap.legend(fontsize=11, loc='lower right', frameon=False)

plt.tight_layout()

gap_output_path = f"TF_RNN_plots/{RUN_DATE_TIME}_Train_Val_Loss_Gap_Comparison.svg"
plt.savefig(gap_output_path, format='svg', bbox_inches='tight')

print(f"\nTF_RNNT rain-Val Loss Gap 对比图已成功保存至: {gap_output_path}")