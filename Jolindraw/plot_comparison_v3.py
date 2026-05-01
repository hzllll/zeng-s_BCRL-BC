import os
import glob
import re
import matplotlib.pyplot as plt
from datetime import datetime
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")

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
    # 注意 Exp-1 的文件名是大写的 TFExp-1
    "Exp-1:Underfitting": glob.glob("../train_log/*TFExp-1*.txt")[0] if glob.glob("../train_log/*TFExp-1*.txt") else None,
    "Exp-2:Overfitting": glob.glob("../train_log/*TFexp-2*.txt")[0] if glob.glob("../train_log/*TFexp-2*.txt") else None,
    "Exp-3:SmallBatch": glob.glob("../train_log/*TFexp-3*.txt")[0] if glob.glob("../train_log/*TFexp-3*.txt") else None,
    "Exp-4:Dropout=0.0": glob.glob("../train_log/*TFexp-4*.txt")[0] if glob.glob("../train_log/*TFexp-4*.txt") else None,
    "Exp-5:Dropout=0.30": glob.glob("../train_log/*TFexp-5*.txt")[0] if glob.glob("../train_log/*TFexp-5*.txt") else None,
    # Exp-6 (0328_1024BSIZE)
    "Exp-6:Final Model": glob.glob("../train_log/*0328_1024BSIZE*.txt")[0] if glob.glob("../train_log/*0328_1024BSIZE*.txt") else None 
}

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

base_colors = ['#A9A9A9', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C']
color_idx = 0

for exp_name, losses in data.items():
    # 依然保留截断到 200 轮的逻辑
    t_loss = losses["train"][:200]
    v_loss = losses["val"][:200]
    epochs = range(1, len(t_loss) + 1)
    
    if "Exp-6" in exp_name:
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
    ax.set_ylim(0.01, 0.15)
    
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
output_path = f"Transformer_plots/{RUN_DATE_TIME}_All_Exps_Loss_Comparison_Clean.svg"
os.makedirs("Transformer_plots", exist_ok=True)
plt.savefig(output_path, format='svg', bbox_inches='tight')
print(f"\n干净版对比图已成功保存至: {output_path}")