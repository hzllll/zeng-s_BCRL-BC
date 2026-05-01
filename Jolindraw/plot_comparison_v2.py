import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")

# ==========================================
# 0. 全局美学设置 (参考学术顶会风格)
# ==========================================
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 确保负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 定义指数移动平均(EMA)平滑函数 (类似TensorBoard)
def smooth_curve(scalars, weight=0.85):
    """
    weight: 平滑度，取值 0~1。越大越平滑。
    """
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ==========================================
# 1. 配置实验日志文件路径和标签
# ==========================================
log_files = {
    "Exp-1": glob.glob("../train_log/*TFExp-1*.txt")[0] if glob.glob("../train_log/*TFExp-1*.txt") else None,
    "Exp-2": glob.glob("../train_log/*TFexp-2_v2*.txt")[0] if glob.glob("../train_log/*TFexp-2_v2*.txt") else None,
    "Exp-3": glob.glob("../train_log/*TFexp-3*.txt")[0] if glob.glob("../train_log/*TFexp-3*.txt") else None,
    "Exp-4": glob.glob("../train_log/*TFexp-4*.txt")[0] if glob.glob("../train_log/*TFexp-4*.txt") else None,
    "Exp-5": glob.glob("../train_log/*TFexp-5*.txt")[0] if glob.glob("../train_log/*TFexp-5*.txt") else None,
    "Exp-6 (Final Model)": glob.glob("../train_log/*0328_1024BSIZE*.txt")[0] if glob.glob("../train_log/*0328_1024BSIZE*.txt") else None 
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
# 3. 开始绘图
# ==========================================
# 去除 seaborn 样式，使用干净的默认背景
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

base_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b'] # 换了一组更沉稳的学术配色
color_idx = 0

for exp_name, losses in data.items():
    # 截断数据到200轮
    t_loss_raw = losses["train"][:200]
    v_loss_raw = losses["val"][:200]
    epochs = range(1, len(t_loss_raw) + 1)
    
    # 计算平滑后的曲线
    t_loss_smooth = smooth_curve(t_loss_raw, weight=0.85)
    v_loss_smooth = smooth_curve(v_loss_raw, weight=0.85)
    
    if "Exp-6" in exp_name:
        color = '#d62728'  # 经典的学术红
        linewidth = 2.5
        alpha_main = 1.0
        alpha_shadow = 0.25 # 阴影透明度
        zorder = 10
    else:
        color = base_colors[color_idx % len(base_colors)]
        linewidth = 1.5
        alpha_main = 0.8
        alpha_shadow = 0.15 # 阴影透明度更低
        zorder = 1
        color_idx += 1

    # 【左图：Train Loss】
    # 1. 画原始数据作为浅色阴影背景
    ax1.plot(epochs, t_loss_raw, color=color, alpha=alpha_shadow, linewidth=1.0, zorder=zorder-1)
    # 2. 画平滑后的数据作为主曲线
    ax1.plot(epochs, t_loss_smooth, label=exp_name, color=color, linewidth=linewidth, alpha=alpha_main, zorder=zorder)

    # 【右图：Val Loss】
    # 1. 画原始数据作为浅色阴影背景
    ax2.plot(epochs, v_loss_raw, color=color, alpha=alpha_shadow, linewidth=1.0, zorder=zorder-1)
    # 2. 画平滑后的数据作为主曲线
    ax2.plot(epochs, v_loss_smooth, label=exp_name, color=color, linewidth=linewidth, alpha=alpha_main, zorder=zorder)

# ==========================================
# 4. 图表细节美化 (应用学术风格)
# ==========================================
axes = [ax1, ax2]
titles = ['Training Loss', 'Validation Loss']

for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    
    # 限制坐标轴范围
    ax.set_xlim(1, 200)
    ax.set_ylim(0.01, 0.15)
    
    # 刻度线设置：向外，并调整大小
    ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1.5)
    
    # 隐藏上方和右侧的边框 (Despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 加粗左侧和下方的边框线
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 图例设置：去除边框 (frameon=False)
    ax.legend(fontsize=12, loc='upper right', frameon=False)

plt.tight_layout()

# 5. 保存图片
output_path = f"Transformer_plots/{RUN_DATE_TIME}_All_Exps_Loss_Comparison_Academic.svg"
os.makedirs("Transformer_plots", exist_ok=True)
plt.savefig(output_path, format='svg', bbox_inches='tight')
print(f"\n学术风格对比图已成功保存至: {output_path}")