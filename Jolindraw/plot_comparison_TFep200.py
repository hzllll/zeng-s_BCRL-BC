import os
import glob
import re
import matplotlib.pyplot as plt
from datetime import datetime
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")

# 1. 配置实验日志文件路径和标签
log_files = {
    "Exp-1": glob.glob("../train_log/*TFExp-1*.txt")[0] if glob.glob("../train_log/*TFExp-1*.txt") else None,
    "Exp-2": glob.glob("../train_log/*TFexp-2*.txt")[0] if glob.glob("../train_log/*TFexp-2*.txt") else None,
    "Exp-3": glob.glob("../train_log/*TFexp-3*.txt")[0] if glob.glob("../train_log/*TFexp-3*.txt") else None,
    "Exp-4": glob.glob("../train_log/*TFexp-4*.txt")[0] if glob.glob("../train_log/*TFexp-4*.txt") else None,
    "Exp-5": glob.glob("../train_log/*TFexp-5*.txt")[0] if glob.glob("../train_log/*TFexp-5*.txt") else None,
    "Exp-6 (Final Model)": glob.glob("../train_log/*0328_1024BSIZE*.txt")[0] if glob.glob("../train_log/*0328_1024BSIZE*.txt") else None 
}

# 解析日志文件的函数
def parse_log_file(filepath):
    train_losses = []
    val_losses = []
    if not filepath or not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return train_losses, val_losses
        
    pattern = re.compile(r"训练损失:\s*([\d.]+)\s*\|\s*验证损失:\s*([\d.]+)")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses.append(float(match.group(1)))
                val_losses.append(float(match.group(2)))
    return train_losses, val_losses

# 2. 读取所有数据
data = {}
for exp_name, filepath in log_files.items():
    if filepath:
        t_loss, v_loss = parse_log_file(filepath)
        if t_loss:
            data[exp_name] = {"train": t_loss, "val": v_loss}
            print(f"成功加载 {exp_name}: 找到 {len(t_loss)} 轮数据")
    else:
        print(f"未找到 {exp_name} 的日志文件，请检查路径！")

# 3. 开始绘图
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

base_colors = ['#A9A9A9', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C']
color_idx = 0

for exp_name, losses in data.items():
    # 【核心修改】：截断数据到200轮
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
        alpha = 0.6
        zorder = 1
        color_idx += 1

    ax1.plot(epochs, t_loss, label=exp_name, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax2.plot(epochs, v_loss, label=exp_name, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

# 4. 图表细节美化
ax1.set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.tick_params(axis='both', labelsize=12)
ax1.legend(fontsize=12, loc='upper right')
ax1.set_xlim(1, 200)       # 【新增】：固定 X 轴范围
ax1.set_ylim(0.01, 0.15) 

ax2.set_title('Validation Loss Comparison', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(fontsize=12, loc='upper right')
ax2.set_xlim(1, 200)       # 【新增】：固定 X 轴范围
ax2.set_ylim(0.01, 0.15)

plt.tight_layout()

# 5. 保存图片
output_path = f"Transformer_plots/{RUN_DATE_TIME}_All_Exps_Loss_Comparison_200.svg"
os.makedirs("Transformer_plots", exist_ok=True)
plt.savefig(output_path, format='svg', bbox_inches='tight')
print(f"\n对比图已成功保存至: {output_path}")