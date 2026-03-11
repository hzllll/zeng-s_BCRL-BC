import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# # 读取npy
# data_mlp = np.load('losses_choose_mlp_plot.npy')
# data_rnn = np.load('losses_RNN_Multi1_choose_plot.npy')
# data_lstm = np.load('losses_LSTM_Multi1_choose_plot.npy')
# data_gru = np.load('losses_GRU_Multi1_choose_plot.npy')
# print(data_mlp.shape)

# 设置全局字体为 Times New Roman（用于英文）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 13

# 创建宋体字体属性（用于中文）
chinese_font = font_manager.FontProperties(family='SimSun', size=12)

# 数据
datasets = [2**i for i in range(15, 24)]  # 2^15, 2^16, ..., 2^23
# random = [29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28]
# expert = [84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71]
# seed_1 = [66.92, 70.50, 76.60, 77.53, 79.65, 77.88, 84.32, 85.43, 85.23]
# seed_2 = [63.17, 71.41, 72.61, 77.46, 76.15, 78.59, 84.46, 84.73, 84.89]
# seed_3 = [58.70, 66.80, 75.20, 73.31, 77.54, 82.57, 83.29, 83.33, 84.74]
# seed_4 = [61.79, 71.61, 77.03, 77.00, 77.84, 80.93, 84.57, 84.90, 85.65]

random = [29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28]
expert = [84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71]
# seed_2 = [67.15, 71.67, 78.27, 80.93, 82.75, 82.36, 84.99, 85.56, 84.83]
# seed_1 = [63.60, 72.18, 75.98, 81.66, 80.45, 82.87, 84.93, 84.96, 84.29]
# seed_4 = [60.13, 68.77, 77.37, 77.31, 81.54, 84.85, 84.16, 83.66, 84.34]
# seed_3 = [63.42, 73.98, 78.50, 80.20, 81.64, 85.01, 84.84, 85.13, 85.45]
# # 63.57 	71.65 	77.53 	80.02 	81.59 	83.77 	84.73 	84.83 	84.73
# # 62.65 	70.08 	75.36 	76.33 	77.80 	79.99 	84.16 	84.60 	85.13
# # 66.56 	66.56 	66.56 	66.56 	66.56 	66.56 	66.56 	66.56 	66.56
RD = [63.57, 71.65, 77.53, 80.02, 81.59, 83.77, 84.73, 84.83, 84.73]
NO_RD = [62.65, 70.08, 75.36, 76.33, 77.80, 79.99, 84.16, 84.60, 84.93]
NO_FH = [66.56, 66.56, 66.56, 66.56, 66.56, 66.56, 66.56, 66.56, 66.56]



# 设置中文字体
plt.rcParams['font.family'] = 'SimSun'  # 设置为中文宋体

# 创建一个画布和轴对象
fig, ax = plt.subplots(figsize=(7, 5))

# 绘制数据
# ax.plot(datasets, random, label="Random", marker='o', linestyle='-', color='b')
# ax.plot(datasets, expert, label="Expert", marker='o', linestyle='-', color='r')

# ax.plot(datasets, random, label="随机策略", marker='o', linestyle='-', color='b')
# ax.plot(datasets, expert, label="专家策略", marker='o', linestyle='-', color='r')
# ax.plot(datasets, seed_1, label="Seed=1", marker='o', linestyle='-', color='g')
# ax.plot(datasets, seed_2, label="Seed=2", marker='o', linestyle='-', color='c')
# ax.plot(datasets, seed_3, label="Seed=3", marker='o', linestyle='-', color='m')
# ax.plot(datasets, seed_4, label="Seed=4", marker='o', linestyle='-', color='y')

ax.plot(datasets, random, label="随机策略", marker='o', linestyle='-', color='b')
ax.plot(datasets, NO_RD, label="未添加扰动", marker='o', linestyle='-', color='c')
ax.plot(datasets, NO_FH, label="未场景泛化", marker='o', linestyle='-', color='m')
ax.plot(datasets, RD, label="添加扰动", marker='o', linestyle='-', color='g')
ax.plot(datasets, expert, label="专家策略", marker='o', linestyle='-', color='r')



# 设置标题和标签
# ax.set_title("数据集大小对网络性能的影响", fontsize=14)
ax.set_xlabel("数据集大小", fontsize=15)
ax.set_ylabel("测试场景评分", fontsize=15)

# 设置横坐标为对数刻度
ax.set_xscale('log')
# ax.set_yscale('log')

# 设置横坐标刻度
ax.set_xticks([2**i for i in range(15, 24)])  # 设置对应的 x 轴刻度
ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(15, 24)], rotation=0)  # 格式化显示

# 设置纵坐标刻度
ax.set_yticks(np.arange(25, 90, 5))

# # 只绘制有刻度的网格线
# ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)

# 添加图例
ax.legend(fontsize=11)

# 设置网格线在坐标轴下方
ax.set_axisbelow(True)

# 显示图形
plt.tight_layout()
# plt.savefig('场景泛化与扰动的影响.svg', format='svg')
plt.savefig('数据集大小对网络性能的影响3.svg', format='svg')

plt.show()







# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 设置全局字体为 Times New Roman（用于英文）
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# plt.rcParams['font.size'] = 15
#
# # 创建宋体字体属性（用于中文）
# chinese_font = font_manager.FontProperties(family='SimSun', size=15)
#
# # 数据定义
# datasets = [2**i for i in range(15, 24)]  # 2^15 到 2^23，共9个点
#
# # 未扰动数据（四个种子）
# seed_5 = [66.92, 70.50, 76.60, 77.53, 79.65, 77.88, 84.32, 85.43, 85.23]
# seed_6 = [63.17, 71.41, 72.61, 77.46, 76.15, 78.59, 84.46, 84.73, 84.89]
# seed_7 = [58.70, 66.80, 75.20, 73.31, 77.54, 82.57, 83.29, 83.33, 84.74]
# seed_8 = [61.79, 71.61, 77.03, 77.00, 77.84, 80.93, 84.57, 84.90, 85.65]
#
# # 扰动数据（四个种子）
# seed_1 = [63.60, 72.18, 75.98, 81.66, 80.45, 82.87, 84.93, 84.96, 84.29]
# seed_2 = [67.15, 71.67, 78.27, 80.93, 82.75, 82.36, 84.99, 85.56, 84.83]
# seed_3 = [63.42, 73.98, 78.50, 80.20, 81.64, 85.01, 84.84, 85.13, 85.45]
# seed_4 = [60.13, 68.77, 77.37, 77.31, 81.54, 84.85, 84.16, 83.66, 84.34]
#
# # 计算未扰动数据的最大值、最小值和均值
# no_perturb_seeds = np.array([seed_5, seed_6, seed_7, seed_8])
# no_perturb_min = np.min(no_perturb_seeds, axis=0)
# no_perturb_max = np.max(no_perturb_seeds, axis=0)
# no_perturb_mean = np.mean(no_perturb_seeds, axis=0)
#
# # 计算扰动数据的最大值、最小值和均值
# perturb_seeds = np.array([seed_1, seed_2, seed_3, seed_4])
# perturb_min = np.min(perturb_seeds, axis=0)
# perturb_max = np.max(perturb_seeds, axis=0)
# perturb_mean = np.mean(perturb_seeds, axis=0)
#
# # 基线数据
# random = [29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28]
# expert = [84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71]
# NO_FH = [66.56, 66.56, 66.56, 66.56, 66.56, 66.56, 66.56, 66.56, 66.56]
#
# # 创建画布和轴对象
# fig, ax = plt.subplots(figsize=(7, 5))
#
# # 绘制未扰动数据的范围和均值
# ax.fill_between(datasets, no_perturb_min, no_perturb_max, alpha=0.2, color='c')
# ax.plot(datasets, no_perturb_mean, marker='o', linestyle='-', color='c', label="未扰动均值")
#
# # 绘制扰动数据的范围和均值
# ax.fill_between(datasets, perturb_min, perturb_max, alpha=0.2, color='g')
# ax.plot(datasets, perturb_mean, marker='o', linestyle='-', color='g', label="扰动均值")
#
# # 绘制基线数据
# ax.plot(datasets, NO_FH, marker='o', linestyle='-', color='m', label="未场景泛化")
# ax.plot(datasets, random, marker='o', linestyle='-', color='b', label="随机策略")
# ax.plot(datasets, expert, marker='o', linestyle='-', color='r', label="专家策略")
#
# # 设置轴标签
# ax.set_xlabel("数据集大小", fontproperties=chinese_font)  # 中文使用宋体
# ax.set_ylabel("Return")  # 英文使用 Times New Roman
#
# # 设置横坐标为对数刻度
# ax.set_xscale('log')
#
# # 设置横坐标刻度
# ax.set_xticks([2**i for i in range(15, 24)])  # 设置对应的 x 轴刻度
# ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(15, 24)], rotation=0)  # LaTeX 格式，使用 Times New Roman
#
# # 设置纵坐标刻度
# ax.set_yticks(np.arange(25, 90, 5))
#
# # 添加图例（右下角）
# ax.legend(prop=chinese_font, loc='lower right')
#
# # 设置网格线在坐标轴下方
# ax.set_axisbelow(True)
#
# # 显示图形
# plt.tight_layout()
# plt.savefig('场景泛化与扰动的影响.svg', format='svg')
# plt.show()