# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 设置全局字体为 Times New Roman（用于英文）
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# plt.rcParams['font.size'] = 12
#
# # 创建宋体字体属性（用于中文）
# chinese_font = font_manager.FontProperties(family='SimSun', size=12)
#
# # 数据
# datasets = [2**i for i in range(15, 24)]  # 2^15, 2^16, ..., 2^23
# random = [29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28]
# expert = [84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71]
# seed_1 = [66.92, 70.50, 76.60, 77.53, 79.65, 77.88, 84.32, 85.43, 85.23]
# seed_2 = [63.17, 71.41, 72.61, 77.46, 76.15, 78.59, 84.46, 84.73, 84.89]
# seed_3 = [58.70, 66.80, 75.20, 73.31, 77.54, 82.57, 83.29, 83.33, 84.74]
# seed_4 = [61.79, 71.61, 77.03, 77.00, 77.84, 80.93, 84.57, 84.90, 85.65]
#
# # 设置中文字体
# plt.rcParams['font.family'] = 'SimSun'  # 设置为中文宋体
#
# # 创建一个画布和轴对象
# fig, ax = plt.subplots(figsize=(7, 5))
#
# ax.plot(datasets, random, label="随机策略", marker='o', linestyle='-', color='b')
# ax.plot(datasets, expert, label="专家策略", marker='o', linestyle='-', color='r')
# ax.plot(datasets, seed_1, label="Seed=1", marker='o', linestyle='-', color='g')
# ax.plot(datasets, seed_2, label="Seed=2", marker='o', linestyle='-', color='c')
# ax.plot(datasets, seed_3, label="Seed=3", marker='o', linestyle='-', color='m')
# ax.plot(datasets, seed_4, label="Seed=4", marker='o', linestyle='-', color='y')
#
# # 设置标题和标签
# ax.set_xlabel("数据集大小", fontsize=15)
# ax.set_ylabel("测试场景中表现", fontsize=15)
#
# # 设置横坐标为对数刻度
# ax.set_xscale('log')
#
# # 设置横坐标刻度
# ax.set_xticks([2**i for i in range(15, 24)])  # 设置对应的 x 轴刻度
# ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(15, 24)], rotation=0)  # 格式化显示
#
# # 设置纵坐标刻度
# ax.set_yticks(np.arange(25, 90, 5))
#
# # 添加图例
# ax.legend(fontsize=10)
#
# # 设置网格线在坐标轴下方
# ax.set_axisbelow(True)
#
# # 显示图形
# plt.tight_layout()
# # plt.savefig('场景泛化与扰动的影响.svg', format='svg')
# plt.savefig('数据集大小对网络性能的影响pld.svg', format='svg')
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置全局字体为 Times New Roman（用于英文）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12

# 创建宋体字体属性（用于中文）
chinese_font = font_manager.FontProperties(family='SimSun', size=12)

# 数据
datasets = [2**i for i in range(15, 24)]  # 2^15, 2^16, ..., 2^23
random = [29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28, 29.28]
expert = [84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71, 84.71]
seed_1 = [66.92, 70.50, 76.60, 77.53, 79.65, 77.88, 84.32, 85.43, 85.23]
seed_2 = [63.17, 71.41, 72.61, 77.46, 76.15, 78.59, 84.46, 84.73, 84.89]
seed_3 = [58.70, 66.80, 75.20, 73.31, 77.54, 82.57, 83.29, 83.33, 84.74]
seed_4 = [61.79, 71.61, 77.03, 77.00, 77.84, 80.93, 84.57, 84.90, 85.65]

# 设置中文字体
plt.rcParams['font.family'] = 'SimSun'  # 设置为中文宋体

# 创建一个画布和轴对象
fig, ax = plt.subplots(figsize=(7, 5))

# 使用不同的标记，保留原颜色
ax.plot(datasets, random, label="随机策略", marker='o', linestyle='-', color='b')
ax.plot(datasets, expert, label="专家策略", marker='s', linestyle='-', color='r')
ax.plot(datasets, seed_1, label="Seed=1", marker='^', linestyle='-', color='g')
ax.plot(datasets, seed_2, label="Seed=2", marker='v', linestyle='-', color='c')
ax.plot(datasets, seed_3, label="Seed=3", marker='D', linestyle='-', color='m')
ax.plot(datasets, seed_4, label="Seed=4", marker='*', linestyle='-', color='y')

# 设置标题和标签
ax.set_xlabel("数据集大小", fontsize=15)
ax.set_ylabel("测试场景得分", fontsize=15)

# 设置横坐标为对数刻度
ax.set_xscale('log')

# 设置横坐标刻度
ax.set_xticks([2**i for i in range(15, 24)])  # 设置对应的 x 轴刻度
ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(15, 24)], rotation=0)  # 格式化显示

# 设置纵坐标刻度
ax.set_yticks(np.arange(25, 90, 5))

# 添加图例
ax.legend(fontsize=10)

# 设置网格线在坐标轴下方
ax.set_axisbelow(True)

# 显示图形
plt.tight_layout()
plt.savefig('数据集大小对网络性能的影响pld.svg', format='svg')
plt.show()