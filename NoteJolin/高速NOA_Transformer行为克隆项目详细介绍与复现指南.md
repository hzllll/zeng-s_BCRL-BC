# 高速NOA、Transformer、辅助驾驶与行为克隆项目详细介绍与复现指南

## 1. 这份项目到底在做什么

这是一个面向**高速基本路段 NOA（Navigate on Autopilot，领航辅助驾驶）**的行为克隆项目。它的目标不是直接做端到端“自动驾驶一把梭”，而是走一条更工程化、更容易解释的路线：

1. 先用一个较强的**专家策略**在 OnSite 高速场景中开车；
2. 把专家在各种场景中的驾驶决策记录下来，做成监督学习数据集；
3. 用 `Transformer` 学习“看到什么交通状态，就输出什么控制动作”；
4. 再把训练好的模型装回 OnSite 仿真环境里做**闭环测试**；
5. 最后再放到 `highway-env` 和 `NGSIM` 上测试泛化能力。

一句话概括：

**这是一个“用专家规划器教 Transformer 开高速”的模仿学习/行为克隆项目。**

从工程链路看，它其实包含了三层能力：

1. **规划层**：专家策略负责“会开车”；
2. **学习层**：Transformer 负责“学会专家怎么开”；
3. **评估层**：OnSite、highway-env、NGSIM 负责“检验它学得像不像、稳不稳、能不能泛化”。

---

## 2. 先给小白讲明白几个核心概念

### 2.1 什么是 NOA

NOA 是领航辅助驾驶。放到这个项目里，可以通俗理解成：

- 车辆在高速基本路段场景下，
- 需要根据周围车辆、目标区域、车道边界等信息，
- 自动完成跟车、保持车道、绕行、变道、通过场景等决策。

它不是“从起点到终点的全栈真实车系统”，而是一个**在标准化仿真场景中验证驾驶策略能力**的研究型/实验型项目。

### 2.2 什么是行为克隆

行为克隆（Behavior Cloning, BC）是模仿学习里最直观的一种方法：

- 输入：当前交通状态；
- 标签：专家此刻采取的动作；
- 学习目标：让神经网络在看到相似状态时，输出与专家尽量接近的动作。

换句话说，它像是“看老师开车录像学驾驶”。

### 2.3 什么是专家策略

本项目里，专家策略不是人类驾驶员，而是一个**规则/规划算法**，核心在：

- `planner_example(谢师兄用的触须法)/Planner.py`
- `planner_example(谢师兄用的触须法)/EM_Planner.py`
- `planner/main_get_multi_choose_dataset_core.py`

其中最关键的思想是**触须法 / 采样路径簇规划**：

- 先生成多条候选路径；
- 再结合道路边界、障碍物、动态风险给每条路径打分；
- 选出代价最低的一条作为当前规划结果；
- 再配合速度规划、控制器输出加速度和转角。

所以专家不是“拍脑袋开”，而是“先生成候选，再择优执行”。

### 2.4 什么是 Transformer

本项目的主模型在 `get_clone_learning_Transformer6_7.py` 中定义。它不是语言模型，而是一个用于轨迹/动作预测的编码器-解码器网络。

它做的事情是：

- 输入当前时刻的交通状态；
- 编码主车、目标点、周围 8 辆车的信息；
- 一次性输出未来 5 个时间步的控制动作：
  - 加速度 `acc`
  - 转角 `steer`

也就是说，它不是只预测一步，而是预测**未来 5 步控制序列**。

### 2.5 什么是闭环测试

闭环测试的意思是：

- 模型输出动作；
- 仿真器真的执行这个动作；
- 执行动作后，环境会变化；
- 下一时刻模型要基于“自己上一步造成的新环境”继续决策。

这和只拿固定数据做预测不同。闭环测试更接近真实驾驶，因为模型一旦某一步偏了，后面就会越来越偏。

### 2.6 什么是开环测试

开环测试常见于数据集回放，比如 `test_ngsim_transformer.py`。

它的特点是：

- 不真的接管环境；
- 只是给模型一个当前状态；
- 看模型预测和真实未来轨迹/动作差多少。

所以：

- **OnSite / highway-env** 更偏闭环；
- **NGSIM** 更偏开环。

---

## 3. 项目主线与非主线：先分清什么最重要

用户给出的要求非常关键，这里先明确：

### 3.1 主线

这份项目的主线是：

1. OnSite 高速场景
2. 专家策略生成数据
3. `Transformer` 行为克隆训练
4. OnSite 闭环测试评分
5. highway-env / NGSIM 泛化评估

对应主线脚本主要有：

- `GenerateScenarios.py`
- `planner/main_get_multi_choose_dataset_core.py`
- `Get_Dataset_Closest_Proper_RL.py`
- `test_dataset_jolin.py`
- `get_clone_learning_Transformer6_7.py`
- `planner/sup_train/test_transformer_simulation.py`
- `test_highway_env_transformer.py`
- `test_ngsim_transformer.py`

### 3.2 对比实验

以下脚本主要是对比实验，不是主线：

- `get_clone_learning_lw_mi.py`：GRU
- `LSTM_get_clone_learning.py`：LSTM
- `vanilla_RNN_get_clone_learning.py`：普通 RNN
- `AR_get_clone_learning.py`：自回归 Transformer 基线
- `run_compareRNN.sh`：统一启动对比实验

### 3.3 可忽略部分

根据你的要求，下面这些不是当前要重点看的：

- `*his5`
- `*Hist5`
- `get_clone_learning_Transformer_Hist5.py`
- `planner/sup_train/test_transformer_history5.py`
- `get_clone_learning_GRU_Hist5.py`

这些属于另一条“带历史帧/历史轨迹”的实验支线，不是当前 Transformer6_7 主线复现的重点。

---

## 4. 仓库里最值得先认识的文件与文件夹

### 4.1 数据与结果相关

- `dataset/`
  - 训练用 `.npy` 数据集
  - 当前训练脚本直接读取：
    - `dataset/GRU_choose_closest8_properRL_0.npy`
    - `dataset/GRU_choose_closest8_properRL_1.npy`
    - `dataset/GRU_choose_closest8_properRL_out.npy`

- `Transformer_checkpoints/`
  - Transformer 训练好的模型权重 `.pth`

- `Transformer_plots/`
  - Transformer 训练/验证损失曲线 `.svg`

- `planner/inputs/inputs_B`
  - OnSite 训练集场景

- `planner/inputs/inputs_C`
  - OnSite 测试集场景

- `planner/outputs/`
  - OnSite 闭环测试输出目录，每个实验会生成一批 `*_result.csv`

- `highway_env_results/`
  - highway-env 泛化测试输出

- `ngsim_results/`
  - NGSIM 开环评估输出

### 4.2 训练与测试主脚本

- `get_clone_learning_Transformer6_7.py`
  - 主训练脚本

- `planner/sup_train/test_transformer_simulation.py`
  - OnSite 闭环测试脚本

- `test_highway_env_transformer.py`
  - highway-env 闭环泛化测试脚本

- `test_ngsim_transformer.py`
  - NGSIM 开环泛化评估脚本

### 4.3 专家策略与数据生产

- `planner_example(谢师兄用的触须法)/Planner.py`
  - 触须法规划器主干

- `planner_example(谢师兄用的触须法)/EM_Planner.py`
  - 速度规划、ST 图、QP/OSQP 相关核心

- `GenerateScenarios.py`
  - 场景扰动/扩增脚本，批量改初始状态与目标区域

- `planner/main_get_multi_choose_dataset_core.py`
  - 跑专家策略，输出带辅助字段的结果 CSV

- `Get_Dataset_Closest_Proper_RL.py`
  - 从专家结果 CSV 构造最终训练样本并保存为 `.npy`

### 4.4 对比实验与分析

- `get_clone_learning_lw_mi.py`
- `LSTM_get_clone_learning.py`
- `vanilla_RNN_get_clone_learning.py`
- `AR_get_clone_learning.py`
- `run_compareRNN.sh`
- `Jolindraw/plot_comparison.py`
- `Jolindraw/plot_comparison_v2.py`
- `Jolindraw/plot_comparison_v3.py`
- `Jolindraw/plot_comparison_TFep200.py`

---

## 5. 端到端主流程总览

如果把整个项目看成一条工业流水线，可以分成下面 6 个大阶段：

1. **阶段 A：准备/扩增 OnSite 场景**
2. **阶段 B：运行专家策略，得到专家驾驶结果 CSV**
3. **阶段 C：把 CSV 加工成 `.npy` 监督学习数据**
4. **阶段 D：训练 Transformer，得到模型权重与损失图**
5. **阶段 E：OnSite 闭环测试，导出结果 CSV，交给官方软件评分**
6. **阶段 F：highway-env / NGSIM 泛化测试**

下面按这 6 个阶段，一步一步详细讲。

---

## 6. 阶段 A：准备或扩增 OnSite 场景

### 6.1 这一阶段的目的

目标是让专家和模型看到更多样的初始状态与目标区域，而不是只在极少数“标准起点”上训练。

从代码上看，这一阶段主要由 `GenerateScenarios.py` 完成。它不是训练脚本，也不是测试脚本，而是**场景加工脚本**。

### 6.2 主要脚本

#### `GenerateScenarios.py`

这个脚本做了几件事：

1. 读取 OnSite 场景目录；
2. 读取每个场景中的 `.xosc` 文件；
3. 根据道路方向和车道边界，随机生成多组：
   - 初始位置 `x_init / y_init`
   - 初始速度 `v_init`
   - 初始航向角 `heading_init`
   - 目标区域 `y_target`
4. 把修改后的 `.xosc` 和对应 `.xodr` 拷贝到新目录下。

也就是说，它相当于在做：

**“用同一个原始场景模板，批量制造更多起点不同、目标不同的训练样本场景。”**

### 6.3 代码里能明确看出的关键信息

从 `GenerateScenarios.py` 可以明确看出：

- 它遍历多个 `inputs` 目录；
- 会根据道路左右方向分别处理；
- 会为每个场景生成很多扰动版本；
- 会修改 `.xosc` 中注释字段里的：
  - `v_init`
  - `x_init`
  - `y_init`
  - `heading_init`
  - `y_target`

### 6.4 输入与输出

输入：

- 原始 OnSite 场景目录
- 每个场景里的 `.xosc` 和 `.xodr`

输出：

- 一批新的场景目录
- 每个目录里有修改后的 `.xosc`
- 对应复制好的 `.xodr`

### 6.5 建议运行方式

这个脚本当前带有明显的**Windows 硬编码路径**，例如 `D:\...`，因此在 Linux 上直接运行前必须先改路径。

建议流程：

1. 先打开 `GenerateScenarios.py`
2. 把里面的 `input_dir`、`output_dir` 等路径改成当前机器真实路径
3. 再运行：

```bash
python GenerateScenarios.py
```

### 6.6 这一阶段完成后你应该看到什么

你应该能在新的输出目录中看到大量按场景复制扩增后的子文件夹，每个子文件夹里至少有：

- 一个 `.xosc`
- 一个 `.xodr`

如果这一步做完，说明“训练/测试场景池”已经准备好了。

---

## 7. 阶段 B：运行专家策略，得到专家驾驶结果 CSV

### 7.1 这一阶段的目的

这一步的目标是：

**让专家规划器在 OnSite 场景中真正开起来，并把每一步的驾驶轨迹和状态记录下来。**

这是整个项目的“教师示范”阶段。

### 7.2 主要脚本

这一阶段至少涉及两类代码：

#### 1）专家规划器核心

- `planner_example(谢师兄用的触须法)/Planner.py`
- `planner_example(谢师兄用的触须法)/EM_Planner.py`
- `planner_example(谢师兄用的触须法)/control.py`

#### 2）将专家接入 OnSite 并批量跑场景

- `planner/main_get_multi_choose_dataset_core.py`

### 7.3 专家规划器到底做了什么

#### `planner_example(谢师兄用的触须法)/Planner.py`

这个文件可以理解为“专家大脑外壳”。它负责：

1. 读取当前自车与障碍车状态；
2. 构造 9 条左右不同偏移的候选路径；
3. 评估每条路径的代价；
4. 选出最优路径；
5. 调用速度规划模块；
6. 输出车辆控制量。

这里最重要的几个函数/逻辑是：

- `plan_paths_sn_90`
  - 生成 9 条候选路径，是触须法的核心之一

- `select_best_path_index_numpy_dyn`
  - 对候选路径打分，综合考虑：
    - 路径延续性
    - 偏离全局路径程度
    - 是否越界
    - 动态障碍物风险

- `exec_waypoint_nav_demo_em_stitch_sn_qp4` / `exec_waypoint_nav_demo_em_stitch_sn`
  - 负责实际规划循环与控制输出

#### `planner_example(谢师兄用的触须法)/EM_Planner.py`

这个文件更像“专家大脑的数学内核”，主要负责：

- ST 图构建
- 动态规划 `speed_dp`
- 凸空间生成 `convex_space_gen`
- OSQP 二次规划 `speed_planning_osqp`
- 轨迹增密 `increase_points`
- SL 坐标转换、障碍物投影、碰撞代价计算

通俗地说：

- `Planner.py` 更像“我要选哪条路、怎么跟踪”；
- `EM_Planner.py` 更像“这条路上我该怎么分配速度，怎么避障更稳”。

### 7.4 `planner/main_get_multi_choose_dataset_core.py` 的角色

这个脚本非常关键。它可以看作：

**“把专家规划器接到 OnSite 环境里，批量跑输入场景，并把结果整理成后续可做数据集的 CSV。”**

它的工作链路大致是：

1. 用 `scenarioOrganizer` 读取输入场景目录；
2. 用 `env.Env()` 创建 OnSite 仿真环境；
3. 对每个场景执行：
   - `envi.make(...)` 初始化环境
   - 提取道路信息、目标区域、初始状态
   - 调用 `global_path(...)`
   - 调用 `LP2(...)` 专家规划器
   - 循环执行 `planner.exec_waypoint_nav_demo(...)`
   - 每一步执行 `envi.step(action)`
4. 场景结束后：
   - 读取该场景结果 CSV
   - 在末尾插入辅助列：
     - `steer_ego`
     - `goal_x`
     - `goal_y`
     - `lane1`
     - `lane2`
     - `lane3`
     - `lane4`
   - 再写到新目录

这非常重要，因为后续做监督学习时，训练数据不只要知道“专家做了什么动作”，还要知道：

- 目标点在哪里；
- 车道边界在哪里；
- 每个时刻的自车转角是多少。

### 7.5 这一阶段的输入与输出

输入：

- 扩增后的 OnSite 场景目录
- 专家规划器代码

输出：

- OnSite 原始结果 CSV
- 增强版结果 CSV（多了 `steer_ego`、`goal_x`、`lane1~4` 等列）

### 7.6 建议运行命令

同样，这个脚本当前也有大量硬编码路径，尤其是 Windows 风格路径，运行前需要先改。

修改完成后建议运行：

```bash
python planner/main_get_multi_choose_dataset_core.py
```

### 7.7 运行完后会看到什么

你应该会在输出目录中看到很多类似：

- `xxxx_result.csv`

这些 CSV 就是专家在场景中“开完之后”的结果记录，是行为克隆数据的原始素材。

---

## 8. 阶段 C：把专家 CSV 加工成 `.npy` 训练数据

### 8.1 这一阶段的目的

这一步是把“仿真结果表格”变成“神经网络能直接读取的张量样本”。

也就是把：

- 当前时刻交通状态
- 对应未来 5 步专家动作

整理成监督学习样本。

### 8.2 核心脚本

#### `Get_Dataset_Closest_Proper_RL.py`

这是当前仓库里最直接负责**CSV → NPY** 的脚本。

### 8.3 它具体做了什么

从代码可明确看出，它会：

1. 遍历一个目录下的所有专家结果 CSV；
2. 跳过终止状态不符合要求的场景；
3. 针对每个有效场景逐帧提取样本；
4. 构造每个样本的输入状态 `states`；
5. 再拼上未来 5 步的标签动作；
6. 最终堆叠成一个大数组并 `np.save(...)`。

### 8.4 它构造的单条样本长什么样

这个脚本最终构造的是**64 维样本**：

- 前 54 维：输入特征
- 后 10 维：监督标签

其中：

#### 输入 54 维 = 主车 6 维 + 8 辆障碍车 × 6 维

主车 6 维包括：

1. `v_ego`
2. `yaw_ego`
3. `goal_x - x_ego`
4. `goal_y - y_ego`
5. 上车道边界相对量
6. 下车道边界相对量

每辆障碍车 6 维包括：

1. 相对纵向距离
2. 相对横向距离
3. 相对速度
4. 航向角
5. 所在车道左边界相对主车的位置
6. 所在车道右边界相对主车的位置

最多保留最近的 8 辆车，不足时用占位补齐：

- `[200, 0, 0, 0, lane_left, lane_right]`

#### 标签 10 维 = 未来 5 步控制动作

每一步包含：

- `a_ego`
- `steer_ego`

一共未来 5 步，所以 10 维。

### 8.5 左右方向统一

这个脚本有个很关键的设计：

如果场景是“从右往左开”，它会做坐标镜像：

- `x -> -x`
- `y -> -y`
- `goal -> -goal`
- `lane -> -lane`
- `yaw -> yaw - pi`

这样做的好处是：

**把双向高速场景统一成一种坐标语义，让网络不用分别学“左到右”和“右到左”两套规则。**

### 8.6 会生成什么文件

脚本末尾可以看到：

- `GRU_choose_closest8_properRL_0.npy`
- `GRU_choose_closest8_properRL_1.npy`
- `GRU_choose_closest8_properRL_out.npy`

虽然当前文件里实际取消注释的是 `GRU_choose_closest8_properRL_guass_0.npy`，但从整个仓库主训练脚本的调用方式可以确认，真正训练使用的是：

- `dataset/GRU_choose_closest8_properRL_0.npy`
- `dataset/GRU_choose_closest8_properRL_1.npy`
- `dataset/GRU_choose_closest8_properRL_out.npy`

也就是说，主线训练数据最终就是这 3 个文件。

### 8.7 数据规模

从现有训练日志可以直接确认这 3 个文件的规模：

- `GRU_choose_closest8_properRL_0.npy`：`(2807565, 64)`
- `GRU_choose_closest8_properRL_1.npy`：`(2718791, 64)`
- `GRU_choose_closest8_properRL_out.npy`：`(4411230, 64)`

合并后总样本量：

- `9,937,586` 条样本

这说明这个项目的数据量其实非常大，不是一个“小打小闹”的玩具训练。

### 8.8 数据维度检查脚本

#### `test_dataset_jolin.py`

这个脚本用来检查 `.npy` 数据格式是否正确，尤其用于确认：

- 是不是 64 列；
- 是否符合当前 6 维主车 + 8×6 障碍车 + 10 维标签的设计。

建议在正式训练前先跑：

```bash
python test_dataset_jolin.py
```

### 8.9 建议运行命令

运行前先改好 `Get_Dataset_Closest_Proper_RL.py` 里的输入目录 `abs_path` 和目标 `np.save` 文件名。

然后执行：

```bash
python Get_Dataset_Closest_Proper_RL.py
```

如果想把输出放到 `dataset/`，建议把最终的 `.npy` 文件移动或重命名到：

- `dataset/GRU_choose_closest8_properRL_0.npy`
- `dataset/GRU_choose_closest8_properRL_1.npy`
- `dataset/GRU_choose_closest8_properRL_out.npy`

---

## 9. 阶段 D：训练 Transformer 主模型

### 9.1 主脚本

#### `get_clone_learning_Transformer6_7.py`

这是本项目最核心的训练脚本，也是你当前应该最重视的代码文件。

### 9.2 这个脚本做了什么

它完整覆盖了：

1. 加载多个 `.npy` 数据文件
2. 合并数据集
3. 切分训练集/验证集
4. 定义 Transformer 模型
5. 定义损失函数、优化器、学习率调度器
6. 训练模型
7. 保存最优权重
8. 绘制损失曲线

### 9.3 输入数据

这个脚本默认读取：

- `dataset/GRU_choose_closest8_properRL_0.npy`
- `dataset/GRU_choose_closest8_properRL_1.npy`
- `dataset/GRU_choose_closest8_properRL_out.npy`

然后在 `preprocess_data()` 中完成：

- 输入 `states = dataset[:, :-10]`，共 54 维
- 标签 `actions = dataset[:, -10:]`，共 10 维

标签还会做归一化：

- 加速度除以 `6`
- 转角除以 `0.15`
- 最终裁剪到 `[-1, 1]`

这意味着模型输出也被设计成 `[-1, 1]` 区间，最后在闭环测试时再反归一化。

### 9.4 模型结构

`TransformerTrajectoryPredictor` 这部分值得重点理解。

#### 输入组织方式

输入共 54 维，其中：

- 前 6 维：主车 + 目标点
- 后 48 维：8 辆障碍车 × 6 维

模型会把它拆成：

- 1 个主车/目标 token
- 8 个障碍车 token

所以编码器输入序列长度是：

- `1 + 8 = 9`

#### 编码器

编码器负责建模当前这一帧里：

- 主车
- 目标点
- 周围 8 车

之间的关系。

#### 解码器

解码器里没有传统的自回归历史动作输入，而是用：

- `future_queries`

作为未来 5 个时刻的查询向量，一次性生成未来 5 步动作。

#### 输出

输出维度为：

- `(batch, 5, 2)`

即：

- 未来 5 步
- 每步输出 2 个值：
  - `acc`
  - `steer`

### 9.5 训练超参数

当前脚本中能明确读到的主配置是：

- `EXP_NAME = "Exp-6"`
- `D_MODEL = 256`
- `FFN_DIM = 1024`
- `num_encoder_layers = 3`
- `num_decoder_layers = 3`
- `SEQ_LENGTH = 5`
- `CAR_NUM = 8`
- `LEARNING_RATE = 5e-4`
- `BATCH_SIZE = 1024`
- `EPOCHS = 512`
- `EARLY_STOPPING_PATIENCE = 500`
- `dropout = 0.15`
- 加速度损失权重 `1`
- 转角损失权重 `5`

### 9.6 学习率调度策略

这个脚本当前使用的是：

- `AdamW`
- `CosineAnnealingWarmRestarts`

这也是你在 `NoteJolin/transformer6_7迭代记录.md` 里反复讨论的一个重点。

它的直观作用是：

- 学习率周期性退火；
- 在某些轮次重新拉高学习率；
- 尝试跳出局部最优；
- 提升最终泛化能力。

### 9.7 输出文件

这个脚本会自动创建目录：

- `Transformer_checkpoints/`
- `Transformer_plots/`

并生成：

#### 模型权重

命名规则类似：

- `Transformer_checkpoints/Exp-6_Tf_trajectory_model_时间_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth`

#### 损失曲线图

命名规则类似：

- `Transformer_plots/Exp-6_Tf_loss_curve_时间_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.svg`

### 9.8 建议训练命令

在仓库根目录执行：

```bash
python get_clone_learning_Transformer6_7.py
```

### 9.9 训练过程中建议关注什么

建议同时观察：

1. 训练损失是否持续下降
2. 验证损失是否同步下降
3. 是否出现明显过拟合
4. 学习率重启点附近 loss 是否有波动
5. 最优模型大概出现在哪个 epoch

### 9.10 补充：脚本支持“续跑”

脚本里有：

- 如果 `MODEL_SAVE_PATH` 已存在，就加载已有模型继续训练

所以它支持简单意义上的“续跑”，但要注意：

- 它只恢复模型权重；
- **不会恢复优化器和调度器的完整状态**；
- 因此学习率节奏可能与真正断点续训不同。

这也是你实验记录里“断电重启反而效果更好”的一个重要背景。

---

## 10. 阶段 E：OnSite 闭环测试与官方评分

### 10.1 主脚本

#### `planner/sup_train/test_transformer_simulation.py`

这份脚本负责把训练好的 Transformer 装回 OnSite 环境中跑闭环。

### 10.2 这个脚本做了什么

它包含下面几件核心事情：

1. 定义与训练阶段完全一致的 Transformer 模型结构；
2. 读取某个 `.pth` 权重；
3. 从 OnSite 仿真环境提取 54 维状态；
4. 调模型输出未来 5 步动作；
5. 反归一化动作：
   - `acc *= 6`
   - `steer *= 0.15`
6. 依次执行这 5 步动作；
7. 场景结束后输出结果 CSV。

### 10.3 为什么一定要“训练与测试超参数一致”

这个脚本里有一句很关键的话：

> 请确保这些参数与训练时的配置完全一致！

原因很简单：

- `d_model`
- `FFN_DIM`
- `encoder/decoder 层数`
- `CAR_NUM`
- `SEQ_LENGTH`

只要有一个不一致，`.pth` 权重就可能加载失败，或者即使强行对齐也会语义错误。

### 10.4 特征提取是怎么做的

脚本中最关键的特征函数是：

- `observation_to_state1_simRL_proper`
- `observation_to_state2_simRL_proper`

它们负责把 OnSite 观测转换成训练阶段一致的 54 维输入。

主要逻辑包括：

1. 提取主车速度、航向角
2. 提取目标点相对位置
3. 提取车道边界相对位置
4. 从所有周围车中筛出横向距离不太远的车
5. 选最近的 8 辆车
6. 组成和训练阶段一致的特征顺序
7. 不足 8 辆则补零/补占位

这一点非常关键：

**训练时怎么组织特征，测试时就必须怎么组织特征。**

### 10.5 支持批量跑多个实验

脚本底部定义了一个 `experiments` 列表，当前示例里有：

- `Exp-3`
- `Exp-4`
- `Exp-5`

它会自动：

1. 在 `Transformer_checkpoints/` 中找对应实验名开头的模型；
2. 对 `inputs_B` 和 `inputs_C` 依次跑测试；
3. 输出到 `planner/outputs/` 下按实验名命名的新目录。

### 10.6 输入目录

当前脚本写死了：

- `planner/inputs/inputs_B`
- `planner/inputs/inputs_C`

从仓库实际内容可以确认，这两个目录里包含大量 OnSite 场景子目录，每个场景一般包括：

- `*_exam.xosc`
- `*.xodr`

### 10.7 输出目录与输出文件

脚本会在：

- `planner/outputs/`

下生成实验目录，名字类似：

- `Exp-5_rC_0420_12_1024BS_256dm_1024FFN_ednc3_CoAnWarmRest_zDATASET`

每个目录下会生成大量：

- `xxxx_result.csv`

这些 `result.csv` 是后续交给 OnSite 官方评分软件的输入文件。

### 10.8 建议运行步骤

先改这几个地方：

1. `MODEL_PATH` 或实验匹配规则
2. `experiments` 中的超参数配置
3. 确认模型结构和训练时一致

然后运行：

```bash
python planner/sup_train/test_transformer_simulation.py
```

### 10.9 这一步得到什么结果

这一阶段结束后，你会得到两类结果：

1. **闭环驾驶结果 CSV**
2. **OnSite 官方评分**

评分维度是：

- 安全性 Safety（50）
- 效率性 Efficiency（30）
- 舒适性 Comfort（20）

总分 100。

### 10.10 如何理解这三个分数

#### 安全性

最重要，主要看：

- 是否碰撞
- 是否发生危险行为
- 是否违反安全约束

#### 效率性

主要看：

- 是否能较顺畅完成任务
- 是否龟速
- 是否拖延通过场景

#### 舒适性

主要看：

- 加速度是否过猛
- 转向是否过急
- 是否频繁抖动

这三个指标一起构成“既能开过去，又别开得太莽，也别开得太肉”的综合评价。

---

## 11. 阶段 F：泛化能力测试

这个项目的一个优点是，没有把评估只停留在 OnSite 内部，而是继续做了两个方向的泛化：

1. `highway-env` 闭环泛化
2. `NGSIM` 开环泛化

---

## 12. highway-env 泛化测试

### 12.1 主脚本

#### `test_highway_env_transformer.py`

### 12.2 这一脚本的定位

它相当于把 OnSite 学到的策略迁移到另一个完全不同的仿真框架：

- `gymnasium`
- `highway-env`

这一步不是为了得到 OnSite 官方分，而是为了回答一个更研究的问题：

**模型是否只会做 OnSite 题库，还是换一个高速仿真环境也能开？**

### 12.3 它做了什么

1. 载入 Transformer 模型；
2. 创建 `highway-v0` 环境；
3. 把 highway-env 的车道和车辆信息映射到 OnSite 同风格特征；
4. 构造与训练阶段一致的 54 维输入；
5. 一次推理得到 5 步动作；
6. 在环境中执行；
7. 统计每回合：
   - 是否碰撞
   - 平均速度
   - 累计奖励
   - 推理耗时

### 12.4 关键对齐点

这个脚本非常强调“输入格式对齐训练集”，包括：

- 主车 6 维特征
- 8 车最近邻策略
- 车道边界表示方式
- 未来 5 步动作输出

也就是说，它不是简单拿模型去跑，而是很认真地做了特征语义对齐。

### 12.5 输出结果

它会把结果保存到：

- `highway_env_results/`

并输出：

- 文本报告 `.txt`
- 图表 `.svg`

### 12.6 建议运行命令

```bash
python test_highway_env_transformer.py
```

### 12.7 你会看到什么

终端会逐回合打印：

- 是否碰撞
- 步数
- 时长
- 累计奖励
- 平均速度
- 推理耗时

最终还会给出：

- 安全完成率
- 碰撞率
- 平均奖励
- 平均速度
- P50/P95/P99 推理时间

这一步主要用于判断：

- 模型是否有基础的闭环控制能力；
- 模型运行是否够快；
- 换环境后是否立刻崩掉。

---

## 13. NGSIM 泛化测试

### 13.1 主脚本

#### `test_ngsim_transformer.py`

### 13.2 这一脚本的定位

它用的是**真实高速公路轨迹数据**，而不是仿真生成数据。

NGSIM 是美国真实交通轨迹数据集，因此这一部分回答的问题是：

**模型在真实分布的数据上，对未来短时动作/轨迹的预测误差怎么样？**

### 13.3 这一步为什么叫开环

因为它不是让模型接管车辆闭环驾驶，而是：

1. 回放某一时刻真实交通状态；
2. 让模型预测未来 5 步动作；
3. 用简单运动学把动作积分成未来轨迹；
4. 再和 NGSIM 里真实轨迹比误差。

所以它是典型的开环评估。

### 13.4 这个脚本做了什么

主要流程如下：

1. 读取 NGSIM CSV
2. 把英尺单位转成米
3. 计算航向角 `yaw`
4. 从航向变化率反推 `steer`
5. 计算目标车道边界
6. 构造与训练完全一致的 54 维输入
7. 模型预测未来 5 步 `acc + steer`
8. 积分得到预测轨迹
9. 与真实未来轨迹计算误差

### 13.5 评估指标

脚本中明确给出了几类指标：

- `ADE`
  - 平均位移误差

- `FDE`
  - 最终位移误差

- `RMSE_acc`
  - 加速度均方根误差

- `RMSE_steer`
  - 转向角均方根误差

- `RMSE_lat`
  - 横向位置误差

- `RMSE_lon`
  - 纵向位置误差

这些指标能帮助你判断：

- 模型是不是大体知道应该往哪开；
- 更擅长纵向控制还是横向控制；
- 对真实道路数据有没有明显 domain gap。

### 13.6 输出结果

输出目录：

- `ngsim_results/`

输出文件包括：

- 日志 `.log`
- 文本报告 `.txt`
- 图表 `.svg`

### 13.7 建议运行命令

先把 NGSIM CSV 放到脚本指定目录，或者修改：

- `NGSIM_CSV_PATH`

然后运行：

```bash
python test_ngsim_transformer.py
```

---

## 14. 对比实验：为什么仓库里还有 RNN、LSTM、GRU、AR-Transformer

### 14.1 它们的作用

这些不是主线模型，而是用来回答：

**Transformer 到底是不是比传统序列模型更适合这个任务？**

所以它们承担的是“基线”角色。

### 14.2 主要脚本

- `get_clone_learning_lw_mi.py`：GRU
- `LSTM_get_clone_learning.py`：LSTM
- `vanilla_RNN_get_clone_learning.py`：普通 RNN
- `AR_get_clone_learning.py`：自回归 Transformer

### 14.3 它们的共同点

这些脚本基本都：

- 读取同一组 `dataset/GRU_choose_closest8_properRL_*.npy`
- 使用相同的 54 维输入
- 预测未来 5 步 `acc + steer`
- 也输出 `.pth` 和损失图

这保证了对比公平性。

### 14.4 统一启动脚本

#### `run_compareRNN.sh`

这个脚本会分两批跑：

第一批：

- `vanilla_RNN_get_clone_learning.py`
- `LSTM_get_clone_learning.py`

第二批：

- `AR_get_clone_learning.py`
- `get_clone_learning_lw_mi.py`

同时把日志写入 `train_log/`。

建议命令：

```bash
bash run_compareRNN.sh
```

但需要注意：

- 这个脚本最后有 `shutdown`
- 在 Linux 服务器上执行前必须确认是否真的要自动关机
- 如果不想关机，先删掉或注释掉最后一行

### 14.5 对比实验输出

会生成：

- `RNN_checkpoints/`
- `LSTM_checkpoints/`
- `GRU_checkpoints/`
- `Transformer_checkpoints/`

以及：

- `RNN_plots/`
- `LSTM_plots/`
- `GRU_plots/`
- `Transformer_plots/`

### 14.6 日志与画图脚本

以下脚本用于把多个实验 loss 画到一起：

- `Jolindraw/plot_comparison.py`
- `Jolindraw/plot_comparison_v2.py`
- `Jolindraw/plot_comparison_v3.py`
- `Jolindraw/plot_comparison_TFep200.py`

如果想整理训练曲线对比图，可以运行类似：

```bash
python Jolindraw/plot_comparison.py
```

---

## 15. 一份适合新手照着走的完整复现顺序

下面给一套“从 0 到 1”的建议顺序。注意，这里分成两种情况：

- **情况 A：你要完整重建数据**
- **情况 B：你只想快速复现训练与测试**

### 15.1 情况 A：完整重建数据 + 训练 + 测试

#### 第 1 步：准备场景

修改 `GenerateScenarios.py` 中的路径后运行：

```bash
python GenerateScenarios.py
```

目标：

- 生成扩增后的场景目录

#### 第 2 步：跑专家策略并导出结果 CSV

修改 `planner/main_get_multi_choose_dataset_core.py` 中的输入输出路径后运行：

```bash
python planner/main_get_multi_choose_dataset_core.py
```

目标：

- 在输出目录生成大量 `*_result.csv`
- 每个 CSV 里包含专家轨迹和补充字段

#### 第 3 步：构造 `.npy` 训练集

修改 `Get_Dataset_Closest_Proper_RL.py` 中的 `abs_path` 和 `np.save()` 目标文件后运行：

```bash
python Get_Dataset_Closest_Proper_RL.py
```

目标：

- 生成 `.npy` 数据文件

#### 第 4 步：检查数据维度

```bash
python test_dataset_jolin.py
```

目标：

- 确认数据为 64 列
- 确认输入 54 维、标签 10 维

#### 第 5 步：训练 Transformer

```bash
python get_clone_learning_Transformer6_7.py
```

目标：

- 生成 Transformer 权重 `.pth`
- 生成 loss 曲线 `.svg`

#### 第 6 步：OnSite 闭环测试

先在 `planner/sup_train/test_transformer_simulation.py` 中确认：

- 模型超参数
- 模型路径
- 实验列表

然后运行：

```bash
python planner/sup_train/test_transformer_simulation.py
```

目标：

- 生成 `planner/outputs/.../*.csv`

#### 第 7 步：官方评分

把 `planner/outputs/` 下对应实验目录中的 `*_result.csv` 放入 OnSite 官方评分软件，得到：

- Safety
- Efficiency
- Comfort

#### 第 8 步：泛化测试

```bash
python test_highway_env_transformer.py
python test_ngsim_transformer.py
```

目标：

- 获得闭环泛化与开环泛化结果

### 15.2 情况 B：快速复现（跳过重新制数）

如果仓库里的 `dataset/GRU_choose_closest8_properRL_*.npy` 已经可用，那么你可以直接从训练开始：

```bash
python test_dataset_jolin.py
python get_clone_learning_Transformer6_7.py
python planner/sup_train/test_transformer_simulation.py
python test_highway_env_transformer.py
python test_ngsim_transformer.py
```

这是最适合“先跑通项目”的路径。

---

## 16. 当前仓库中可以直接确认的若干事实

为了避免文档写得“像真理但其实是猜的”，这里把当前代码里能直接确认的东西单独列出来。

### 16.1 能直接确认

1. 主训练脚本是 `get_clone_learning_Transformer6_7.py`
2. 主闭环测试脚本是 `planner/sup_train/test_transformer_simulation.py`
3. 泛化测试脚本是：
   - `test_highway_env_transformer.py`
   - `test_ngsim_transformer.py`
4. 主训练数据文件名是：
   - `GRU_choose_closest8_properRL_0.npy`
   - `GRU_choose_closest8_properRL_1.npy`
   - `GRU_choose_closest8_properRL_out.npy`
5. 单条样本总维度是 64
6. 训练输入维度是 54
7. 标签维度是 10
8. 标签含义是未来 5 步的 `[acc, steer]`
9. Transformer 主结构是 6 维主车特征 + 8×6 维障碍车特征
10. OnSite 闭环输出目录在 `planner/outputs/`

### 16.2 结合代码和命名规则推断

1. 原始专家策略数据链条是：
   - OnSite 场景
   - 专家跑场景
   - 结果 CSV
   - 进一步加工成 `.npy`

2. `GenerateScenarios.py` 承担的是场景扩增而不是训练
3. `planner/main_get_multi_choose_dataset_core.py` 承担的是专家跑场景并补充 CSV 字段
4. `Get_Dataset_Closest_Proper_RL.py` 是 CSV 转 NPY 的最后一跳

这些推断与仓库结构是高度一致的，但如果你后面还保留了仓库外部脚本，则实际生产过程可能还夹杂其他中间脚本。

---

## 17. 新手最容易踩的坑与注意事项

下面这一部分故意写长一点，因为真正把项目跑起来，通常不是“算法难”，而是“工程细节坑多”。

### 17.1 第一大坑：很多脚本里写的是 Windows 路径

这是当前仓库最明显的问题之一。

例如：

- `D:\...`
- `E:\...`

这些路径在 Linux 上一定不能直接用。

所以你在运行以下脚本前，几乎都要先检查路径：

- `GenerateScenarios.py`
- `planner/main_get_multi_choose_dataset_core.py`
- `Get_Dataset_Closest_Proper_RL.py`

如果不先改：

- 要么直接报文件不存在；
- 要么生成到错误位置；
- 要么你以为跑了，其实输出到另一个你没注意的目录。

### 17.2 第二大坑：训练和测试的模型结构必须完全一致

这条非常重要。

只要下面任意一项不一致，就会出问题：

- `D_MODEL`
- `FFN_DIM`
- `num_encoder_layers`
- `num_decoder_layers`
- `dropout`
- `CAR_NUM`
- `SEQ_LENGTH`

常见后果：

1. `.pth` 加载失败
2. 虽然能加载，但推理行为异常
3. 闭环分数奇差

所以最稳妥的办法是：

- 训练完一个模型后，把它对应的超参数一起记录下来；
- 测试脚本里逐项核对。

### 17.3 第三大坑：数据特征顺序不能错

这个项目不是“你大概提一下周围车就行”，而是**特征顺序严格绑定训练语义**。

尤其要保持一致的地方包括：

- 主车 6 维顺序
- 障碍车 6 维顺序
- 取最近 8 车的规则
- 不足 8 车时的补全方式
- 左右方向统一方式

只要顺序错一位，模型就不是“性能差一点”，而是“完全学错语义”。

### 17.4 第四大坑：不要把 `his5 / Hist5` 文件混进主线

你已经明确说了要忽略这一支，这点必须坚持。

因为一旦把 `his5` 的训练脚本、测试脚本、模型权重混进来，会发生：

1. 输入维度不匹配；
2. 你以为是主线 Transformer，实际跑的是历史帧版本；
3. 文档和结果对应不上；
4. 很容易越看越乱。

所以建议你在复现阶段只盯住：

- `get_clone_learning_Transformer6_7.py`
- `planner/sup_train/test_transformer_simulation.py`

### 17.5 第五大坑：闭环分数不完全等于 loss 高低

这是行为克隆项目里很典型的问题。

训练时你看到：

- `train loss` 很低
- `val loss` 更低

不代表闭环就一定高分。

原因是：

1. 开环误差和闭环累积误差不是一回事；
2. 某些小的动作偏差，在闭环里会逐步滚雪球；
3. 模型也可能在 loss 上更好，但驾驶风格更保守或更抖；
4. OnSite 的评分是安全、效率、舒适三者综合，不只看拟合误差。

这也是为什么你在实验记录中会出现：

- 某版 loss 更低
- 但 OnSite 评分反而更差

这不是矛盾，而是闭环任务的正常现象。

### 17.6 第六大坑：脚本支持“加载已有模型”，但不是真正完整断点续训

例如：

- `get_clone_learning_Transformer6_7.py`
- `get_clone_learning_lw_mi.py`
- `LSTM_get_clone_learning.py`

都支持“如果已有权重则加载再训”。

但这里通常只恢复了：

- 模型参数

没有完整恢复：

- 优化器状态
- 调度器状态
- 已训练 epoch 计数

所以如果你中途断训再跑，表面上像“接着训”，实际上学习率轨迹常常已经变了。

这既可能是坑，也可能变成意外收益，但你必须知道它不是严格意义上的 resume。

### 17.7 第七大坑：`run_compareRNN.sh` 最后会关机

这一点非常容易忽略。

脚本最后是：

- `shutdown`

如果你只是想后台跑基线实验，但不想跑完自动关机，必须先删掉或注释掉这一行。

### 17.8 第八大坑：OnSite 输出目录会很大

`planner/outputs/` 里真实会生成非常多的 CSV。

如果你连续跑很多实验：

- 磁盘占用会很快变大；
- 目录会越来越乱；
- 容易搞不清哪个目录对应哪个实验。

建议做法：

1. 每次实验都带上明确命名
2. 运行后立刻整理评分记录
3. 保留最关键实验的输出目录
4. 不要无脑重复跑十几遍又不归档

### 17.9 第九大坑：NGSIM 是开环，不要和 OnSite 分数直接混为一谈

很多新手会说：

- “NGSIM ADE 很低，是不是说明它一定闭环很强？”

不一定。

因为：

- NGSIM 是轨迹误差评估；
- OnSite 是闭环驾驶评分；
- 两者关注重点不同。

最合理的理解方式是：

- OnSite：考“会不会自己开”
- NGSIM：考“对真实数据的动作/轨迹拟合是不是靠谱”

### 17.10 第十大坑：highway-env 的奖励也不是 OnSite 分数

同理，`highway-env` 的：

- reward
- crash rate
- avg speed

也不能直接等价于 OnSite 的安全/效率/舒适分数。

它更像一个外部泛化参考，而不是官方考试成绩。

---

## 18. 这份项目最值得学习的地方

如果从研究和工程角度看，这个项目有几个非常值得学习的点。

### 18.1 它不是盲目端到端，而是“专家先行、网络学习”

这条路线的好处是：

- 数据标签清晰
- 容易解释
- 容易做对比实验
- 更符合很多竞赛/论文型项目的节奏

### 18.2 它把规划器知识“蒸馏”给了神经网络

专家规划器原本比较复杂：

- 候选路径
- ST 图
- 速度规划
- QP/OSQP
- 控制器

而 Transformer 最终做的是：

- 直接从状态到控制序列映射

这本质上是一种知识压缩/蒸馏思路。

### 18.3 它没有只做开环，而是做了闭环和泛化

很多项目只做到：

- loss 下降
- 画一条曲线

就结束了。

而这份项目往前走了很多步：

- OnSite 闭环
- 官方评分
- highway-env
- NGSIM
- 还有 RNN/LSTM/GRU 对比

这让整个项目更像一份完整研究工程，而不是单个模型脚本。

---

## 19. 如果你是第一次接触这个项目，最推荐的阅读顺序

为了避免一上来就被很多脚本绕晕，建议按下面顺序读。

### 第一层：先建立全局印象

1. 先读本文档
2. 再看 `NoteJolin/transformer6_7迭代记录.md`

### 第二层：只看主线

1. `get_clone_learning_Transformer6_7.py`
2. `planner/sup_train/test_transformer_simulation.py`
3. `test_highway_env_transformer.py`
4. `test_ngsim_transformer.py`

### 第三层：补专家数据来源

1. `planner/main_get_multi_choose_dataset_core.py`
2. `Get_Dataset_Closest_Proper_RL.py`
3. `planner_example(谢师兄用的触须法)/Planner.py`
4. `planner_example(谢师兄用的触须法)/EM_Planner.py`

### 第四层：再看对比实验

1. `get_clone_learning_lw_mi.py`
2. `LSTM_get_clone_learning.py`
3. `vanilla_RNN_get_clone_learning.py`
4. `AR_get_clone_learning.py`
5. `run_compareRNN.sh`

这样会比“从仓库根目录随便点”高效得多。

---

## 20. 最后用一句最容易记住的话总结这个项目

这份项目的本质是：

**先用触须法专家在 OnSite 高速场景里生成高质量驾驶示范，再把示范整理成 54 维状态到未来 5 步控制序列的监督学习数据，随后用 Transformer 做行为克隆，并通过 OnSite 闭环评分、highway-env 闭环泛化和 NGSIM 开环误差三个层面验证模型能力。**

如果你把这句话真正理解了，那么这整个仓库的主线你就已经抓住了。

---

## 21. 附：最常用命令清单

### 21.1 数据与训练主线

```bash
python GenerateScenarios.py
python planner/main_get_multi_choose_dataset_core.py
python Get_Dataset_Closest_Proper_RL.py
python test_dataset_jolin.py
python get_clone_learning_Transformer6_7.py
python planner/sup_train/test_transformer_simulation.py
python test_highway_env_transformer.py
python test_ngsim_transformer.py
```

### 21.2 对比实验

```bash
bash run_compareRNN.sh
```

或单独运行：

```bash
python vanilla_RNN_get_clone_learning.py
python LSTM_get_clone_learning.py
python get_clone_learning_lw_mi.py
python AR_get_clone_learning.py
```

### 21.3 结果整理与画图

```bash
python Jolindraw/plot_comparison.py
python Jolindraw/plot_comparison_v2.py
python Jolindraw/plot_comparison_v3.py
python Jolindraw/plot_comparison_TFep200.py
```

