# BCRL 项目 Highway-env 综合测试与操作文档 v2

本文档对应的测试代码为项目根目录下的 `test_highway_env_transformer_v2.py`。

这份 v2 脚本是在早期 `test_highway_env_transformer.py` 基础上扩展出来的综合测试版本。早期版本主要回答“模型能否在 highway-env 中闭环跑通、是否碰撞、平均速度和推理耗时如何”。v2 版本进一步加入了四类场景、默认 1000 集规模、细化评分、逐集 CSV、完整 TXT 报告、每集静态多帧可视化图和总体统计图，更适合用于论文实验、模型横向对比和错误案例分析。

文档面向第一次接触 `highway-env` 的同学，因此会尽量把每一步“代码为什么这样写、结果怎么生成、文件怎么看、换模型要改哪里”讲清楚。

---

## 1. Highway-env 测试在本项目中的定位

### 1.1 它和 Onsite 测试的关系

本项目的 Transformer 行为克隆模型来自 `get_clone_learning_Transformer6_7.py` 训练，训练数据特征格式与 Onsite 场景测试高度相关。Onsite 测试脚本 `planner/sup_train/test_transformer_simulation.py` 的核心流程是：

1. 读取 Onsite 官方格式的 `.xosc` / `.xodr` 场景。
2. 从仿真环境中提取主车、目标点、周围车辆、车道边界等信息。
3. 拼接成模型训练时使用的输入特征。
4. 模型输出未来若干步动作。
5. 将动作交给 Onsite 仿真环境执行。
6. 测试完成后再用 Onsite 官方评分软件评分。

`test_highway_env_transformer_v2.py` 做的是同一类闭环控制测试，但把外部环境从 Onsite 换成了 `highway-env`：

1. 由 `highway-env` 创建高速公路交通流。
2. 脚本主动布置或调整不同类型的交通场景。
3. 从 `highway-env` 的底层车辆对象中读取位置、速度、航向角等物理量。
4. 手动转换成训练模型能够理解的 54 维特征。
5. 模型输出未来 5 步连续动作。
6. 环境执行动作，并记录安全、效率、舒适、推理时间、可视化快照等信息。

因此，Onsite 测试和 Highway-env 测试最大的相同点是：都在做“观测环境 -> 构造模型输入 -> 模型推理 -> 执行动作 -> 再观测”的闭环控制。最大的不同点是：Onsite 使用官方场景文件和官方评分器，Highway-env 使用 Gymnasium 环境、内置道路和脚本自定义评分体系。

### 1.2 Highway-env 不读取 Onsite 的 `inputs` 场景

v2 脚本不会读取 `planner/inputs`、`inputs` 或其他 Onsite 场景文件夹。

原因是：

1. Onsite 的场景文件是 OpenSCENARIO / OpenDRIVE 风格，服务于 Onsite 仿真平台。
2. `highway-env` 是 Gymnasium 风格环境，通过 `gym.make("highway-v0", config=...)` 创建道路、车辆和交通流。
3. v2 脚本的四类测试场景由 `setup_scenario` 函数在 `highway-v0` 内部构造，而不是从外部场景文件加载。

所以，运行 `test_highway_env_transformer_v2.py` 时，测试对象是模型在 `highway-env` 平台中面对不同交通布局时的闭环表现，不是 Onsite 官方测试集的替代评分。

### 1.3 v2 脚本主要解决的问题

v2 脚本相比早期版本，重点解决以下问题：

1. 不再只做单一随机高速流，而是分成跟车、合适时机变道、超车、避让四类场景。
2. 不再只输出安全率、奖励、速度、推理耗时，而是增加 Safety、Efficiency、Comfort、Total 四类分数。
3. 不再只保存总体图，而是给每一集都生成多帧静态可视化图，方便追踪车辆运动过程。
4. 不再只生成简短 `.txt`，而是同时生成逐集 `.csv`、完整 `.txt`、总体统计 `.png` 和每集场景 `.png`。
5. 默认测试规模为每类场景 250 集，共 1000 集，更适合作为稳定性和泛化能力评估。

---

## 2. 运行前准备与基本命令

### 2.1 入口文件

当前测试入口是：

```bash
python test_highway_env_transformer_v2.py
```

建议在需要实时观察终端输出时使用：

```bash
python -u test_highway_env_transformer_v2.py
```

`-u` 表示不缓冲输出。长时间测试时，如果不用 `-u`，终端日志有时会延迟显示，让人误以为程序卡住。

### 2.2 三种常用运行方式

快速验证模式：

```bash
python -u test_highway_env_transformer_v2.py --quick
```

该模式每类场景只跑 4 集，共 16 集。适合检查模型路径是否正确、环境是否能创建、报告和可视化是否能正常生成。

自定义每类场景集数：

```bash
python -u test_highway_env_transformer_v2.py -n 50
```

该命令表示四类场景各跑 50 集，总计 200 集。适合中等规模调试。

完整默认测试：

```bash
python -u test_highway_env_transformer_v2.py
```

默认每类场景 250 集，四类场景共 1000 集。由于每一集都会绘制可视化图，运行时间和磁盘占用都会明显增加，建议确认脚本已经通过 `--quick` 验证后再跑完整测试。

### 2.3 运行后的输出位置

每次运行都会在项目根目录下新建一个时间戳目录：

```text
highway_env_results_v2/{月日_时分秒}/
```

例如：

```text
highway_env_results_v2/0430_153812/
```

该目录是一次完整测试的结果包。不同时间运行会生成不同目录，互不覆盖。

---

## 3. v2 测试脚本总体结构

`test_highway_env_transformer_v2.py` 大致可以分成以下模块：

1. 全局配置区：定义模型路径、网络超参数、环境参数、场景类型、可视化帧数。
2. Transformer 模型定义：重新定义与训练时一致的网络结构，用于加载 `.pth` 权重。
3. 车道映射与特征提取：把 `highway-env` 的物理状态转换成模型训练时使用的 54 维输入。
4. TTC 计算：计算主车与前车的 Time-To-Collision，用于安全评分。
5. 评分系统：计算 Safety、Efficiency、Comfort、Total、目标达成等指标。
6. 场景构造：生成跟车、变道、超车、避让四类测试场景。
7. 单集运行与快照记录：执行闭环控制，并保存每集用于画图的关键帧。
8. 可视化：把车辆位置、车道线和危险车辆画成每集 PNG。
9. 报告生成：输出 CSV、TXT、总体统计图。
10. 主函数：解析命令行参数、加载模型、跑全部场景、标记代表性集、生成结果。

---

## 4. 全局配置区详细说明

全局配置位于 `test_highway_env_transformer_v2.py` 开头，主要在第 45 行到第 82 行附近。

### 4.1 设备与项目路径

`DEVICE` 自动判断使用 GPU 还是 CPU：

```text
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

如果服务器有 CUDA，模型会放到 GPU 上推理；否则使用 CPU。

`BASE_DIR` 是当前脚本所在目录，也就是项目根目录。后续模型路径和结果路径都基于它拼接。

### 4.2 模型路径

`MODEL_PATH` 位于第 48 行到第 52 行附近，当前指向：

```text
Transformer_checkpoints/Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth
```

这是本次 Highway-env 测试要加载的 Transformer `.pth` 权重。换模型时最常改的就是这里。

### 4.3 网络超参数

当前脚本中的网络超参数包括：

1. `D_MODEL = 256`：Transformer token embedding 维度。
2. `FFN_DIM = 4 * D_MODEL`：前馈网络维度，当前为 1024。
3. `NUM_ENC_LAYERS = 3`：Encoder 层数。
4. `NUM_DEC_LAYERS = 3`：Decoder 层数。
5. `OUTPUT_DIM = 2`：每一步输出两个控制量，即加速度和转角。
6. `SEQ_LENGTH = 5`：一次预测未来 5 步动作。
7. `CAR_NUM = 8`：模型输入最多使用 8 辆周围车。

这些参数必须与训练脚本保持一致。否则 `model.load_state_dict(...)` 会因为层数、权重形状或参数名不匹配而报错。

### 4.4 环境参数

环境参数包括：

1. `LANES_COUNT = 3`：三车道高速路。
2. `LANE_WIDTH = 4.0`：车道宽度 4 米。
3. `VEHICLES_COUNT = 20`：环境中基础车辆数量。
4. `SIM_FREQ = 10`：仿真频率 10 Hz。
5. `POLICY_FREQ = 10`：策略执行频率 10 Hz。
6. `EPISODE_DURATION = 25`：每集最长 25 秒。
7. `STEPS_PER_INFER = 5`：模型每次推理输出 5 步动作，并连续执行这 5 步。
8. `GOAL_LOOKAHEAD = 500.0`：虚拟目标点在主车前方 500 米。
9. `TARGET_SPEED = 25.0`：评分参考速度，约等于 90 km/h。

这里需要特别注意：v2 脚本每集时长是 25 秒，不是早期脚本中的 60 秒。完整测试 1000 集时，如果仍设置 60 秒并给每集都画图，耗时会大幅上升。

### 4.5 场景与测试规模

场景类型定义为：

```text
SCENARIO_TYPES = ["following", "lane_change", "overtaking", "avoidance"]
```

中文含义：

1. `following`：跟车。
2. `lane_change`：合适时机变道。
3. `overtaking`：超车。
4. `avoidance`：避让。

默认每类场景集数：

```text
EPISODES_PER_SCENARIO = 250
```

所以默认总测试集数为：

```text
250 * 4 = 1000
```

`VIS_PANELS = 4` 表示每一集的可视化图中默认展示 4 个关键时间帧。不是每一步都画一张图，而是在一集内按时间均匀采样 4 个快照，然后组合成一张 PNG。

---

## 5. Transformer 模型定义逻辑

模型类是 `TransformerTrajectoryPredictor`，位于第 86 行到第 136 行附近。

测试脚本必须重新定义模型类，因为 `.pth` 文件通常只保存权重参数，不保存完整 Python 类定义。加载权重前，代码需要先实例化一个结构完全一致的模型对象。

### 5.1 输入格式

模型输入是一个 54 维向量：

1. 前 6 维：主车与目标相关特征。
2. 后 48 维：8 辆周围车，每辆 6 维。

在 `forward` 中：

```text
main_target_feat = x[:, 0:6]
vehicle_feats = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)
```

这说明模型强依赖输入顺序。如果 `extract_features` 构造的特征顺序与训练时不一致，模型就会把速度、距离、航向角、车道边界等含义理解错。

### 5.2 网络内部流程

模型内部主要流程：

1. 主车和目标特征通过 `embedding_main_target` 映射到 `d_model` 维。
2. 周围车特征通过 `embedding_vehicle` 映射到 `d_model` 维。
3. 周围车 token 和主车 token 拼接成 token 序列。
4. 经过 Transformer Encoder 建模周围车与主车之间的交互关系。
5. 使用 `future_queries` 作为 Decoder 的可学习查询，生成未来 5 个时间步的隐向量。
6. 经过 `head` 输出 `(batch, 5, 2)` 的动作序列。
7. 最后一层 `Tanh` 将动作限制在 `[-1, 1]` 左右。

### 5.3 输出格式

模型输出形状为：

```text
(1, 5, 2)
```

含义是：

1. `1`：batch size。
2. `5`：未来 5 个控制步。
3. `2`：每一步包含两个动作量，分别为加速度控制和转角控制。

在单集运行中，脚本会把这 5 步动作依次送给环境执行。

---

## 6. 车道映射与 54 维特征提取

### 6.1 为什么需要车道映射

`highway-env` 和训练数据中的车道约定并不完全一致。脚本不能直接把 `highway-env` 的 y 坐标原样传入模型，否则模型可能会把左右边界关系理解反。

`build_map_info` 位于第 142 行到第 151 行附近。它把 `highway-env` 的三车道结构转换成 Onsite 风格的 `map_info`：

1. highway lane 0：中心线大约是 `y = 0`。
2. highway lane 1：中心线大约是 `y = 4`。
3. highway lane 2：中心线大约是 `y = 8`。

脚本构造 `map_info` 时反向映射：

1. `map_info[0]` 对应 highway lane 2。
2. `map_info[1]` 对应 highway lane 1。
3. `map_info[2]` 对应 highway lane 0。

这样做的目标是尽量保持车道边界特征与 Onsite 训练数据中的语义一致。

### 6.2 主车车道边界

`_ego_lane_bounds` 用于判断主车当前所在车道，并返回主车到该车道边界的相对距离。

返回值是：

```text
(left_bound - ego_y, right_bound - ego_y)
```

这些值最终会进入模型输入，用于帮助模型理解主车距离道路边界或车道边界还有多远。

### 6.3 周围车车道边界

`_veh_lane_bounds` 用于判断周围车所在车道，并返回该车道上下边界相对主车 y 坐标的距离。

它服务于每辆周围车的第 5、6 个特征。模型训练时不仅知道周围车的相对位置，也知道周围车所在车道边界，这对变道、超车和避让判断有帮助。

### 6.4 `extract_features` 的输入输出

`extract_features(env, mi, goal_x, goal_y)` 位于第 172 行到第 209 行附近。

输入：

1. `env`：当前 `highway-env` 环境。
2. `mi`：由 `build_map_info` 构造的车道边界信息。
3. `goal_x`：当前场景设定的虚拟目标点 x 坐标。
4. `goal_y`：当前场景设定的虚拟目标点 y 坐标。

输出：

```text
长度为 54 的 Python list
```

随后会被转换成 PyTorch Tensor：

```text
torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
```

### 6.5 主车 6 维特征

脚本首先读取：

1. `ego.position`：主车位置。
2. `ego.speed`：主车速度。
3. `ego.heading`：主车航向角。
4. `ego.LENGTH`：主车长度。

主车 6 维特征为：

1. `ev`：主车速度。
2. `eyaw`：归一化到 `[-pi, pi]` 的主车航向角。
3. `goal_x - ex`：主车到虚拟目标点的纵向距离。
4. `goal_y - ey`：主车到虚拟目标点的横向距离。
5. `mi[0]["left_bound"] - ey`：主车到整体上边界的相对距离。
6. `mi[2]["right_bound"] - ey`：主车到整体下边界的相对距离。

这里的目标点不是 `highway-env` 官方定义的终点，而是脚本为每类场景人工设置的目标。比如变道场景会把 `goal_y` 设置到目标车道中心，跟车、超车、避让场景通常把目标保持在当前车道中心。

### 6.6 周围车筛选逻辑

脚本遍历：

```text
env.unwrapped.road.vehicles[1:]
```

第 0 辆车是主车，所以从第 1 辆开始都是周围车。

筛选过程如下：

1. 只考虑横向距离主车不超过 6 米的车辆，过滤掉太远车道的车辆。
2. 计算周围车与主车的纵向净距离，而不是中心点距离。
3. 只保留综合距离小于 200 米的候选车。
4. 按照距离从近到远排序。
5. 最多取最近的 8 辆车。

“纵向净距离”的计算会考虑车长：

1. 如果周围车在前方，使用 `dx - 半车长和`。
2. 如果周围车在后方，使用 `dx + 半车长和`。
3. 如果两车纵向投影已经重叠，记为 `0.0`。

这样比简单中心点距离更接近真实驾驶中的车头到车尾距离。

### 6.7 每辆周围车的 6 维特征

每辆被选中的周围车会追加 6 个特征：

1. `ld`：相对主车的纵向净距离。
2. `vy - ey`：相对主车的横向距离。
3. `v.speed - ev`：相对速度。
4. `vyaw`：周围车航向角。
5. `lb`：周围车所在车道左边界相对主车的距离。
6. `rb`：周围车所在车道右边界相对主车的距离。

8 辆车一共是：

```text
8 * 6 = 48 维
```

加上主车 6 维，总输入正好是 54 维。

### 6.8 Padding 补齐逻辑

如果候选周围车不足 8 辆，脚本会使用以下默认值补齐：

```text
[200.0, 0.0, 0.0, 0.0, elb, erb]
```

含义是：

1. 这辆“虚拟车”距离主车很远，约 200 米。
2. 横向距离为 0。
3. 相对速度为 0。
4. 航向角为 0。
5. 车道边界使用主车所在车道边界。

Padding 的目的不是表示真实存在一辆车，而是保证输入维度固定为 54，避免模型维度不匹配。

---

## 7. TTC 与评分系统

评分逻辑位于第 215 行到第 304 行附近，主要由 `_min_ttc` 和 `compute_scores` 完成。

### 7.1 TTC 是什么

TTC 是 Time-To-Collision，即“如果两车保持当前相对速度，大约多久后会碰撞”。

`_min_ttc(ego, vehicles)` 只关注主车前方、同车道附近的车辆：

1. 如果车在主车后方，不计算。
2. 如果横向距离超过 3 米，认为不是同一危险通道，不计算。
3. 如果间距已经小于等于 0，TTC 记为 0。
4. 如果主车没有比前车更快，不会追上，TTC 视为无穷大。
5. 否则用净间距除以接近速度，得到 TTC。

TTC 越小，说明追尾风险越高。

### 7.2 总分结构

`compute_scores` 返回：

```text
Safety(50) + Efficiency(30) + Comfort(20) = Total(100)
```

也就是说总分满分 100，由三部分组成：

1. 安全性 Safety：满分 50。
2. 效率性 Efficiency：满分 30。
3. 舒适性 Comfort：满分 20。

这套评分不是 `highway-env` 官方原生总分，而是 v2 脚本为了更接近论文式指标拆分而自定义的综合评分。

### 7.3 Safety 安全性评分

Safety 满分 50，由三部分组成：

1. 碰撞分 `col_pts`：无碰撞得 35 分，碰撞得 0 分。
2. TTC 分 `ttc_pts`：最高 10 分，根据平均 TTC 分段给分。
3. 车道边界分 `lane_pts`：最高 5 分，根据越界比例扣分。

TTC 分段逻辑：

1. 平均 TTC 大于 3 秒，得 10 分。
2. 平均 TTC 大于 2 秒，得 7 分。
3. 平均 TTC 大于 1 秒，得 4 分。
4. 否则得 1 分。

车道边界分会检查主车 y 坐标是否跑出道路范围。越界比例越高，分数越低。

### 7.4 Efficiency 效率性评分

Efficiency 满分 30，由两部分组成：

1. 速度分 `speed_pts`：最高 20 分。
2. 距离分 `dist_pts`：最高 10 分。

速度分用平均速度与 `TARGET_SPEED = 25.0 m/s` 比较：

```text
speed_pts = min(avg_speed / TARGET_SPEED, 1.0) * 20
```

如果平均速度达到或超过 25 m/s，就能拿满速度分。低于该速度则按比例给分。

距离分比较本集实际前进距离与期望距离：

```text
expected_distance = TARGET_SPEED * duration
```

如果模型在规定时间内前进距离足够接近目标速度对应的距离，就能获得更高效率分。

### 7.5 Comfort 舒适性评分

Comfort 满分 20，由两部分组成：

1. Jerk 分 `jk_pts`：最高 10 分。
2. 转角变化率分 `st_pts`：最高 10 分。

Jerk 通过相邻加速度动作差除以时间步长得到：

```text
jerk = abs(acc[i + 1] - acc[i]) / dt
```

转角变化率通过相邻转角动作差除以时间步长得到：

```text
steer_rate = abs(steer[i + 1] - steer[i]) / dt
```

直观理解：

1. 加速度变化越平滑，乘坐舒适性越好。
2. 方向盘变化越平滑，车辆横向控制越稳定。
3. 如果动作突然大幅跳变，即使没有撞车，舒适性分也会降低。

### 7.6 目标达成率

`goal_achieved` 的判断逻辑是：

```text
not crashed and total_steps >= EPISODE_DURATION * POLICY_FREQ * 0.95
```

也就是说，一集被认为“目标达成”需要同时满足：

1. 没有碰撞。
2. 实际运行步数达到本集理论最大步数的 95% 以上。

在当前配置下，每集最长 25 秒、10 Hz，所以理论最大步数是 250 步。达到约 238 步以上且不碰撞，才算目标达成。

### 7.7 车道变换次数

`n_lane_changes` 通过主车 y 坐标估算所在车道：

```text
lane = int(round(y / LANE_WIDTH))
```

然后统计相邻时刻车道编号发生变化的次数。这个指标可以帮助判断模型是否在变道、超车场景中发生了横向动作，也可以发现模型是否存在频繁来回摆动的问题。

---

## 8. 四类场景构造逻辑

场景构造由 `_make_env` 和 `setup_scenario` 完成，位于第 310 行到第 403 行附近。

### 8.1 基础环境 `_make_env`

`_make_env` 使用：

```text
gym.make("highway-v0", config={...})
```

核心配置包括：

1. `observation.type = "Kinematics"`：使用车辆运动学观测。
2. `features = ["x", "y", "vx", "vy", "heading"]`：环境可提供位置、速度、航向角等物理量。
3. `absolute = True`：使用绝对坐标。
4. `normalize = False`：不使用环境内置归一化。
5. `action.type = "ContinuousAction"`：使用连续动作。
6. `acceleration_range = (-6, 6)`：加速度范围。
7. `steering_range = (-0.15, 0.15)`：转角范围。
8. `other_vehicles_type = "highway_env.vehicle.behavior.IDMVehicle"`：周围车由 IDM 行为模型控制。
9. `offscreen_rendering = True`：适配无显示器服务器。

虽然环境本身有 observation 配置，但当前模型并不直接使用 `obs`。脚本通过 `env.unwrapped` 访问底层车辆对象，然后手动拼接 54 维训练格式特征。

### 8.2 随机种子逻辑

每一集都会生成一个 seed：

```text
seed = int(rng.integers(100000))
env.reset(seed=seed)
```

主函数中使用：

```text
rng = np.random.default_rng(42)
```

这意味着在脚本和依赖版本不变的情况下，测试具有一定可复现性。CSV 中会保存每集 seed，便于后续定位某个具体场景。

### 8.3 跟车场景 `following`

目标：测试模型在同车道前方存在慢车时，能否保持安全距离并稳定行驶。

构造方式：

1. 先寻找同车道、前方 20 到 70 米范围内的车辆。
2. 如果找到，将其速度设置为 16 到 21 m/s。
3. 如果没有找到合适车辆，脚本会尝试额外创建一辆 `IDMVehicle` 放在主车前方 35 到 55 米处。
4. 虚拟目标点设在当前车道前方 500 米。

这一场景更关注安全跟车、TTC 和速度控制。

### 8.4 合适时机变道场景 `lane_change`

目标：测试模型能否根据目标车道位置产生合理横向控制。

构造方式：

1. 获取主车当前车道。
2. 从其他车道中随机选择一个目标车道。
3. 将虚拟目标点 `goal_y` 设置为目标车道中心线。
4. `goal_x` 仍设置在主车前方 500 米。

这一场景不一定强制主车必须完成一次变道，但目标点会引导模型产生向目标车道行驶的意图。最终是否变道、是否安全、是否舒适，由闭环结果和评分体现。

### 8.5 超车场景 `overtaking`

目标：测试模型面对前方较慢车辆时，是否能够通过速度或横向动作完成更高效率通行。

构造方式：

1. 在主车当前车道前方 30 到 50 米处放置一辆慢车。
2. 慢车速度设置为 10 到 15 m/s。
3. 虚拟目标点仍设置在当前车道前方。

该场景会制造明显的前方慢车压力。理想情况下，模型需要避免追尾，并在可能时表现出超车或合理减速行为。

### 8.6 避让场景 `avoidance`

目标：测试模型面对近距离前方障碍或慢车时的紧急避让能力。

构造方式：

1. 优先寻找主车同车道前方车辆。
2. 将其位置移动到主车前方 15 到 25 米。
3. 将其速度设置为低于主车速度的值。
4. 如果没有合适车辆，则额外创建一辆慢车放在主车前方。
5. 虚拟目标点保持当前车道前方。

该场景通常比跟车更困难，因为前车距离更近、速度差更明显，更容易暴露模型刹车不足、反应迟缓或转向过猛的问题。

---

## 9. 单集闭环运行逻辑

单集运行函数是 `run_episode(env, model, mi, stype, ep_idx, rng)`，位于第 441 行到第 503 行附近。

### 9.1 单集初始化

每集开始时：

1. 调用 `setup_scenario` 创建指定场景。
2. 得到虚拟目标点 `goal_x`、`goal_y` 和本集 seed。
3. 设置最大步数 `max_steps = EPISODE_DURATION * POLICY_FREQ`。
4. 根据 `VIS_PANELS` 计算需要保存快照的时间步。

当前默认：

```text
EPISODE_DURATION = 25
POLICY_FREQ = 10
max_steps = 250
VIS_PANELS = 4
```

所以一集最多执行 250 步，并大约保存第 0、83、166、249 步附近的快照。如果提前碰撞，还会额外保存结束时的快照。

### 9.2 单集数据字典

函数内部维护一个字典 `d`，用于保存本集全部关键信息：

1. `scenario_type`：场景英文类型。
2. `episode_idx`：该场景内第几集。
3. `seed`：本集随机种子。
4. `goal_x`、`goal_y`：虚拟目标点。
5. `crashed`：是否碰撞。
6. `total_steps`：实际执行步数。
7. `total_reward`：环境累计奖励。
8. `inference_times`：每次模型前向推理耗时。
9. `ego_x`、`ego_y`、`ego_speed`、`ego_heading`：主车轨迹和状态。
10. `actions`：模型输出并执行的动作。
11. `rewards`：每一步环境奖励。
12. `min_ttc`：每一步最小 TTC。
13. `vis_frames`：用于可视化绘图的关键帧。

后续生成 CSV、TXT 和 PNG 都依赖这个字典。

### 9.3 闭环控制循环

单集主循环做以下事情：

1. 如果当前 step 是可视化采样点，则调用 `_snapshot` 保存当前车辆布局。
2. 调用 `extract_features` 构造 54 维输入。
3. 转为 Tensor 并放到 `DEVICE`。
4. 使用 `time.perf_counter()` 开始计时。
5. 在 `torch.no_grad()` 下执行模型推理。
6. 记录本次推理耗时，单位毫秒。
7. 依次取出模型预测的 5 步动作。
8. 对每一步动作调用 `env.step(act)`。
9. 记录主车位置、速度、航向角、动作、奖励和 TTC。
10. 如果环境返回 `terminated` 或 `truncated`，结束本集。

这就是典型的滚动时域控制逻辑。模型一次预测未来 5 步，但不会一次性把整集都预测完，而是每执行 5 步重新观察环境，再预测新的 5 步。

### 9.4 碰撞判断

本集结束后，代码通过：

```text
d["crashed"] = bool(info.get("crashed", terminated)) if terminated else False
```

判断是否碰撞。

一般来说，`info["crashed"]` 是最直接的碰撞标记。如果环境因为 terminated 结束但 info 中没有明确字段，则用 terminated 作为兜底。

### 9.5 单集评分

最后调用：

```text
d["scores"] = compute_scores(d)
```

把本集记录的轨迹、动作、TTC、速度、碰撞状态转换成 Safety、Efficiency、Comfort 和 Total 等指标。

---

## 10. 更换测试模型时需要修改哪里

如果只是换另一个同结构 Transformer 权重，修改很少。如果换了网络结构、输入特征或输出形式，就需要改更多地方。下面按风险从低到高列出。

### 10.1 模型权重路径：必须检查

代码位置：

```text
test_highway_env_transformer_v2.py 第 48-52 行附近
变量：MODEL_PATH
```

修改内容：

1. 如果新模型仍在 `Transformer_checkpoints` 目录下，只需要替换文件名。
2. 如果模型在其他目录，可以把 `MODEL_PATH` 改成绝对路径。
3. 确保文件后缀是实际保存的 `.pth` 或兼容 PyTorch 加载的权重文件。

示例：

```python
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "your_new_model.pth",
)
```

注意事项：

1. 路径错了会在第 810 行附近触发“找不到模型文件”并退出。
2. 路径对但结构不匹配，会在 `model.load_state_dict(...)` 时报错。

### 10.2 Transformer 超参数：同结构权重必须一致

代码位置：

```text
test_highway_env_transformer_v2.py 第 54-60 行附近
```

需要核对：

1. `D_MODEL`：训练时的 Transformer 隐层维度。
2. `FFN_DIM`：训练时的 FFN 维度。
3. `NUM_ENC_LAYERS`：Encoder 层数。
4. `NUM_DEC_LAYERS`：Decoder 层数。
5. `OUTPUT_DIM`：动作维度。
6. `SEQ_LENGTH`：预测步长。
7. `CAR_NUM`：输入周围车数量。

这些属于“模型结构超参数”。只要其中一个和训练时不同，就可能导致权重形状不一致。例如训练时 `D_MODEL=256`，测试时写成 `128`，所有线性层和 Transformer 层都会 shape mismatch。

### 10.3 模型实例化参数：容易漏改

代码位置：

```text
test_highway_env_transformer_v2.py 第 801-808 行附近
```

主函数中实际创建模型：

```text
TransformerTrajectoryPredictor(
    d_model=D_MODEL,
    output_dim=OUTPUT_DIM,
    seq_length=SEQ_LENGTH,
    car_num=CAR_NUM,
    nhead=8,
    num_encoder_layers=NUM_ENC_LAYERS,
    num_decoder_layers=NUM_DEC_LAYERS,
    dim_feedforward=FFN_DIM,
    dropout=0.0,
)
```

如果训练时 `nhead` 不是 8，或者测试时需要保留特殊 dropout 设置，也要在这里同步。通常测试时 `dropout=0.0` 是合理的，因为 `model.eval()` 会关闭 Dropout，但为了结构参数完全一致，最好仍核对训练脚本。

### 10.4 模型类定义：结构变化时必须替换

代码位置：

```text
test_highway_env_transformer_v2.py 第 86-136 行附近
类：TransformerTrajectoryPredictor
```

如果新模型还是同一个 Transformer 类，只换权重，不需要改这里。

如果出现以下情况，必须修改或替换模型类：

1. 从 Transformer 换成 GRU、LSTM、RNN。
2. Transformer 的 embedding 层、Encoder/Decoder、future query 或 head 结构改变。
3. 输出不再经过 `Tanh`。
4. 模型输入不再是 54 维扁平向量。
5. 模型输出不再是 `(batch, 5, 2)`。

替换原则：

1. 测试脚本中的模型类必须与训练时保存权重的模型类完全对应。
2. `forward` 的输入输出接口必须能被后面的 `run_episode` 使用。
3. 如果输出格式变了，`run_episode` 中取动作的逻辑也要改。

### 10.5 特征提取逻辑：输入格式变化时必须改

代码位置：

```text
test_highway_env_transformer_v2.py 第 172-209 行附近
函数：extract_features
```

如果新模型仍使用 54 维输入，且顺序保持一致，不需要改。

如果训练数据的输入格式发生以下变化，就必须同步修改：

1. 主车特征数量不再是 6。
2. 周围车数量不再是 8。
3. 每辆周围车特征不再是 6。
4. 特征顺序改变。
5. 新增了历史帧、轨迹点、道路曲率、车道中心线等特征。
6. 输入做了归一化或标准化。

特别注意：如果训练时对输入做过均值方差标准化，测试脚本也必须使用同样的标准化参数。当前 v2 脚本没有显式加载 normalization 参数，它假设训练模型接受的是当前这种物理量尺度的 54 维输入。

### 10.6 输出动作含义：动作定义变化时必须改

代码位置：

```text
test_highway_env_transformer_v2.py 第 478-489 行附近
```

当前逻辑认为模型输出的每一步动作是：

```text
[acceleration_action, steering_action]
```

并且可以直接传给 `highway-env` 的 `ContinuousAction`。

如果新模型输出的是：

1. 真实物理加速度和真实转角。
2. 归一化后的动作。
3. 目标速度和目标车道。
4. 轨迹点而不是控制量。

就不能直接复用当前 `env.step(act)`。需要添加动作解码或控制器，把模型输出转换成环境接受的连续动作。

### 10.7 环境与测试规模：实验设计变化时再改

代码位置：

```text
test_highway_env_transformer_v2.py 第 62-82 行附近
```

可以按实验需要修改：

1. `VEHICLES_COUNT`：车辆越多，交通越密，测试越难。
2. `EPISODE_DURATION`：单集越长，越能暴露长期稳定性问题，但耗时更大。
3. `EPISODES_PER_SCENARIO`：每类场景集数，影响统计可信度。
4. `TARGET_SPEED`：效率评分参考速度。
5. `VIS_PANELS`：每集可视化快照数量。

这类参数属于“测试协议参数”，不是模型结构参数。修改它们不会导致权重加载失败，但会改变测试难度和分数可比性。做论文对比时，必须保证所有模型使用同一组测试协议参数。

---

## 11. 运行结果与文件生成详解

### 11.1 输出目录结构

一次完整运行会生成类似目录：

```text
highway_env_results_v2/
└── 0430_153812/
    ├── episode_details_0430_153812.csv
    ├── test_report_0430_153812.txt
    ├── score_distribution_0430_153812.png
    ├── inference_time_0430_153812.png
    ├── scenario_comparison_0430_153812.png
    └── episodes/
        ├── following/
        ├── lane_change/
        ├── overtaking/
        └── avoidance/
```

其中时间戳 `0430_153812` 由运行时刻生成。每次测试都会生成一个新的结果目录。

### 11.2 终端输出

运行过程中，终端会先打印：

1. 当前设备。
2. 模型路径。
3. 四类场景名称。
4. 每场景集数和总集数。
5. 每集时长和步数。
6. 输出目录。

然后每跑完一集，会打印一行摘要：

```text
[跟车] EP    1/250 (1/1000) | 安全 | T= 86.3 S= 50.0 E= 24.1 C= 12.2 | spd=22.8 | infer=1.35ms
```

这行信息的含义：

1. `[跟车]`：当前场景类型。
2. `EP 1/250`：当前场景内第 1 集，共 250 集。
3. `(1/1000)`：全局第 1 集，共 1000 集。
4. `安全` 或 `碰撞`：本集是否发生碰撞。
5. `T`：总分 Total。
6. `S`：Safety 安全分。
7. `E`：Efficiency 效率分。
8. `C`：Comfort 舒适分。
9. `spd`：本集平均速度。
10. `infer`：本集平均模型推理耗时。

全部仿真完成后，会打印最终汇总，并给出 CSV 和 TXT 路径。

### 11.3 CSV 文件：逐集结构化结果

文件名格式：

```text
episode_details_{timestamp}.csv
```

生成函数：

```text
generate_csv(results, path)
```

CSV 是最适合后续二次分析的文件。它每一行对应一个 Episode，每一列对应一个指标。

主要字段说明：

1. `scenario_type`：英文场景类型，如 `following`。
2. `scenario_cn`：中文场景名，如 `跟车`。
3. `episode_idx`：该场景内第几集，从 0 开始。
4. `seed`：本集随机种子，用于定位场景。
5. `crashed`：是否碰撞，1 表示碰撞，0 表示未碰撞。
6. `goal_achieved`：是否目标达成，1 表示达成，0 表示未达成。
7. `total_steps`：实际执行步数。
8. `duration_s`：实际持续秒数。
9. `total_reward`：`highway-env` 环境累计奖励。
10. `avg_speed`：本集平均速度。
11. `distance`：本集纵向前进距离。
12. `n_lane_changes`：估算车道变化次数。
13. `mean_ttc`：平均 TTC。
14. `min_ttc_val`：本集最小 TTC。
15. `avg_jerk`：平均加速度变化率。
16. `avg_steer_rate`：平均转角变化率。
17. `safety`：安全分，满分 50。
18. `efficiency`：效率分，满分 30。
19. `comfort`：舒适分，满分 20。
20. `total`：总分，满分 100。
21. `avg_inference_ms`：本集平均推理耗时。
22. `representative`：代表性标记，例如最佳、最难、中位、碰撞。

如果后续要用 Excel、Python、Origin 或其他工具画统计图，优先读取 CSV。

### 11.4 TXT 文件：完整可读测试报告

文件名格式：

```text
test_report_{timestamp}.txt
```

生成函数：

```text
generate_txt_report(results, path)
```

TXT 报告更适合人工阅读，主要包含：

1. 报告标题、生成时间、模型路径、设备和测试设置。
2. 总测试集数、安全完成比例、碰撞次数、目标达成率。
3. 平均总分、平均安全分、平均效率分、平均舒适分、平均速度。
4. 推理耗时均值、P50、P95、P99。
5. 分场景汇总表。
6. 代表性集标记。
7. 逐集详情。

分场景汇总可以快速比较四类场景中模型的强弱。例如：

1. 如果避让场景碰撞率明显更高，说明模型面对近距离慢车或障碍时处理不足。
2. 如果变道场景舒适分低，说明横向动作可能抖动或转角变化过快。
3. 如果超车场景效率分低，说明模型可能过于保守，跟在慢车后不敢加速或变道。

逐集详情适合定位具体异常 Episode，再去 `episodes/{scenario_type}/` 中打开对应图片看轨迹。

### 11.5 总体评分分布图

文件名格式：

```text
score_distribution_{timestamp}.png
```

生成函数：

```text
generate_summary_plots(results, save_dir, ts)
```

这张图包含 2x2 四个子图：

1. Safety 分布。
2. Efficiency 分布。
3. Comfort 分布。
4. Total 分布。

每个子图会按场景类型绘制直方图。它可以帮助判断：

1. 分数是否集中在高分区域。
2. 是否存在明显低分长尾。
3. 哪类场景整体更困难。
4. 是否有某类指标系统性偏低。

### 11.6 推理耗时分布图

文件名格式：

```text
inference_time_{timestamp}.png
```

图中是所有模型推理耗时的直方图，并用红色虚线标出平均耗时。

关注点：

1. 平均耗时是否足够低。
2. 是否存在少量特别慢的推理长尾。
3. P95、P99 是否仍满足实时控制需求。

注意：这里统计的是模型前向推理时间，不包括环境 step、绘图、CSV 写入等耗时。

### 11.7 场景对比柱状图

文件名格式：

```text
scenario_comparison_{timestamp}.png
```

该图比较四类场景下 Safety、Efficiency、Comfort 的平均分。

适合回答：

1. 模型在哪类场景最安全。
2. 模型在哪类场景效率最低。
3. 模型在哪类场景舒适性最差。
4. 不同模型之间是否存在明显场景偏科。

### 11.8 每集可视化图片

每集图片保存在：

```text
highway_env_results_v2/{timestamp}/episodes/{scenario_type}/
```

例如：

```text
highway_env_results_v2/0430_153812/episodes/following/
highway_env_results_v2/0430_153812/episodes/lane_change/
highway_env_results_v2/0430_153812/episodes/overtaking/
highway_env_results_v2/0430_153812/episodes/avoidance/
```

文件名格式大致为：

```text
ep_0001_score86.png
ep_0023_score91_最佳.png
ep_0047_score42_碰撞.png
```

含义：

1. `ep_0001`：该场景内第 1 集。
2. `score86`：本集总分约为 86。
3. `_最佳`、`_最难`、`_中位`、`_碰撞`：代表性标记。

每张图片由多个横向道路快照组成，默认 4 个面板。每个面板表示该 Episode 中某个时间步的交通状态。

---

## 12. 可视化过程详细说明

v2 脚本的可视化分为两类：逐集场景可视化和总体统计可视化。

### 12.1 快照记录 `_snapshot`

`_snapshot(env, step)` 位于第 409 行到第 427 行附近。

它在指定 step 记录当前所有车辆的信息：

1. 车辆 x 坐标。
2. 车辆 y 坐标。
3. 航向角。
4. 速度。
5. 长度。
6. 宽度。
7. 颜色。

颜色含义：

1. 主车为绿色。
2. 普通周围车为蓝色。
3. 如果某辆前方车与主车横向接近，并且快速 TTC 小于 2 秒，则标为红色。

红色车辆通常表示当前帧中更值得关注的潜在危险车。

### 12.2 快照采样时间点

在 `run_episode` 中，代码根据 `VIS_PANELS` 生成快照采样点：

```text
for k in range(VIS_PANELS):
    vis_indices.add(int(k * (max_steps - 1) / (VIS_PANELS - 1)))
```

当前 `VIS_PANELS = 4`，因此会在整集时间范围内均匀选取 4 个关键帧。

如果一集提前碰撞，且碰撞发生时不在原本采样点上，脚本还会额外保存结束帧：

```text
if terminated and step not in vis_indices:
    d["vis_frames"].append(_snapshot(env, step))
```

所以碰撞集可能有 5 个面板，方便查看最后撞车瞬间。

### 12.3 车辆外形绘制

`_vehicle_corners` 位于第 509 行到第 517 行附近。

它根据车辆中心点、航向角、长度和宽度计算四个角点。绘图时不是画一个简单点，而是画出车身矩形，这样可以更直观看到车辆姿态、间距和是否接近碰撞。

### 12.4 每集图片绘制 `draw_episode`

`draw_episode(ep, save_path, cn_name)` 位于第 520 行到第 567 行附近。

绘制流程：

1. 根据本集 `vis_frames` 数量创建多行子图。
2. 对每个快照面板画车道线。
3. 对每辆车计算矩形角点。
4. 用 `matplotlib.patches.Polygon` 把车辆画成多边形。
5. 以主车 x 坐标为中心设置视野范围：前方约 260 米，后方约 60 米。
6. y 方向显示完整三车道范围。
7. 在总标题中写入场景名、EP 编号、碰撞/安全状态、Total/Safety/Efficiency/Comfort 分数和代表性标记。
8. 保存为 PNG。

这类图片适合人工检查：

1. 模型是否贴近前车。
2. 模型是否压线或出界。
3. 变道是否合理。
4. 碰撞前是否有明显危险车辆。
5. 车辆是否频繁横向摆动。

### 12.5 代表性集标记

全部 Episode 跑完后，主函数会按场景类型标记代表性集，逻辑在第 853 行到第 872 行附近。

每类场景中：

1. 非碰撞集中总分最高的一集标为 `最佳`。
2. 非碰撞集中总分最低的一集标为 `最难`。
3. 非碰撞集中总分居中的一集标为 `中位`。
4. 所有碰撞集标为 `碰撞`。

这些标记会同时写入 CSV、TXT 和每集图片文件名。查看结果时可以先看这些代表性集，不必一开始就翻完 1000 张图片。

### 12.6 总体统计图生成

`generate_summary_plots` 位于第 716 行到第 770 行附近，生成三类总体图：

1. `score_distribution_{timestamp}.png`：分数分布直方图。
2. `inference_time_{timestamp}.png`：推理耗时分布图。
3. `scenario_comparison_{timestamp}.png`：不同场景平均分对比柱状图。

使用的方法是 `matplotlib`：

1. `plt.subplots(...)` 创建画布。
2. `ax.hist(...)` 绘制直方图。
3. `ax.bar(...)` 绘制柱状图。
4. `fig.savefig(...)` 保存 PNG。
5. `plt.close(fig)` 关闭图，避免大量图片占用内存。

### 12.7 可视化结果如何解读

逐集图建议这样看：

1. 先从 TXT 报告中找到碰撞集或低分集。
2. 根据场景类型和 EP 编号去 `episodes/{scenario_type}/` 找对应图片。
3. 看第一帧：初始车流是否困难。
4. 看中间帧：主车是否开始贴近前车、偏离车道、转向过猛。
5. 看最后帧：如果碰撞，判断是追尾、侧向擦碰、出界还是避让失败。

总体图建议这样看：

1. `score_distribution` 看整体分数是否稳定，有没有长尾低分。
2. `scenario_comparison` 看模型是否在某类场景明显偏弱。
3. `inference_time` 看模型是否具备实时推理能力。

---

## 13. 关于 `Jolindraw/plot_comparison.py` 的补充说明

`Jolindraw/plot_comparison.py` 不是 Highway-env 测试脚本的一部分，但它和“选择哪个模型拿去 Highway-env 测试”密切相关。

它的作用是比较不同训练实验的训练损失和验证损失，帮助判断哪个模型更值得进入闭环测试。

### 13.1 脚本读取哪些日志

脚本使用 `glob.glob("../train_log/*关键词*.txt")` 在 `train_log` 目录中自动查找日志。

当前配置包括：

1. `Exp-1:Underfitting`：欠拟合实验。
2. `Exp-2:Overfitting`：过拟合实验。
3. `Exp-3:SmallBatch`：小 batch 实验。
4. `Exp-4:Dropout=0.0`：无 dropout 实验。
5. `Exp-5:Dropout=0.15`：dropout 0.15 实验。
6. `Exp-6:Final Model`：最终模型实验。

### 13.2 日志解析逻辑

`parse_log_file` 使用正则表达式提取每行中的训练损失和验证损失：

```text
训练损失: xxx | 验证损失: xxx
```

提取后分别放入：

1. `train_losses`
2. `val_losses`

后续绘图时，横轴是 epoch，纵轴是 loss。

### 13.3 绘图逻辑

脚本创建左右两张图：

1. 左图：Training Loss Comparison。
2. 右图：Validation Loss Comparison。

为了突出最终模型，`Exp-6` 使用红色、较粗线条和更高 `zorder`。其他实验使用较浅颜色和较低透明度。

脚本还设置了：

```text
ax.set_ylim(0.01, 0.15)
```

这样可以避免失败实验早期过大的 loss 把其他曲线压扁。这个设置是为了更清楚地观察主要模型在低 loss 区间的差异。

### 13.4 输出文件

输出目录：

```text
Jolindraw/Transformer_plots/
```

文件名格式：

```text
{月日_小时}_All_Exps_Loss_Comparison.svg
```

这是训练过程对比图，不是 Highway-env 测试结果图。它回答的是“哪个训练实验收敛更好、验证损失更低”；Highway-env v2 回答的是“模型放到闭环交通环境中是否安全、高效、舒适”。

---

## 14. 小白操作建议

### 14.1 第一次运行建议

第一次不要直接跑 1000 集，建议：

```bash
python -u test_highway_env_transformer_v2.py --quick
```

确认以下内容正常：

1. 没有模型路径错误。
2. 没有 `highway-env` 或 `gymnasium` 导入错误。
3. 终端能逐集输出分数。
4. `highway_env_results_v2/{timestamp}/` 能生成。
5. CSV、TXT、总体 PNG 和每集 PNG 都存在。

确认无误后，再使用：

```bash
python -u test_highway_env_transformer_v2.py
```

跑完整默认测试。

### 14.2 对比多个模型时的注意事项

如果要比较多个模型，必须保证：

1. 四类场景相同。
2. 每类场景集数相同。
3. `EPISODE_DURATION` 相同。
4. `TARGET_SPEED` 相同。
5. `VEHICLES_COUNT` 相同。
6. 评分函数相同。
7. 随机种子逻辑相同。

否则两个模型的分数可能不可比，因为它们面对的测试难度不同。

### 14.3 看结果的推荐顺序

建议按以下顺序看结果：

1. 先看终端最终汇总，确认总体安全率、目标达成率和平均总分。
2. 再看 `test_report_{timestamp}.txt`，重点看分场景汇总。
3. 打开 `scenario_comparison_{timestamp}.png`，判断哪类场景最弱。
4. 打开 `score_distribution_{timestamp}.png`，看是否有大量低分长尾。
5. 打开 CSV，按 `total`、`crashed`、`min_ttc_val` 或 `comfort` 排序定位问题集。
6. 最后打开对应 `episodes/{scenario_type}/` 下的逐集 PNG，看具体车辆运动过程。

### 14.4 常见异常现象与解释

如果模型经常碰撞：

1. 检查输入特征是否与训练一致。
2. 检查动作输出是否需要反归一化。
3. 检查 `MODEL_PATH` 是否加载了正确模型。
4. 看碰撞集 PNG，判断是追尾、侧碰还是出界。

如果模型速度很低：

1. 可能模型过于保守。
2. 可能训练数据中低速样本占比过高。
3. 可能目标速度或虚拟目标点设置没有诱导加速。

如果舒适分很低：

1. 查看动作是否频繁跳变。
2. 重点关注 `avg_jerk` 和 `avg_steer_rate`。
3. 逐集图中可能能看到车辆横向摆动或频繁修正。

如果推理耗时异常高：

1. 检查是否使用 CPU。
2. 检查 GPU 是否被其他程序占用。
3. 确认统计的是模型推理而不是整集绘图时间。

如果可视化图片很多、磁盘占用大：

1. 默认完整测试会生成约 1000 张逐集图。
2. 可以先用 `-n 50` 或 `--quick` 调试。
3. 如需减少图片数量，可以修改主函数中“绘制每集可视化”的逻辑，只画代表性集或碰撞集。

---

## 15. 当前 v2 测试脚本的局限与后续可优化点

v2 脚本已经比早期版本完整很多，但仍有一些需要理解的限制：

1. Safety、Efficiency、Comfort 是项目自定义评分，不是 `highway-env` 官方标准评分。
2. 场景分类是脚本通过调整车辆位置、速度和目标点实现的，不是独立的官方 benchmark。
3. 每集可视化是静态多帧 PNG，不是连续视频或 GIF。
4. 当前目标达成率更多表示“无碰撞并接近跑满时长”，不是到达真实地图终点。
5. 当前特征仍是为了适配训练时的 Onsite 风格 54 维输入，不是 `highway-env` 原生 observation。
6. 如果要做严格论文对比，应固定代码版本、模型版本、随机种子和测试协议，并保留完整结果目录。

后续可以考虑的优化：

1. 增加只绘制代表性集的开关，减少完整 1000 集绘图耗时。
2. 增加 GIF 或 MP4 输出，便于展示连续运动过程。
3. 将评分权重写成配置参数，便于做消融实验。
4. 保存每一步轨迹明细，例如 `ego_x/y/speed/action/ttc` 的 per-step CSV。
5. 为不同模型自动生成横向对比报告。
6. 增加更严格的场景复现功能，例如根据 CSV 中 seed 单独重跑某一集。

---

## 16. 一句话总结

`test_highway_env_transformer_v2.py` 是当前项目中用于 Highway-env 闭环测试的综合评估脚本。它不依赖 Onsite 场景文件，而是在 `highway-v0` 中构造跟车、变道、超车、避让四类场景；它把环境状态转换成模型训练时一致的 54 维输入，加载 Transformer `.pth` 权重进行滚动控制，并最终输出安全、效率、舒适、总分、目标达成率、推理耗时、逐集 CSV、完整 TXT 报告和逐集可视化 PNG。使用它时，最重要的是保证模型路径、模型结构超参数、输入特征顺序和动作输出含义与训练时完全一致。


## `redraw_top_episodes.py`详解

下面按你的 4 个问题，结合 `redraw_top_episodes.py` 一步步解释。

## 1. 这份可视化代码的整体逻辑

`redraw_top_episodes.py` 不是重新跑完整 1000 集测试的脚本，而是一个**后处理 + 精选回合重绘脚本**。

它的目标是：

1. 读取 `test_highway_env_transformer_v2.py` 已经生成的 `episode_details_*.csv`。
2. 从四类场景中分别挑选高分、无碰撞的回合。
3. 用 CSV 中保存的 `seed` 重新构造对应场景。
4. 重新加载同一个 Transformer 模型。
5. 重跑这些精选 episode。
6. 用更清晰的可视化方式重新画图。
7. 生成一份精选回合详细报告。

核心入口在 `main()`：

```860:868:redraw_top_episodes.py
def main():
    ts = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join(BASE_DIR, "highway_env_top20_vis", ts)
    for st in SCENARIO_TYPES:
        os.makedirs(os.path.join(save_dir, st), exist_ok=True)

    print("=" * 70)
    print(f"精选高分回合可视化 — Top-{TOP_N}/场景")
```

也就是说，它会新建一个输出目录：

`highway_env_top20_vis/{时间戳}/`

虽然目录名叫 `top20_vis`，但当前代码里实际是：

```85:87:redraw_top_episodes.py
TOP_N            = 5        # 每种场景精选数量
VIS_N_PANELS     = 4        # 每集面板数
VIS_WINDOW_HALF  = 200.0    # 每帧以自车为中心的半窗口宽度 (m), 总显示 400m
```

所以当前实际是**每类场景选 Top-5**，四类场景合计最多 20 集。

---

## 2. 它如何选择要重画的 episode

脚本读取的源文件是：

```61:64:redraw_top_episodes.py
SOURCE_CSV = os.path.join(
    BASE_DIR, "highway_env_results_v2", "0410_104509",
    "episode_details_0410_104509.csv",
)
```

这个 CSV 正是 `test_highway_env_transformer_v2.py` 生成的逐集结果文件。

筛选逻辑在 `main()` 中：

```876:890:redraw_top_episodes.py
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
```

含义是：

1. 按 `scenario_type` 把 CSV 行分成四组。
2. 每组只保留 `crashed == "0"` 的无碰撞回合。
3. 按 `total` 总分从高到低排序。
4. 每类取前 `TOP_N` 个。

所以它不是随机选图，而是**按原始测试结果中 Total 总分最高的无碰撞 episode 选图**。

四类场景仍然是：

```92:98:redraw_top_episodes.py
SCENARIO_TYPES = ["following", "lane_change", "overtaking", "avoidance"]
SCENARIO_CN = {
    "following":   "跟车",
    "lane_change": "合适时机变道",
    "overtaking":  "超车",
    "avoidance":   "避让",
}
```

---

## 3. 重新仿真的逻辑

选出 CSV 中的高分回合后，脚本不会直接读取 v2 已保存的 PNG，而是用 seed 重新跑一遍对应 episode。

关键代码：

```925:932:redraw_top_episodes.py
for rank_i, orig_row in enumerate(chosen, start=1):
    env_seed    = int(orig_row["seed"])
    orig_ep_idx = int(orig_row["episode_idx"])
    run_count  += 1

    try:
        ep = run_episode_v3(env, model, mi, st, rank_i, env_seed)
```

`env_seed` 来自原始 CSV。它会传入：

```508:510:redraw_top_episodes.py
def run_episode_v3(env, model, mi, stype, ep_rank, env_seed):
    gx, gy = setup_scenario_reproducible(env, stype, env_seed)
    max_steps = EPISODE_DURATION * POLICY_FREQ
```

然后 `setup_scenario_reproducible()` 会：

```327:335:redraw_top_episodes.py
def setup_scenario_reproducible(env, stype, env_seed):
    """
    使用 env_seed 复现场景。
    env.reset(seed=env_seed) 复现 highway-env 的初始场景布局；
    场景额外布置使用由 env_seed 派生的独立 rng，保证每次调用结果相同。
    """
    env.reset(seed=env_seed)
    rng = np.random.default_rng(env_seed)  # 从相同 seed 派生，保证可复现
```

这里要注意一个专业细节：它能复现 `highway-env` 的基础 reset 场景，但不一定和 `test_highway_env_transformer_v2.py` 中的二次布置完全逐像素一致。因为 v2 中场景额外布置使用的是主函数里的全局 `rng = np.random.default_rng(42)` 连续抽样；而这里是用 `env_seed` 派生一个新的 rng。这样做的好处是每次重画自己可复现，但和原始 1000 集中的额外车辆速度/位置可能存在轻微差异。

---

## 4. 单集运行和模型推理逻辑

`run_episode_v3()` 和 v2 的 `run_episode()` 非常相似，本质仍是闭环控制：

1. 构造场景。
2. 提取 54 维特征。
3. 模型推理得到未来 5 步动作。
4. 在环境中执行 5 步。
5. 记录轨迹、速度、动作、奖励、TTC。
6. 最后重新计算评分。

核心闭环如下：

```535:554:redraw_top_episodes.py
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
        act = aseq_np[i]
        obs, rew, terminated, truncated, info = env.step(act)
```

与 v2 最大区别是：  
这里在**模型刚推理完、动作还没执行前**保存快照，因此图里可以画出“当前模型预测的未来 5 步轨迹点”。

---

## 5. 可视化相比 v2 改进在哪里

文件开头已经总结了这份脚本的改进点：

```8:16:redraw_top_episodes.py
特性（相比 v2 的可视化方案改进）：
  ① 全集固定绝对坐标系：所有面板 x/y 轴范围相同，显示完整行驶轨迹
  ② 目标点：红色方块标注于图中实际 goal_x/goal_y 处
  ③ 自车预测轨迹：绿色小圆点显示当前推理的 5 步预测位置
  ④ 障碍车编号：按距自车距离从近到远编号 1,2,3...
  ⑤ 颜色语义：自车=绿，普通障碍=蓝，危险障碍（TTC<2s）=深蓝，目标点=红
  ⑥ 时间标注：面板标题显示 "t = X.Xs"
```

### 5.1 预测轨迹怎么来的

它不是环境真实轨迹，而是用模型刚输出的 5 步动作做一个简化运动学前推：

```431:449:redraw_top_episodes.py
def compute_pred_traj(x, y, speed, heading, aseq_np):
    """
    用简化自行车模型推演自车未来 5 步位置。
    aseq_np shape: (5, 2)，值域 [-1, 1]（模型 Tanh 输出）
    highway-env ContinuousAction 映射：
      acc   = action[0] * 6      → [-6, 6] m/s²
      steer = action[1] * 0.15   → [-0.15, 0.15] rad
    """
```

所以图中的绿色小圆点表示：  
**在当前时刻，模型认为接下来 5 个控制步主车大概会走到哪里。**

这对分析模型很有用：

- 预测点很平滑：说明动作稳定。
- 预测点突然横跳：说明转向可能激进。
- 预测点朝前车撞过去：说明模型对危险车辆反应不足。
- 预测点偏离道路：说明模型可能在横向控制上有问题。

### 5.2 快照记录内容

`_snapshot_v3()` 会保存：

```493:502:redraw_top_episodes.py
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
```

也就是每个面板需要画的全部信息：当前时间、主车位置、目标点、周围车、预测轨迹。

### 5.3 颜色语义

车辆颜色在 `_snapshot_v3()` 中定义：

```478:490:redraw_top_episodes.py
vehs = [{
    "x": float(ex), "y": float(ey), "heading": float(eh),
    "speed": float(ev), "width": float(ego.WIDTH),
    "color": "limegreen", "label": None, "is_ego": True,
}]
for label_num, (_, v) in enumerate(others[:10], start=1):
    vx, vy = v.position
    ttc = _quick_ttc(ego, v)
    color = "navy" if (ttc < 2.0 and vx > ex) else "royalblue"
```

含义：

- 绿色：自车。
- 蓝色：普通障碍车。
- 深蓝色：前方 TTC 小于 2 秒的潜在危险障碍车。
- 红色方块：目标点。

---

## 6. 更换测试模型需要改哪里

### 6.1 模型权重路径

位置：

```55:59:redraw_top_episodes.py
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_"
    "enc3_dec3_500es_CoAnWarmRest_zDATASET.pth",
)
```

这是路径配置。换 `.pth` 文件时，首先改这里。

如果只是同结构 Transformer 的不同 checkpoint，通常只改 `MODEL_PATH` 即可。

### 6.2 源 CSV 路径

位置：

```61:64:redraw_top_episodes.py
SOURCE_CSV = os.path.join(
    BASE_DIR, "highway_env_results_v2", "0410_104509",
    "episode_details_0410_104509.csv",
)
```

这是可视化数据来源。它决定“从哪一次 v2 测试结果中挑 Top episode”。

如果你重新跑了 v2 测试，例如生成了：

`highway_env_results_v2/0430_160000/episode_details_0430_160000.csv`

那这里也必须改成新的 CSV，否则它仍然会画旧测试批次中的高分集。

### 6.3 模型结构超参数

位置：

```66:72:redraw_top_episodes.py
D_MODEL        = 256
FFN_DIM        = 4 * D_MODEL
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 3
OUTPUT_DIM     = 2
SEQ_LENGTH     = 5
CAR_NUM        = 8
```

这些是结构超参数：

- `D_MODEL`：Transformer 隐层维度。
- `FFN_DIM`：前馈网络维度。
- `NUM_ENC_LAYERS`：Encoder 层数。
- `NUM_DEC_LAYERS`：Decoder 层数。
- `OUTPUT_DIM`：动作维度。
- `SEQ_LENGTH`：预测步数。
- `CAR_NUM`：周围车数量。

它们必须和训练时一致，否则加载权重时会出现 size mismatch。

### 6.4 模型类定义

位置：

```103:153:redraw_top_episodes.py
class TransformerTrajectoryPredictor(nn.Module):
    ...
```

如果只是换同结构 Transformer，不用改。

如果换成 GRU、LSTM、RNN、自回归 Transformer，或者 Transformer 内部结构变了，就必须替换这个类，并同步调整 `forward()` 输出格式。

当前脚本默认模型输出是：

`(1, 5, 2)`

也就是 5 步，每步 2 个动作。

### 6.5 模型实例化参数

位置：

```895:901:redraw_top_episodes.py
model = TransformerTrajectoryPredictor(
    d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH,
    car_num=CAR_NUM, nhead=8,
    num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS,
    dim_feedforward=FFN_DIM, dropout=0.0,
).to(DEVICE)
```

这里也要核对 `nhead=8`、`dropout=0.0` 等参数。尤其是如果训练时不是 8 头注意力，这里必须同步。

### 6.6 输入特征逻辑

位置：

```188:224:redraw_top_episodes.py
def extract_features(env, mi, goal_x, goal_y):
    ...
```

如果新模型仍然吃 54 维输入，不用改。

如果新模型输入变了，比如：

- 加了历史帧；
- 周围车不是 8 辆；
- 每辆车特征不是 6 维；
- 使用归一化；
- 输入顺序变化；
- 加入车道中心线、曲率、目标速度等新特征；

那就必须改 `extract_features()`。

### 6.7 动作解释逻辑

位置包括：

```553:554:redraw_top_episodes.py
act = aseq_np[i]
obs, rew, terminated, truncated, info = env.step(act)
```

以及预测轨迹处：

```442:443:redraw_top_episodes.py
acc   = float(aseq_np[i, 0]) * 6
steer = float(aseq_np[i, 1]) * 0.15
```

当前脚本默认模型输出是 `[-1, 1]` 范围内的连续动作，直接交给 `highway-env` 的 `ContinuousAction`。

如果新模型输出的是真实物理加速度/转角，或者输出轨迹点、目标车道、目标速度，就不能直接 `env.step(act)`，需要增加动作转换器。

---

## 7. 运行结果和文件生成

运行命令：

```bash
python redraw_top_episodes.py
```

建议使用：

```bash
python -u redraw_top_episodes.py
```

输出目录：

```text
highway_env_top20_vis/{timestamp}/
```

目录结构大致是：

```text
highway_env_top20_vis/
└── 0430_154500/
    ├── following/
    │   ├── rank01_ep0023_T95.png
    │   ├── rank02_ep0047_T94.png
    │   └── ...
    ├── lane_change/
    ├── overtaking/
    ├── avoidance/
    └── detail_report_0430_154500.txt
```

### 7.1 每集 PNG 文件

文件名由这里生成：

```949:953:redraw_top_episodes.py
fname = (f"rank{rank_i:02d}_ep{orig_ep_idx+1:04d}"
         f"_T{sc['total']:.0f}.png")
fpath = os.path.join(save_dir, st, fname)
draw_episode_v3(ep, fpath, SCENARIO_CN[st], rank_i, orig_ep_idx)
```

例如：

`rank01_ep0038_T96.png`

含义：

- `rank01`：该场景中精选排名第 1。
- `ep0038`：它来自原始 v2 测试 CSV 中第 38 个 episode。
- `T96`：重跑后总分约 96。

每张 PNG 中包含多个时间面板，当前默认 4 个面板。图里会显示：

- 道路边界和车道线；
- 自车；
- 周围车；
- 危险车；
- 目标点；
- 自车未来 5 步预测轨迹；
- 当前时间 `t = X.Xs`；
- 本集 Total / Safety / Efficiency / Comfort 分数。

### 7.2 TXT 详细报告

文件名：

`detail_report_{timestamp}.txt`

生成位置：

```961:963:redraw_top_episodes.py
report_path = os.path.join(save_dir, f"detail_report_{ts}.txt")
generate_detail_report(all_results, all_orig_rows, report_path)
```

报告内容包括：

1. 模型路径。
2. 基线 CSV 来源。
3. 精选方式。
4. 原始 1000 集和精选集的指标对比。
5. 分场景精选汇总。
6. 每个精选 episode 的详细结果。
7. 每类场景的最佳、中位、最难简评。

不过这里有两个小问题需要注意：

```723:723:redraw_top_episodes.py
f.write(f"  {'指标':<22} {'原始 1000 集':>14} {'精选 80 集':>14}\n")
```

当前 `TOP_N = 5`，四类场景合计应该是 20 集，不是 80 集。所以报告里“精选 80 集”是历史遗留文案，不代表当前真实数量。

另外文件开头说明写的是 Top-20：

```5:6:redraw_top_episodes.py
从已有 1000 集测试结果 CSV 中，每种场景选取 Top-20 高分无碰撞回合，
用保存的 seed 复现仿真场景，以改进后的可视化方案绘图，并生成详细 TXT 报告。
```

但当前代码实际 `TOP_N = 5`。所以应以代码中的 `TOP_N` 为准。

---

## 8. 它和 `test_highway_env_transformer_v2.py` 的关系

这两个文件的关系可以理解为：

`test_highway_env_transformer_v2.py` 是主测试脚本。  
`redraw_top_episodes.py` 是结果复盘和高质量可视化脚本。

### 8.1 v2 负责完整测试

`test_highway_env_transformer_v2.py` 做完整测试：

- 四类场景；
- 默认 1000 集；
- 输出完整 CSV；
- 输出完整 TXT；
- 输出每集基础 PNG；
- 输出总览统计图。

核心输出是：

`highway_env_results_v2/{timestamp}/episode_details_{timestamp}.csv`

### 8.2 redraw 依赖 v2 的 CSV

`redraw_top_episodes.py` 必须依赖这个 CSV：

```61:64:redraw_top_episodes.py
SOURCE_CSV = os.path.join(
    BASE_DIR, "highway_env_results_v2", "0410_104509",
    "episode_details_0410_104509.csv",
)
```

没有这个 CSV，它无法知道哪些 episode 是高分、无碰撞、对应 seed 是什么。

### 8.3 redraw 复用 v2 的核心逻辑

它复用了 v2 的很多核心设计：

- 相同的模型结构；
- 相同的 54 维特征；
- 相同的四类场景；
- 相同的 scoring 逻辑；
- 相同的 `highway-v0` 环境；
- 相同的连续动作接口；
- 相同的 Safety / Efficiency / Comfort / Total 指标。

### 8.4 redraw 改进的是可视化质量

v2 的逐集可视化更偏“批量记录”，每集都画，图比较基础。  
`redraw_top_episodes.py` 更偏“论文展示/报告展示”，只挑高分代表集，画得更细：

- 固定坐标轴；
- 目标点更明显；
- 车辆编号；
- 危险车标色；
- 预测轨迹点；
- 标题包含原始 EP 和精选排名；
- 生成专门的精选回合报告。

所以推荐使用流程是：

1. 先跑 `test_highway_env_transformer_v2.py` 得到完整测试结果。
2. 查看 `test_report` 和总体图，确认模型整体表现。
3. 用 `redraw_top_episodes.py` 从 CSV 中挑高分无碰撞 episode。
4. 生成更适合汇报、论文、答辩展示的精选可视化图。

---

## 9. 需要特别注意的代码问题

有几个小点建议你理解清楚：

1. 文件名和注释里说 Top-20，但当前 `TOP_N = 5`，实际是每场景 Top-5。
2. 输出目录叫 `highway_env_top20_vis`，但实际数量仍由 `TOP_N` 决定。
3. TXT 报告中写了“精选 80 集”，但当前实际通常是 20 集，这是文案遗留。
4. 重放场景使用 `env_seed` 派生 rng，能保证 redraw 自己每次一致，但不一定和 v2 原始场景二次布置完全一致。
5. 如果某个场景没有任何无碰撞 episode，`chosen[-1]` 可能报错，因为代码默认至少能选到一个安全回合。

总体来说，这个脚本的价值不在于重新评估全量性能，而在于**从 v2 的完整测试结果中挑出代表性高分回合，并生成更清晰、更适合展示的可视化复盘图和详细报告**。