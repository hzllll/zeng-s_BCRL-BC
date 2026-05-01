# BCRL 项目 Highway-env 测试与操作文档

本文档用于说明：如何使用 `highway-env` 对本项目训练得到的行为克隆模型进行闭环测试，当前测试脚本 `test_highway_env_transformer.py` 的详细逻辑是什么，换模型时应该改哪些地方，运行后会生成哪些结果文件，以及测试结果和训练曲线是如何可视化的。

文档面向第一次接触 `highway-env` 的同学，因此会尽量把“代码在做什么”“指标怎么来的”“结果文件怎么看”解释清楚。

---

## 1. Highway-env 测试的定位

### 1.1 它和 Onsite 测试的关系

本项目原始训练数据、Onsite 测试脚本都围绕高速 NOA 场景展开。`planner/sup_train/test_transformer_simulation.py` 是我们已经实践过的 Onsite 测试脚本，它的基本流程是：

1. 读取 Onsite 场景。
2. 从场景中提取主车、目标点、周围车、车道边界等信息。
3. 拼接成模型训练时使用的输入特征。
4. 模型输出未来若干步的控制量。
5. 把控制量交给 Onsite 环境执行。
6. 最后用 Onsite 官方评分软件评价结果。

`test_highway_env_transformer.py` 做的是类似的闭环测试，但测试平台从 Onsite 换成了 `highway-env`：

1. `highway-env` 自动生成高速公路交通流。
2. 脚本从 `highway-env` 中读取主车和周围车状态。
3. 手动转换成与训练数据一致的 54 维输入。
4. 加载训练好的 Transformer `.pth` 模型进行推理。
5. 将模型输出的加速度和转角动作送回 `highway-env`。
6. 统计碰撞、奖励、速度、存活时长、推理耗时等指标。

也就是说，二者最大的相似点是“模型输入输出接口”和“闭环控制逻辑”相似；最大的不同点是“仿真环境来源”和“评分体系”不同。

### 1.2 Highway-env 不使用 Onsite 的 `inputs` 场景

`highway-env` 测试不读取 `planner/inputs` 或 `inputs` 文件夹下的 `.xosc`、`.xodr` 文件。

原因是：

- Onsite 使用 OpenSCENARIO / OpenDRIVE 一类的场景文件。
- `highway-env` 是一个 Gymnasium 风格的强化学习环境，场景由环境内部根据 `config` 随机生成。
- 当前脚本通过 `gym.make("highway-v0", config=config)` 创建场景，周围车辆由 `highway_env.vehicle.behavior.IDMVehicle` 控制。

因此，运行 `test_highway_env_transformer.py` 时，真正参与测试的是 `highway-v0` 自动生成的高速路交通流，而不是 Onsite 官方测试集。

### 1.3 当前测试脚本主要回答的问题

当前 `test_highway_env_transformer.py` 更适合回答以下问题：

- 训练好的 Transformer 模型能否在随机高速交通流中安全行驶？
- 模型是否容易发生碰撞？
- 模型整体速度是否过慢或过快？
- 每一集能否跑满 60 秒？
- 模型单次推理耗时是否足够低？
- 不同随机测试批次之间表现是否稳定？

它目前还不是一个完整的“论文级多维评分系统”。例如安全性、效率性、舒适性、目标达成率可以在当前脚本基础上继续扩展，但当前版本已经记录了这些扩展所需要的一部分基础数据，例如碰撞、速度、动作、奖励、存活时长、推理时间等。

---

## 2. 测试脚本总体运行流程

当前测试入口文件是：

```bash
python test_highway_env_transformer.py
```

建议实际运行时使用非缓冲模式，便于实时看到每个 Episode 的日志：

```bash
python -u test_highway_env_transformer.py
```

脚本整体流程如下：

1. 设置全局配置：模型路径、网络超参数、环境参数、测试集数、结果保存路径。
2. 定义 Transformer 网络结构：必须与训练脚本中的模型结构一致。
3. 定义车道映射函数：把 `highway-env` 的车道坐标解释成模型训练时的 Onsite 风格。
4. 定义特征提取函数：从环境中构造 54 维模型输入。
5. 定义环境创建函数：创建 `highway-v0` 并设置车辆数、车道数、控制频率等。
6. 定义单集测试函数：执行“观测 -> 模型推理 -> 环境 step -> 记录结果”的闭环。
7. 主函数中加载模型，多集循环测试，汇总统计并保存 `.txt` 和 `.svg` 结果。

---

## 3. 测试代码详细逻辑

### 3.1 全局配置区

代码位置：`test_highway_env_transformer.py` 的开头配置区域。

这一部分决定“测哪个模型”“用什么环境测”“测多少集”“结果保存在哪里”。

主要变量说明：

- `DEVICE`：自动选择 `cuda` 或 `cpu`。如果机器有 GPU，模型会放到 GPU 上推理。
- `BASE_DIR`：当前项目根目录，用于拼接后续路径。
- `MODEL_PATH`：待测试模型权重路径，当前指向：

```text
Transformer_checkpoints/Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth
```

- `D_MODEL`：Transformer 隐层维度，当前为 `256`。
- `FFN_DIM`：前馈网络维度，当前为 `4 * D_MODEL = 1024`。
- `NUM_ENCODER_LAYERS`：Transformer Encoder 层数，当前为 `3`。
- `NUM_DECODER_LAYERS`：Transformer Decoder 层数，当前为 `3`。
- `OUTPUT_DIM`：每一步动作维度，当前为 `2`，对应加速度和方向盘转角。
- `SEQ_LENGTH`：每次预测的未来步数，当前为 `5`。
- `CAR_NUM`：模型输入中最多考虑的周围车数量，当前为 `8`。
- `LANES_COUNT`：`highway-env` 车道数，当前为 `3`。
- `LANE_WIDTH`：车道宽度，当前为 `4.0m`。
- `VEHICLES_COUNT`：环境中的车辆总数配置，当前为 `30`。
- `DURATION`：单集最大时长，当前为 `60s`。
- `SIM_FREQ`：仿真频率，当前为 `10Hz`。
- `POLICY_FREQ`：策略频率，当前为 `10Hz`。
- `NUM_EPISODES`：测试集数，当前为 `50`。
- `STEPS_PER_INFERENCE`：每次模型推理后连续执行几步动作，当前为 `5`。
- `GOAL_LOOKAHEAD`：虚拟目标点前视距离，当前为 `500m`。
- `RENDER`：是否开启渲染，当前为 `False`。
- `SAVE_DIR`：结果保存目录，当前为 `highway_env_results`。

### 3.2 模型定义区

代码位置：`TransformerTrajectoryPredictor` 类。

这里重新定义了训练时使用的 Transformer 模型结构。测试脚本必须有模型结构定义，因为 `.pth` 文件只保存了模型参数，加载参数之前必须先创建同样结构的模型对象。

模型输入输出逻辑：

- 输入：一个 54 维向量。
- 前 6 维：主车和目标点相关特征。
- 后 48 维：8 辆周围车，每辆 6 维。
- 输出：形状为 `(1, 5, 2)` 的动作序列。
- `5` 表示未来 5 个控制步。
- `2` 表示每一步有两个动作量：加速度、转角。

模型内部主要步骤：

1. `main_target_feat = x[:, 0:6]` 取出主车与目标相关特征。
2. `vehicle_feats = x[:, 6:54].reshape(-1, 8, 6)` 取出 8 辆周围车特征。
3. 主车特征和车辆特征分别通过线性层映射到 `d_model` 维。
4. 将周围车 token 与主车 token 拼接。
5. 使用 Transformer Encoder 建模车辆之间的关系。
6. 使用可学习的 `future_queries` 作为 Decoder 查询，预测未来 5 步。
7. 最后一层 `Tanh` 将输出限制在 `[-1, 1]` 附近，方便与 `ContinuousAction` 的归一化动作接口衔接。

### 3.3 车道映射逻辑

相关函数：

- `build_map_info`
- `get_ego_lane_bounds`
- `get_veh_lane_bounds`

为什么需要车道映射：

模型训练时用的是 Onsite 风格特征，其中车道边界、左右车道的相对关系有固定含义。`highway-env` 的车道坐标虽然也是二维平面，但车道顺序和我们训练时的约定并不完全相同。如果直接把 `highway-env` 的 `y` 坐标塞给模型，模型可能会把左、右车道关系理解反，从而导致错误换道或边界判断异常。

当前脚本中的映射方式：

- `highway-env` 三车道大致是：
  - lane 0：中心线 `y=0`
  - lane 1：中心线 `y=4`
  - lane 2：中心线 `y=8`
- 脚本构造 Onsite 风格 `map_info` 时进行了反向映射：
  - `map_info[0]` 对应 highway lane 2
  - `map_info[1]` 对应 highway lane 1
  - `map_info[2]` 对应 highway lane 0

这样做的目标是让模型接收到的车道边界特征更接近训练数据分布。

`get_ego_lane_bounds` 用于判断主车当前在哪条车道，并返回主车相对于车道上下边界的距离。

`get_veh_lane_bounds` 用于判断周围车所在车道，并返回该车道上下边界相对于主车的距离。

### 3.4 特征提取逻辑

相关函数：`extract_features(env, map_info, goal_lookahead=500.0)`。

这是整个测试脚本中最关键的适配函数，因为训练出来的模型不是直接认识 `highway-env` 的原始观测，而是认识我们训练数据格式中的 54 维特征。

#### 3.4.1 主车 6 维特征

脚本首先读取主车：

- `ego = env.unwrapped.vehicle`
- `ego.position` 得到主车位置 `(ego_x, ego_y)`。
- `ego.speed` 得到主车速度。
- `ego.heading` 得到主车航向角。
- `ego.LENGTH` 得到主车长度，用于计算车辆间净距离。

然后构造主车 6 维：

1. `ego_v`：主车速度。
2. `ego_yaw_norm`：归一化到 `[-pi, pi]` 的主车航向角。
3. `goal_x - ego_x`：距离虚拟目标点的纵向距离。
4. `goal_y - ego_y`：距离虚拟目标点的横向距离。
5. `map_info[0]['left_bound'] - ego_y`：主车到上边界的相对距离。
6. `map_info[2]['right_bound'] - ego_y`：主车到下边界的相对距离。

其中“虚拟目标点”并不是 `highway-env` 自带的终点，而是脚本人为设置的目标：

- `goal_x = ego_x + GOAL_LOOKAHEAD`
- `goal_y = 当前车道中心线`

也就是说，模型被引导为：尽量沿当前车道向前行驶，而不是去追踪某个固定地图终点。

#### 3.4.2 周围车筛选逻辑

脚本读取周围车：

```text
other_vehicles = env.unwrapped.road.vehicles[1:]
```

这里跳过第 0 辆车，因为第 0 辆是主车。

筛选逻辑：

1. 只考虑横向距离与主车相差不超过 `6m` 的车辆。
2. 计算车辆与主车的纵向净距离，而不是简单中心点距离。
3. 只保留距离主车 `200m` 以内的候选车辆。
4. 按照与主车的综合距离排序。
5. 取最近的 `CAR_NUM=8` 辆车。

为什么计算“净距离”：

如果只用车辆中心点之间的距离，两车很近时会高估安全距离。脚本使用车辆长度修正：

- 如果周围车在前方，则纵向净距离约为 `dx - 半车长和`。
- 如果周围车在后方，则纵向净距离约为 `dx + 半车长和`。
- 如果两车纵向投影重叠，则净距离记为 `0`。

这更接近真实驾驶中的“车头到车尾距离”。

#### 3.4.3 每辆周围车 6 维特征

对于每一辆被选中的周围车，追加 6 个特征：

1. `lon_dist`：相对主车的纵向净距离。
2. `vy - ego_y`：相对主车的横向距离。
3. `v_speed - ego_v`：相对速度。
4. `v_yaw`：周围车航向角。
5. `lb`：周围车所在车道左边界相对主车的距离。
6. `rb`：周围车所在车道右边界相对主车的距离。

8 辆车共 `8 * 6 = 48` 维。

#### 3.4.4 Padding 补齐逻辑

如果周围车不足 8 辆，脚本会用默认值补齐：

```text
[200.0, 0.0, 0.0, 0.0, ego_lb, ego_rb]
```

含义是：

- 这辆“虚拟车”离主车很远，约 `200m`。
- 横向距离为 0。
- 相对速度为 0。
- 航向角为 0。
- 车道边界使用主车所在车道边界。

这样做的目的不是假装真的有一辆车，而是保证输入维度固定为 54 维，满足模型结构要求。

### 3.5 环境创建逻辑

相关函数：`make_env(render=False)`。

该函数通过 `gym.make("highway-v0", render_mode=render_mode, config=config)` 创建测试环境。

当前核心配置包括：

- `observation.type = "Kinematics"`：使用车辆运动学观测。
- `features = ["x", "y", "vx", "vy", "heading"]`：环境本身可以提供这些基础物理量。
- `absolute = True`：使用绝对坐标。
- `normalize = False`：不使用环境内置归一化。
- `action.type = "ContinuousAction"`：使用连续控制动作。
- `acceleration_range = (-6, 6)`：加速度物理范围。
- `steering_range = (-0.15, 0.15)`：转角物理范围。
- `lanes_count = 3`：三车道高速路。
- `vehicles_count = 30`：交通流车辆数量。
- `duration = 60`：单集最长 60 秒。
- `simulation_frequency = 10`：仿真频率 10Hz。
- `policy_frequency = 10`：策略执行频率 10Hz。
- `other_vehicles_type = "highway_env.vehicle.behavior.IDMVehicle"`：周围车由 IDM 行为模型控制。
- `collision_reward = -1.0`：碰撞惩罚。
- `reward_speed_range = [20, 30]`：奖励函数倾向的速度区间。
- `offscreen_rendering = True`：支持无显示器服务器渲染。

这里需要注意：虽然配置里设置了 observation，但当前脚本没有直接使用 `obs` 作为模型输入，而是使用 `env.unwrapped` 读取底层车辆对象，再手动拼接训练格式特征。

### 3.6 单集测试逻辑

相关函数：`run_episode(env, model, map_info, episode_idx=0)`。

单集测试从 `env.reset()` 开始。每次 reset 都会初始化一个新的高速公路交通流。

单集内循环逻辑：

1. 调用 `extract_features` 提取 54 维特征。
2. 转成 PyTorch Tensor，并移动到 `DEVICE`。
3. 使用 `time.perf_counter()` 记录推理开始时间。
4. `with torch.no_grad()` 关闭梯度计算，执行模型前向推理。
5. 再次调用 `time.perf_counter()`，计算本次推理耗时，单位转成毫秒。
6. 模型输出 `action_seq`，形状为 `(1, 5, 2)`。
7. 依次取出未来 5 步动作，循环调用 `env.step(action)`。
8. 每 step 记录 reward、速度、步数。
9. 如果 `terminated` 或 `truncated` 为真，说明本集结束。
10. 如果 `info.get("crashed", terminated)` 表示碰撞，则记录为碰撞。

为什么每次推理后执行 5 步：

- 训练模型本身就是预测未来 `SEQ_LENGTH=5` 步动作。
- `POLICY_FREQ=10Hz`，每步 `0.1s`。
- 5 步对应 `0.5s` 的短期控制序列。
- 这种方式类似 Onsite 测试中的滚动时域控制：每次预测一小段，执行一小段，再重新观测和预测。

单集返回的数据包括：

- `episode`：第几集。
- `steps`：本集实际执行步数。
- `duration_s`：本集持续时间，计算方式为 `steps / POLICY_FREQ`。
- `reward`：本集累计奖励。
- `crashed`：是否碰撞。
- `avg_speed`：本集平均速度。
- `avg_inference_ms`：本集平均推理耗时。
- `inference_times`：本集所有推理耗时列表。

### 3.7 主函数逻辑

相关函数：`main()`。

主函数负责把前面的模块串起来：

1. 打印设备、模型路径、测试集数、环境配置。
2. 创建 `TransformerTrajectoryPredictor` 模型对象。
3. 检查 `MODEL_PATH` 是否存在。
4. 使用 `model.load_state_dict(torch.load(...))` 加载 `.pth` 权重。
5. 调用 `model.eval()` 切换到测试模式。
6. 构建 `map_info` 车道映射并打印。
7. 调用 `make_env` 创建环境。
8. 循环执行 `NUM_EPISODES` 个 Episode。
9. 每个 Episode 结束后在终端打印一行摘要。
10. 测试结束后关闭环境。
11. 汇总安全完成数、碰撞数、平均奖励、平均速度、平均存活时长、推理耗时统计。
12. 保存 `.txt` 测试报告。
13. 保存 `.svg` 可视化图表。

---

## 4. 更换测试模型时需要修改哪里

如果只是换另一个 Transformer 权重，通常只需要改路径和少量超参数。如果换成 GRU、LSTM、RNN 或自回归 Transformer，则需要改动更多地方。

### 4.1 修改模型权重路径

代码位置：全局配置区的 `MODEL_PATH`。

修改对象：路径配置。

什么时候需要改：

- 想测试新训练出来的 `.pth`。
- 想比较不同实验、不同 checkpoint 的 highway-env 表现。

当前写法类似：

```python
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_0330_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_500es_CoAnWarmRest_zDATASET.pth"
)
```

如果要换模型，只需要把最后的文件名替换为新模型文件名。例如：

```python
"Tf_trajectory_model_xxxx.pth"
```

注意事项：

- 路径必须真实存在。
- `.pth` 对应的网络结构必须和脚本里的模型定义一致。
- 如果只是同结构不同训练轮次，通常只改路径即可。

### 4.2 修改 Transformer 超参数

代码位置：全局配置区的 Transformer 参数。

修改对象：模型结构超参数。

主要包括：

- `D_MODEL`：Transformer token 维度。
- `FFN_DIM`：Transformer 前馈层维度。
- `NUM_ENCODER_LAYERS`：Encoder 层数。
- `NUM_DECODER_LAYERS`：Decoder 层数。
- `OUTPUT_DIM`：输出动作维度。
- `SEQ_LENGTH`：未来预测步数。
- `CAR_NUM`：输入周围车数量。

这些参数必须和训练脚本 `get_clone_learning_Transformer6_7.py` 中保存该 `.pth` 时的模型结构一致。

如果不一致，常见错误包括：

- `size mismatch for embedding_main_target.weight`
- `size mismatch for encoder.layers...`
- `Missing key(s) in state_dict`
- `Unexpected key(s) in state_dict`

判断原则：

- 如果 `.pth` 文件名里包含 `256dmodel_1024FFNdim_enc3_dec3`，则对应 `D_MODEL=256`、`FFN_DIM=1024`、`NUM_ENCODER_LAYERS=3`、`NUM_DECODER_LAYERS=3`。
- 如果新模型是 `512dmodel_2048FFNdim_enc4_dec4`，则这些参数要同步改成 `512`、`2048`、`4`、`4`。

### 4.3 修改模型类定义

代码位置：`TransformerTrajectoryPredictor` 类。

修改对象：网络结构代码。

什么时候必须改：

- 训练脚本里的模型类发生了结构变化。
- 新模型不是当前这个 Transformer。
- 新模型加入了额外输入、额外层、不同激活函数、不同 decoder 逻辑。
- 新模型保存的是 GRU、LSTM、RNN 等非 Transformer 模型。

关键原则：

测试脚本里的模型类必须与训练时创建 `.pth` 的模型类完全一致。这里的“一致”不仅是类名一致，而是层名称、层数量、张量维度都要一致。

如果换成 GRU 或 LSTM，需要做的不是只改 `MODEL_PATH`，而是：

1. 把 `TransformerTrajectoryPredictor` 替换成对应的 GRU/LSTM 类。
2. 修改模型初始化参数。
3. 确认 `forward` 输入维度是否还是 54。
4. 确认输出是否还是 `(batch, 5, 2)`。
5. 如果输出不是 5 步动作序列，需要同步修改 `run_episode` 中动作执行逻辑。

### 4.4 修改输入特征提取逻辑

代码位置：`extract_features`。

修改对象：模型输入接口。

什么时候需要改：

- 新模型输入不是 54 维。
- 新模型使用历史序列输入，而不是单帧输入。
- 新模型使用更多周围车或更少周围车。
- 新模型特征顺序变化。
- 新模型加入道路曲率、目标车道、历史速度、历史动作等额外特征。

当前脚本假设输入格式为：

```text
[主车 6 维] + [8 辆周围车 * 每车 6 维] = 54 维
```

如果模型训练时不是这个格式，必须重写 `extract_features`。否则即使程序能跑，模型的输入语义也是错的，测试结果没有可信度。

### 4.5 修改动作解释方式

代码位置：`run_episode` 中取动作并调用 `env.step(action)` 的部分。

修改对象：模型输出接口。

当前脚本假设：

- 模型输出 `action_seq[0, i, :]`。
- 每个动作是 2 维。
- 第 1 维是加速度控制。
- 第 2 维是转向控制。
- 动作值可以直接传给 `ContinuousAction`。

如果新模型输出含义不同，需要检查：

- 是否需要反归一化。
- 是否需要裁剪到 `[-1, 1]`。
- 输出顺序是否为 `[steering, acceleration]` 而不是 `[acceleration, steering]`。
- 是否只输出单步动作，而不是 5 步动作。

动作接口错位会直接导致车辆乱加速、乱转向，是换模型时最容易出现严重问题的地方。

### 4.6 修改环境参数

代码位置：全局配置区和 `make_env`。

修改对象：测试场景难度。

常见可调项：

- `NUM_EPISODES`：测试集数。快速检查可设为 5，正式评估建议 50、100、1000。
- `VEHICLES_COUNT`：周围车辆数量。越大交通越密集。
- `LANES_COUNT`：车道数量。
- `DURATION`：单集时长。
- `reward_speed_range`：奖励函数鼓励的速度区间。
- `acceleration_range`、`steering_range`：动作映射物理范围。

注意：如果修改 `LANES_COUNT` 或 `LANE_WIDTH`，必须确认 `build_map_info` 和车道边界逻辑仍然正确。

---

## 5. 如何运行测试

### 5.1 基础运行命令

在项目根目录执行：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_highway_env_transformer.py
```

如果不关心实时输出，也可以直接运行：

```bash
python test_highway_env_transformer.py
```

但建议使用 `python -u`，因为普通运行在某些终端或后台进程中可能出现输出缓冲，看起来像“没打印日志”。

### 5.2 运行时终端会显示什么

脚本开头会打印：

- 当前设备：`cuda` 或 `cpu`。
- 模型路径。
- 测试集数。
- 环境配置。
- 模型是否加载成功。
- 车道映射信息。

每个 Episode 结束后会打印一行，例如：

```text
Episode  1/50 | 安全 | 步数: 600 | 时长: 60.0s | 累计奖励: 433.94 | 平均速度: 20.9 m/s | 推理: 21.88 ms
```

这行信息的含义是：

- `Episode 1/50`：第 1 集，总共 50 集。
- `安全` 或 `碰撞`：本集是否发生碰撞。
- `步数 600`：本集执行了 600 个环境 step。
- `时长 60.0s`：因为 `POLICY_FREQ=10Hz`，600 步就是 60 秒。
- `累计奖励`：本集所有 step 的 reward 累加值。
- `平均速度`：本集内所有 step 主车速度的平均值。
- `推理`：本集模型单次前向推理平均耗时。

### 5.3 运行时间预期

当前配置为：

- 50 个 Episode。
- 每个 Episode 60 秒仿真时间。
- 每秒 10 个 step。
- 每集最多 600 step。

因此完整 50 集最多执行 `50 * 600 = 30000` 个环境 step。

实际运行时间与机器性能、是否使用 GPU、是否渲染有关。当前脚本 `RENDER=False`，不会保存逐帧画面，因此主要耗时来自环境 step 和模型推理。

---

## 6. 运行结果与生成文件详解

### 6.1 结果保存目录

运行后结果保存在：

```text
highway_env_results/
```

当前项目中已经有示例结果：

```text
highway_env_results/highway_test_0410_005935.txt
highway_env_results/highway_test_0410_005935.svg
highway_env_results/highway_test_0410_011813.txt
highway_env_results/highway_test_0410_011813.svg
highway_env_results/highway_test_0410_012327.txt
highway_env_results/highway_test_0410_012327.svg
```

文件名中的 `0410_011813` 表示生成时间戳，大致可理解为 `04月10日 01:18:13`。

### 6.2 为什么会出现多组数据

`highway_env_results` 中出现多组 `.txt` 和 `.svg`，不是因为脚本一次运行生成了三套不同评估，而是因为脚本被运行了多次。

每次运行都会生成一对文件：

- 一个 `.txt` 测试报告。
- 一个 `.svg` 汇总图。

例如：

- `highway_test_0410_005935` 是一次 5 集快速验证。
- `highway_test_0410_011813` 是一次 50 集测试。
- `highway_test_0410_012327` 是另一次 50 集测试。

同样是 50 集，结果也可能不完全一样，因为 `highway-env` 每次 `env.reset()` 都会随机生成交通流。当前脚本没有固定随机种子，因此不同运行批次的场景不是完全相同的。

### 6.3 `.txt` 测试报告包含什么

`.txt` 文件是最重要的结果说明文件。

开头部分记录测试基本信息：

- 模型路径。
- 环境配置。
- 测试集数。

中间部分记录总体统计：

- `安全完成`：未碰撞的 Episode 数量及比例。
- `碰撞次数`：发生碰撞的 Episode 数量及比例。
- `平均奖励`：所有 Episode 累计奖励的平均值。
- `平均速度`：所有 Episode 平均速度的均值。
- `平均存活时长`：所有 Episode 持续时间的均值。
- `平均推理耗时`：所有模型推理耗时的平均值。

后半部分记录逐集详情：

- `ep`：Episode 编号。
- `安全` 或 `碰撞`：该集是否碰撞。
- `steps`：执行步数。
- `dur`：持续时长。
- `reward`：累计奖励。
- `speed`：平均速度。

### 6.4 指标如何计算

#### 安全完成

代码中通过 `crashed` 判断碰撞：

```text
crashed = info.get("crashed", terminated)
```

如果一个 Episode 没有碰撞并跑到最大时长，就记为安全。

例如 `highway_test_0410_011813.txt`：

```text
安全完成: 49/50 (98.0%)
碰撞次数: 1/50 (2.0%)
```

说明 50 集中有 49 集安全跑满或安全结束，1 集发生碰撞。

#### 平均奖励

每一步 `env.step(action)` 都会返回一个 `reward`。脚本把该 Episode 内所有 step 的 reward 累加：

```text
total_reward += reward
```

测试结束后对所有 Episode 的累计奖励取平均：

```text
avg_reward = np.mean([r["reward"] for r in results])
```

需要注意：当前 `.txt` 中的平均奖励是“每集累计奖励的平均值”，不是单步 reward，也不是归一化到 0-100 的总分。因此它常见数值是几百，例如 `431.03`、`434.58`。

#### 平均速度

每个环境 step 后，脚本读取：

```text
env.unwrapped.vehicle.speed
```

并保存到 `speeds` 列表。

单集平均速度：

```text
avg_speed = np.mean(speeds)
```

总体平均速度：

```text
avg_speed = np.mean([r["avg_speed"] for r in results])
```

速度单位是 `m/s`。例如 `21.7 m/s` 约等于 `78.1 km/h`。

#### 平均存活时长

每个 Episode 的持续时间由步数换算：

```text
duration_s = total_steps / POLICY_FREQ
```

当前 `POLICY_FREQ=10`，因此：

- 600 step = 60 秒。
- 291 step = 29.1 秒。
- 116 step = 11.6 秒。

如果碰撞较早发生，存活时长会明显低于 60 秒。

#### 推理耗时

脚本只统计模型前向推理耗时，不包括环境 step、特征提取和画图时间。

计时方式：

```text
tic = time.perf_counter()
with torch.no_grad():
    action_seq = model(feat_tensor)
toc = time.perf_counter()
inference_times.append((toc - tic) * 1000)
```

单位是毫秒 `ms`。

主函数中还会在终端打印 P50/P95/P99：

- P50：一半推理小于该耗时。
- P95：95% 推理小于该耗时。
- P99：99% 推理小于该耗时。

当前 `.txt` 文件只写入平均推理耗时，没有写入 P50/P95/P99。如果后续希望日志更完整，可以把这些分位数也写进文件。

### 6.5 当前已有结果示例如何解读

`highway_test_0410_011813.txt`：

- 测试集数：50。
- 安全完成：49/50，安全率 98.0%。
- 碰撞次数：1/50，碰撞率 2.0%。
- 平均奖励：431.03。
- 平均速度：21.7 m/s。
- 平均存活时长：59.4 s。
- 平均推理耗时：21.884 ms。
- 第 28 集发生碰撞，持续 29.1 秒。

`highway_test_0410_012327.txt`：

- 测试集数：50。
- 安全完成：48/50，安全率 96.0%。
- 碰撞次数：2/50，碰撞率 4.0%。
- 平均奖励：434.58。
- 平均速度：21.7 m/s。
- 平均存活时长：59.0 s。
- 平均推理耗时：21.826 ms。
- 第 27 集和第 45 集发生碰撞。

这两次测试使用的是同一模型、同一配置，但结果略有差异。原因是 `highway-env` 每次随机生成交通流，当前脚本没有固定 seed。

---

## 7. 可视化过程与结果详解

当前项目中涉及两类可视化：

1. `test_highway_env_transformer.py` 自动生成的 highway-env 测试结果可视化。
2. `Jolindraw/plot_comparison.py` 生成的训练日志对比可视化。

### 7.1 Highway-env 测试结果可视化

代码位置：`test_highway_env_transformer.py` 末尾绘图部分。

触发条件：

- `all_infer` 非空，即至少完成过一次模型推理。

使用方法：

- 使用 `matplotlib`。
- 创建一行两列子图：

```text
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

左图：推理时间分布直方图。

绘图逻辑：

- `axes[0].hist(all_infer, bins=50, ...)`
- 横轴是单次模型推理耗时，单位 ms。
- 纵轴是该耗时区间出现的次数。
- 红色虚线表示平均推理耗时。

图的用途：

- 判断模型是否满足实时性。
- 判断推理耗时是否集中。
- 如果直方图有长尾，说明偶尔会出现较慢推理。

右图：Episode 累计奖励柱状图。

绘图逻辑：

- `episode_rewards = [r["reward"] for r in results]`
- 每个 Episode 画一个柱。
- 安全集使用蓝色。
- 碰撞集使用红色。
- 黑色虚线表示平均奖励。

图的用途：

- 快速观察每集表现是否稳定。
- 红色柱可以快速定位碰撞 Episode。
- 柱高明显偏低的 Episode 通常值得进一步分析。

保存路径：

```text
highway_env_results/highway_test_{timestamp}.svg
```

例如：

```text
highway_env_results/highway_test_0410_011813.svg
highway_env_results/highway_test_0410_012327.svg
```

当前 `.svg` 是统计图，不是车辆行驶动态图。

### 7.2 当前脚本是否保存每个 Episode 的动态图

当前版本没有保存每个 Episode 的动态图，也没有保存逐帧图片。

原因是：

- 全局配置 `RENDER=False`。
- `run_episode` 中没有调用 `env.render()` 保存帧。
- 当前可视化只在测试结束后画汇总统计图。

如果后续要做类似“每 n 帧截图记录”的可视化，可以扩展：

1. 将 `RENDER=True`。
2. `make_env(render=True)` 时使用 `render_mode="rgb_array"`。
3. 在 `run_episode` 每隔 `n` 个 step 调用 `env.render()`。
4. 把返回的 RGB 图像保存为 `.png` 或缓存到列表。
5. 用 `imageio` 或 `moviepy` 合成 `.gif` / `.mp4`。
6. 对碰撞 Episode、代表性跟车 Episode、代表性换道 Episode 单独标注。

推荐的输出目录结构可以是：

```text
highway_env_results/
  highway_test_时间戳.txt
  highway_test_时间戳.svg
  frames_时间戳/
    ep_000/
      frame_0000.png
      frame_0010.png
      ...
  videos_时间戳/
    ep_000.gif
    ep_027_crash.gif
```

### 7.3 训练日志对比可视化：`Jolindraw/plot_comparison.py`

该脚本不是 highway-env 测试脚本的一部分，而是用于比较不同训练实验的 loss 曲线。

它回答的问题是：

- 哪个模型训练得更稳定？
- 哪个实验出现欠拟合？
- 哪个实验出现过拟合？
- 最终选择用于 highway-env 测试的 checkpoint 是否合理？

#### 7.3.1 日志文件查找

脚本使用 `glob` 自动查找训练日志：

```text
../train_log/*TFExp-1*.txt
../train_log/*TFexp-2*.txt
../train_log/*TFexp-3*.txt
...
```

`log_files` 字典把实验名映射到对应日志文件：

- `Exp-1:Underfitting`
- `Exp-2:Overfitting`
- `Exp-3:SmallBatch`
- `Exp-4:Dropout=0.0`
- `Exp-5:Dropout=0.15`
- `Exp-6:Final Model`

如果某个日志找不到，会打印：

```text
未找到 xxx 的日志文件，请检查路径！
```

#### 7.3.2 正则解析训练损失和验证损失

核心函数：

```text
parse_log_file(filepath)
```

它逐行读取训练日志，并用正则表达式提取：

```text
训练损失: xxx | 验证损失: xxx
```

正则模式：

```text
训练损失:\s*([\d.]+)\s*\|\s*验证损失:\s*([\d.]+)
```

提取到的数据分别放进：

- `train_losses`
- `val_losses`

这样每个实验都会得到两条曲线：训练损失曲线和验证损失曲线。

#### 7.3.3 Matplotlib 绘图逻辑

脚本使用：

```text
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
```

左图：

- 标题：`Training Loss Comparison`
- 横轴：Epochs
- 纵轴：Loss
- 内容：各实验训练损失曲线。

右图：

- 标题：`Validation Loss Comparison`
- 横轴：Epochs
- 纵轴：Loss
- 内容：各实验验证损失曲线。

为了突出最终模型，脚本做了特殊视觉设计：

- 如果实验名包含 `Exp-6`：
  - 颜色设置为红色 `#E63946`。
  - 线宽设置为 `2.5`。
  - 透明度为 `1.0`。
  - `zorder=10`，画在最上层。
- 其他实验：
  - 使用灰色、浅蓝、浅绿、浅紫、浅黄等低饱和颜色。
  - 线宽较细。
  - 透明度为 `0.6`。

这样图中最醒目的曲线就是最终模型，便于在论文或汇报中解释为什么选择该模型。

#### 7.3.4 Y 轴范围限制

脚本设置：

```text
ax1.set_ylim(0.01, 0.15)
ax2.set_ylim(0.01, 0.15)
```

原因是有些失败实验早期 loss 可能很大，例如 0.4。如果不限制 Y 轴，其他正常实验后期的 loss 差异会被压缩得很难观察。

这个设置的作用是：

- 放大主要收敛区间。
- 更清楚比较不同模型后期的训练与验证表现。
- 更适合论文图和实验报告图。

但也要注意：Y 轴截断会隐藏早期特别大的 loss。因此如果要全面展示训练初期发散情况，可以另存一张不截断 Y 轴的图。

#### 7.3.5 输出文件

保存路径：

```text
Jolindraw/Transformer_plots/
```

或脚本当前工作目录下的：

```text
Transformer_plots/
```

具体取决于从哪个目录运行脚本。

文件名格式：

```text
{RUN_DATE_TIME}_All_Exps_Loss_Comparison.svg
```

其中 `RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")`，例如：

```text
0419_21_All_Exps_Loss_Comparison.svg
```

SVG 是矢量图格式，适合插入论文、PPT 或报告，放大后不会明显失真。

---

## 8. 当前测试结果应如何专业表述

如果要在实验记录中描述当前 50 集 highway-env 测试，可以参考以下表达方式：

本项目使用 `highway-env` 的 `highway-v0` 环境对训练得到的 Transformer 行为克隆模型进行闭环仿真测试。测试环境设置为三车道高速公路，交通流车辆数为 30，周围车辆采用 IDM 行为模型，单集最大时长为 60 秒，仿真和策略频率均为 10Hz。模型每次接收由主车、虚拟目标点和最近 8 辆周围车构成的 54 维状态特征，输出未来 5 步连续控制动作，并在环境中滚动执行。

在已有两次 50 集随机测试中，模型分别取得 98.0% 和 96.0% 的安全完成率，平均速度约为 21.7 m/s，平均存活时长接近 60 秒。这说明模型在当前 `highway-v0` 随机交通流设置下具有较好的基础安全性和稳定性。但由于当前测试未固定随机种子，且未进一步划分跟车、超车、避让、换道等细分场景，因此这些结果更适合作为初步闭环泛化测试，而不是最终的分类场景评分结论。

---

## 9. 当前版本的局限与后续建议

当前脚本已经可以完成基础 highway-env 闭环测试，但仍有以下局限：

- 没有固定随机种子，不同运行批次不能一一复现实验场景。
- 没有保存每步详细日志，例如每步位置、速度、动作、最近前车距离等。
- 没有保存每个 Episode 的动态图或逐帧图。
- 没有单独计算舒适性指标，例如加速度变化率、转角变化率、jerk。
- 没有把场景自动分类为跟车、超车、避让、换道等类型。
- 当前“目标达成”主要由是否跑满 `DURATION` 间接体现，没有显式终点或路线目标。
- `.txt` 文件没有记录 P50/P95/P99 推理耗时，虽然终端会打印。

后续如需做更正式的论文级评估，建议扩展：

1. 固定 `seed`，保存每个 Episode 的种子。
2. 测试集数提升到 1000 或更多。
3. 保存逐步 CSV 日志。
4. 增加安全性、效率性、舒适性、稳定性、实时性五类指标。
5. 保存代表性 Episode 的 GIF 或 MP4。
6. 对碰撞、急刹、低速、异常换道等失败样本进行自动标记。
7. 按场景类型分类统计结果。

---

## 10. 快速检查清单

正式运行前建议检查：

- `MODEL_PATH` 是否指向正确 `.pth`。
- 模型类是否与训练脚本完全一致。
- `D_MODEL`、`FFN_DIM`、Encoder/Decoder 层数是否与文件名和训练脚本一致。
- 输入特征是否仍为 54 维。
- 输出动作是否仍为 5 步、每步 2 维。
- `NUM_EPISODES` 是否设置为本次需要的测试数量。
- `RENDER` 是否符合需求：普通统计测试用 `False`，录制视频时再改为 `True`。
- 运行命令是否在项目根目录执行。
- 结果是否保存到 `highway_env_results`。

如果只是复现当前已有测试流程，最小操作就是：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_highway_env_transformer.py
```

运行结束后查看：

```text
highway_env_results/highway_test_时间戳.txt
highway_env_results/highway_test_时间戳.svg
```

