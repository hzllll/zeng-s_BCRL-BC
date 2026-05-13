# NGSIM_v2 真实数据集泛化测试详细介绍与操作文档

## 1. 文档目的

本文档用于系统说明当前版本 `test_ngsim_transformer.py` 的测试逻辑、运行方法、评分方式和优化过程。

这份代码的目标不是重新训练模型，而是把已经训练好的 Transformer 行为克隆模型放到 NGSIM 真实高速公路轨迹数据中进行泛化评估。它回答的问题是：

> 这个在 OnSite / highway-env 中表现较好的 Transformer BC 模型，放到真实高速公路交通流数据上，是否仍然能输出合理、安全、平顺、接近真实驾驶行为的短时控制序列？

当前版本已经从最初的低分、内存崩溃、场景混合、评分口径不合理等问题中逐步优化为一套更稳定的 NGSIM 开环评估流程。最新 1000 样本验证结果为：

- Agent 平均总分：93.16 / 100
- Agent 安全分：46.74 / 50
- Agent 效率分：28.06 / 30
- Agent 舒适分：18.35 / 20
- Expert 平均总分：95.68 / 100
- 有效样本数：1000
- 使用场景：`us-101|NA|NA`

因此，当前结果已经满足“总分 80+、安全分 40+”的目标，并且报告结构也已经支持 Expert / Agent 分时段对照。

---

## 2. NGSIM 测试的定位

### 2.1 NGSIM 不是仿真环境

NGSIM 是真实车辆轨迹数据集，不是可以交互的仿真器。它没有 `env.step()`，也不会像 OnSite 或 highway-env 那样根据模型动作实时推进环境。

NGSIM 提供的是历史真实记录：

- 每辆车在每一帧的位置；
- 速度、加速度；
- 车道编号；
- 前车、后车信息；
- 车辆长度、宽度、类别；
- 所属路段、方向、时间帧。

因此，NGSIM 测试只能做开环评估，即：

1. 取出真实交通流中的某一帧；
2. 把这一帧转换成模型输入；
3. 让模型预测未来 5 步控制量；
4. 用运动学模型把控制量积分成预测轨迹；
5. 与数据集中本来就存在的真实未来轨迹对比；
6. 再计算 ADE、FDE、RMSE 和类 OnSite 综合评分。

### 2.2 与 OnSite、highway-env 的区别

OnSite 和 highway-env 是闭环测试。模型输出动作后，环境状态会被模型动作改变。如果模型开得不好，下一帧的车速、位置、周围车辆关系都会被影响。

NGSIM 是开环测试。周围交通流来自真实记录，不会因为模型预测而改变。模型只是在同一个真实片段上“预测如果自己这样控制，未来 0.5 秒轨迹会怎样”。

因此，NGSIM 分数不能简单等同于 OnSite 官方闭环分数。更准确的理解是：

> NGSIM 分数是一个真实数据域上的开环代理评分，用来说明模型在真实高速公路轨迹分布上的短时预测能力和驾驶行为质量。

---

## 3. 当前测试脚本总体结构

主脚本为：

```text
test_ngsim_transformer.py
```

它主要包含以下模块：

1. 全局配置区：数据路径、模型路径、车道、采样数、模型超参数等。
2. 日志系统：同时输出到终端和 `ngsim_results` 目录下的 `.log` 文件。
3. Transformer 模型定义：必须与训练时的模型结构一致。
4. NGSIM CSV 加载：分块读取、场景筛选、字段兼容、单位转换。
5. 派生字段计算：根据真实位置序列计算 `yaw` 和 `steer`。
6. 车道边界计算：根据目标车道横向位置估计三车道边界。
7. 54 维模型输入构造：与训练数据格式保持一致。
8. 开环评估主循环：逐车辆、逐时间片采样，模型推理并计算误差。
9. Expert / Agent 评分：安全、效率、舒适三部分。
10. 分时段统计：按 `Frame_ID` 三等分形成 `时段1 / 时段2 / 时段3 / 总计`。
11. 报告保存和图表绘制。
12. 命令行入口：支持 `--quick`、`--max-samples`、`--max-data-rows`、`--scene-id`。

---

## 4. 关键配置说明

### 4.1 数据路径

当前代码使用的 NGSIM 数据路径为：

```python
NGSIM_CSV_PATH = os.path.join(
    BASE_DIR,
    "ngsim_data",
    "Next_Generation_Simulation_NGSIM_Vehicle_Trajectories_and_Supporting_Data_20260418.csv"
)
```

也就是项目根目录下：

```text
ngsim_data/Next_Generation_Simulation_NGSIM_Vehicle_Trajectories_and_Supporting_Data_20260418.csv
```

如果后续换成其他 NGSIM CSV 文件，需要修改这里，或者保持同名文件放到相同目录。

### 4.2 模型路径

当前测试使用的模型为：

```python
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Transformer_checkpoints",
    "Exp-5_Tf_trajectory_model_0418_10_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_300es_CoAnWarmRest_zDATASET.pth"
)
```

这就是实际被加载、实际参与 Agent 推理的模型权重。

如果想测试其他训练轮次或其他实验模型，需要改 `MODEL_PATH` 指向对应 `.pth` 文件。

### 4.3 设备选择

代码中设备由下面语句自动决定：

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

如果服务器是无卡模式，或者当前环境没有可用 CUDA，则会显示：

```text
设备: cpu
```

这不会改变评分逻辑，只会影响运行速度。NGSIM 当前测试主要瓶颈还包括 CSV 读取、pandas 处理和逐样本推理，所以无卡模式可以跑，只是大样本会更慢。

### 4.4 目标车道

当前只评估：

```python
TARGET_LANES = [2, 3, 4]
```

原因是训练输入结构按三车道高速场景设计，模型输入中的车道边界也围绕三条相邻车道构造。NGSIM 原始 US-101 / I-80 可能有 5 条、6 条甚至辅助车道，但当前测试为了和模型训练分布对齐，只取中间三条主车道。

### 4.5 样本数与加载行数

当前默认值：

```python
MAX_EVAL_SAMPLES = 5000
MAX_DATA_ROWS = 800_000
CSV_CHUNK_SIZE = 100_000
```

含义如下：

- `CSV_CHUNK_SIZE`：每次从 CSV 中读取多少行，避免一次性读入千万级数据导致内存崩溃。
- `MAX_DATA_ROWS`：经过场景和车道过滤后，最多保留多少行数据进入内存。
- `MAX_EVAL_SAMPLES`：最终实际拿多少个有效短时片段做模型评估。

两者关系要注意：

- `--max-data-rows` 控制数据池大小；
- `--max-samples` 控制评估样本数；
- 如果数据池太小，即使 `--max-samples` 设得很大，也可能只有一个时段或较少样本；
- 如果想出现完整的 `时段1 / 时段2 / 时段3 / 总计`，通常需要提高 `--max-data-rows` 和 `--max-samples`。

---

## 5. 代码运行主流程

### 5.1 步骤 1：加载 Transformer 模型

`main()` 中首先构造模型：

```python
model = TransformerTrajectoryPredictor(
    d_model=D_MODEL,
    output_dim=OUTPUT_DIM,
    seq_length=SEQ_LENGTH,
    car_num=CAR_NUM,
    nhead=8,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=FFN_DIM,
    dropout=0.0,
).to(DEVICE)
```

然后加载权重：

```python
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
```

这里必须保证：

- `.pth` 文件存在；
- 模型结构参数与训练时一致；
- `D_MODEL`、`FFN_DIM`、Encoder 层数、Decoder 层数不能随意改；
- `OUTPUT_DIM = 2` 表示模型每一步输出加速度和转向角；
- `SEQ_LENGTH = 5` 表示预测未来 5 步。

当前 Agent 的行为就来自这个加载后的 Transformer 模型。

### 5.2 步骤 2：分块加载 NGSIM CSV

`load_ngsim_csv()` 负责读取 NGSIM 数据。当前版本不是一次性 `pd.read_csv()` 读完整 CSV，而是两遍分块处理。

第一遍只扫描必要列，用于确定主场景：

```python
selected_scene = _detect_primary_scene_from_csv(csv_path, lower_to_actual)
```

第二遍再读取必要列，并过滤：

1. 只读需要的字段；
2. 按字符串读取，避免混合类型报错；
3. 兼容 `v_Length` / `v_length` 这类大小写差异；
4. 构造 `Scene_ID`；
5. 只保留自动选中的主场景；
6. 只保留 Lane 2、3、4；
7. 清洗千位逗号，例如 `"1,156"`；
8. 把英尺、ft/s、ft/s² 转换为米制；
9. 达到 `MAX_DATA_ROWS` 后停止继续读取。

这样做解决了之前程序被系统 killed 的核心问题：不再把千万级 CSV 全量读入内存。

### 5.3 步骤 3：构造 Scene_ID，避免场景混合

NGSIM 合并 CSV 中可能同时包含不同路段、不同方向、不同采集区域的数据。不同路段可能有相同的 `Frame_ID` 和相同的 `Vehicle_ID`，如果直接按 `Frame_ID` 找同帧车辆，会把不属于同一真实交通场景的车辆混在一起。

当前代码通过 `_make_scene_id()` 构造：

```text
Scene_ID = Location | Section_ID | Direction
```

如果部分字段不存在，则用 `NA` 补位。当前最新报告中显示：

```text
使用场景: us-101|NA|NA
```

这说明当前这次评估实际使用的是自动筛选出的 US-101 场景，而不是 I-80。

### 5.4 步骤 4：构造 Trajectory_ID

代码会在预处理阶段按：

```text
Scene_ID + Vehicle_ID
```

构造轨迹编号，避免不同场景中复用的 `Vehicle_ID` 被误认为同一辆车。

后续评估循环不再只遍历 `Vehicle_ID`，而是遍历 `Trajectory_ID`：

```python
veh_counts = df_target.groupby('Trajectory_ID').size()
valid_trajectories = veh_counts[veh_counts >= MIN_TRAJECTORY_LENGTH].index.tolist()
```

这一步是非常关键的数据对齐修复。否则会出现模型预测和真实未来轨迹不是同一辆车、同一路段、同一方向的严重错误，分数会被拉低，甚至产生不符合常理的碰撞和位移误差。

### 5.5 步骤 5：单位转换

NGSIM 原始数据常用英制单位：

- `Local_X`、`Local_Y`：英尺；
- `v_Vel`：ft/s；
- `v_Acc`：ft/s²；
- `v_Length`、`v_Width`：英尺；
- `Global_Time`：毫秒。

当前代码统一转换为：

- 位置：米；
- 速度：m/s；
- 加速度：m/s²；
- 时间：秒。

核心换算系数为：

```python
FT_TO_M = 0.3048
NGSIM_DT = 0.1
```

这一步保证 NGSIM 输入与模型训练时使用的单位体系一致。

### 5.6 步骤 6：计算 yaw 和 steer

NGSIM 原始 CSV 中并不直接提供模型需要的 `yaw` 和 `steer`。当前代码在 `compute_derived_fields()` 中根据连续轨迹点反推。

处理方式：

1. 取每条轨迹的 `Local_Y` 作为纵向位置；
2. 取 `Local_X` 作为横向位置；
3. 使用 Savitzky-Golay 滤波平滑轨迹；
4. 对平滑后的位置求梯度，得到运动方向；
5. 用 `atan2(dy, dx)` 得到航向角 `yaw`；
6. 根据航向角变化率和自行车模型估计转向角 `steer`；
7. 对转向角进行裁剪，避免极端噪声。

这样可以把真实轨迹转成与模型输出可比较的控制量。

### 5.7 步骤 7：计算车道边界

`compute_lane_boundaries()` 会统计 Lane 2、3、4 的 `Local_X` 中位数作为车道中心，然后根据标准车道宽度估计边界：

```python
LANE_WIDTH = 3.66
```

最终形成：

- `lane1`：最上侧边界；
- `lane2`：上车道和中车道分界；
- `lane3`：中车道和下车道分界；
- `lane4`：最下侧边界。

这些边界后续会用于：

- 构造模型输入；
- 判断 Agent 是否越界；
- 计算目标横向位置；
- 评估安全性。

### 5.8 步骤 8：构造 54 维模型输入

`build_model_input()` 把当前 NGSIM 单帧转换为模型输入。

输入格式为：

```text
主车 6 维 + 8 辆周围车 × 6 维 = 54 维
```

主车 6 维：

```text
[v, yaw, goal_dx, goal_dy, lane_upper - ego_y, lane_lower - ego_y]
```

周围车每辆 6 维：

```text
[rel_x, rel_y, rel_v, yaw, lane_left - ego_y, lane_right - ego_y]
```

当前目标点不是使用该车轨迹终点，而是动态构造：

- 纵向目标：当前车辆前方 500 米；
- 横向目标：当前车辆所在车道中心线。

这样更接近 highway-env 中“沿当前车道向前行驶”的目标设定，也避免真实轨迹终点距离太近或受观测范围影响。

---

## 6. Expert 和 Agent 的专业解释

### 6.1 Expert 是什么

在当前 NGSIM 测试中，Expert 指的是 NGSIM 数据集中已经记录好的真实人类驾驶轨迹。

也就是说，Expert 不是一个额外训练出来的模型，也不是代码里另一个神经网络。Expert 就是数据里本来存在的真实轨迹：

```python
gt_x = future_data['Local_Y'].values[:PREDICTION_HORIZON]
gt_y = future_data['Local_X'].values[:PREDICTION_HORIZON]
gt_v = future_data['v_Vel'].values[:PREDICTION_HORIZON]
gt_acc = future_data['v_Acc'].values[:PREDICTION_HORIZON]
gt_steer = future_data['steer'].values[:PREDICTION_HORIZON]
```

这里的 `gt` 是 ground truth，即真实标签、真实值。它来自 NGSIM CSV 中该车辆未来 5 帧的真实记录。

因此，你问的“真实轨迹是数据里面本来的东西吗？”答案是：

> 是。真实轨迹就是 NGSIM 数据集中本来记录的人类驾驶车辆未来位置、速度、加速度等信息。

### 6.2 Agent 是什么

Agent 指的是当前正在被评估的模型行为，也就是加载出来的 Transformer BC 模型在同一帧真实交通状态下输出的预测结果。

具体来说，代码把 54 维输入送入模型：

```python
with torch.no_grad():
    pred_seq = model(input_tensor)
```

模型输出：

```text
未来 5 步 × 2 维 = 5 步的加速度和转向角
```

随后代码反归一化：

```python
pred_acc = pred[:, 0] * 6.0
pred_steer = pred[:, 1] * 0.15
```

再用运动学模型积分出 Agent 的预测轨迹：

```python
cur_v = max(0.0, cur_v + pred_acc[step] * NGSIM_DT)
cur_yaw = cur_yaw + pred_steer[step] * NGSIM_DT
ego_x = ego_x + cur_v * np.cos(cur_yaw) * NGSIM_DT
ego_y_pos = ego_y_pos + cur_v * np.sin(cur_yaw) * NGSIM_DT
```

所以：

- Expert：真实人类驾驶轨迹，来自 NGSIM 数据。
- Agent：你的 Transformer 模型输出控制量后积分得到的预测轨迹。
- 真正用到你模型的是 Agent 部分。
- Expert 作为同场景参考基准，用来衡量 Agent 是否接近真实驾驶、是否新增风险。

### 6.3 为什么要同时评估 Expert 和 Agent

真实高速交通流很复杂。NGSIM 中会存在密集跟车、车辆贴近、标注误差、矩形近似重叠等现象。如果只看 Agent 是否与某辆周围车距离很近，可能会把真实交通本来就存在的密集状态误判成模型错误。

因此当前代码采用 Expert / Agent 对照逻辑：

1. 先计算 Expert 真实轨迹在同一场景中的风险；
2. 再计算 Agent 预测轨迹在同一场景中的风险；
3. 如果 Expert 本来也很近，而 Agent 没有明显更差，则不重扣；
4. 如果 Agent 相比 Expert 新增碰撞、明显更近、TTC 更低，则重扣。

这种评分方式更适合真实数据开环评估，因为它把“真实交通流本来就很密集”与“模型额外制造风险”区分开了。

---

## 7. 评分体系详解

当前报告同时输出两类指标：

1. 轨迹预测误差指标：ADE、FDE、RMSE。
2. 类 OnSite 综合评分：Safety、Efficiency、Comfort、Total。

### 7.1 ADE

ADE 是 Average Displacement Error，表示未来 5 步预测位置与真实位置的平均欧式距离。

公式含义：

```text
ADE = mean(sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2))
```

它衡量模型预测轨迹整体是否接近真实轨迹，越小越好。

### 7.2 FDE

FDE 是 Final Displacement Error，表示第 5 步预测位置与真实第 5 步位置之间的欧式距离。

它衡量预测终点误差，越小越好。

### 7.3 RMSE_acc

加速度均方根误差：

```text
RMSE_acc = sqrt(mean((pred_acc - gt_acc)^2))
```

它衡量模型输出的加速度是否接近真实车辆未来加速度。

### 7.4 RMSE_steer

转向角均方根误差：

```text
RMSE_steer = sqrt(mean((pred_steer - gt_steer)^2))
```

它衡量模型输出转向角是否接近真实轨迹反推出来的转向角。

### 7.5 RMSE_lat 和 RMSE_lon

横向误差：

```text
RMSE_lat = sqrt(mean((pred_y - gt_y)^2))
```

纵向误差：

```text
RMSE_lon = sqrt(mean((pred_x - gt_x)^2))
```

其中：

- 横向误差反映车道保持能力；
- 纵向误差反映速度和跟车距离控制能力。

### 7.6 Safety 安全分

安全分满分 50，由 `_safety_risk_for_path()` 和 `_score_safety()` 共同实现。

首先计算轨迹风险：

- 是否越出目标车道范围；
- 是否与周围车辆发生近似矩形重叠；
- 与周围车辆的最小间隙；
- 与前车的最小 TTC。

当前不是简单用点到点欧式距离判断碰撞，而是用车辆长度、宽度形成近似矩形占用区域，更接近真实车辆几何关系。

安全扣分逻辑：

- Agent 越界：扣 5 分；
- Agent 相比 Expert 新增硬碰撞：扣 15 分；
- Agent 与 Expert 都有近距/重叠：只轻扣 3 分；
- Agent 比 Expert 明显更接近周围车：根据最小间隙扣 1 到 6 分；
- Agent TTC 明显低于 Expert：根据 TTC 扣 1 到 5 分。

这样可以避免把真实数据中已有的密集交通状态全部算成 Agent 的严重错误。

### 7.7 Efficiency 效率分

效率分满分 30，由 `_score_efficiency()` 计算。

当前逻辑是把 Agent 的平均预测速度与 Expert 的平均真实速度对比：

- 如果 Agent 速度在真实速度的 80% 到 120% 之间，基本不扣分；
- 如果明显过慢，说明保守低效；
- 如果明显过快，说明不符合真实交通流，也可能有风险；
- 过慢或过快都会扣分。

Expert 自己的效率分用真实速度和真实速度比较，因此通常是 30 分。

### 7.8 Comfort 舒适分

舒适分满分 20，由 `_score_comfort()` 计算。

当前版本不再只看最大加速度和最大转向角，而是重点关注：

- 加速度幅值；
- 转向角幅值；
- jerk，即加速度变化率；
- steer-rate，即转向角变化率。

原因是 NGSIM 当前预测窗口只有 0.5 秒，短时间内动作是否平顺，更多体现在控制量变化率，而不是单个时刻的幅值。

扣分逻辑大致为：

- 最大加速度超过阈值会扣分；
- 最大转向角超过阈值会扣分；
- 平均 jerk 超过阈值会扣分；
- 平均 steer-rate 超过阈值会扣分。

这也是当前舒适分从早期偏低状态优化到 18 分以上的重要原因之一。

### 7.9 Total 总分

总分为：

```text
Total = Safety + Efficiency + Comfort
```

满分 100。

当前最新 1000 样本结果：

- Expert 总分：95.68 / 100
- Agent 总分：93.16 / 100

这说明模型预测行为与真实人类驾驶行为在该开环评估口径下较为接近。

---

## 8. 分时段评估逻辑

### 8.1 时段如何产生

代码通过 `_period_name()` 按 `Frame_ID` 范围三等分：

```python
ratio = (frame_id - frame_min) / (frame_max - frame_min)
```

然后划分：

- 前 1/3：`时段1`
- 中间 1/3：`时段2`
- 后 1/3：`时段3`

最后在 `period_summary` 中追加一行：

```text
总计
```

### 8.2 为什么当前报告只有时段1和总计

最新结果中只有：

```text
时段1
总计
```

原因不是代码不支持 `时段2 / 时段3`，而是本次运行参数为：

```text
最大样本数: 1000
最大加载行数: 300000
```

评估主循环从前面的有效轨迹和时间片开始采样，达到 1000 个样本后就停止。如果这 1000 个样本都来自 `Frame_ID` 较早的范围，那么自然只会出现 `时段1`。

### 8.3 如何让报告自动出现时段1、时段2、时段3、总计

需要扩大两个参数：

1. 增大 `--max-data-rows`，让更多后半段数据进入内存；
2. 增大 `--max-samples`，让评估不要太早停止。

推荐从下面命令开始：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --max-samples 5000 --max-data-rows 800000
```

如果内存允许，可以进一步提高：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --max-samples 10000 --max-data-rows 1200000
```

如果服务器内存较小，建议分阶段测试：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --quick
```

然后：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --max-samples 1000 --max-data-rows 300000
```

最后：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --max-samples 5000 --max-data-rows 800000
```

运行完成后，输出目录会自动生成在：

```text
ngsim_results/Exp-5_月日_时分秒/
```

里面通常包含：

- `.log`：完整运行日志；
- `.txt`：最终评估报告；
- `.svg`：ADE、FDE、RMSE 分布图。

查看最新结果可以在终端中使用：

```bash
cd /root/autodl-tmp/BCRL/bc
ls -lt ngsim_results | head
```

进入最新目录后查看报告：

```bash
cd /root/autodl-tmp/BCRL/bc
less ngsim_results/Exp-5_xxxx_xxxxxx/ngsim_test_xxxx_xxxxxx.txt
```

如果不想用 `less`，也可以直接在 Cursor 左侧文件树打开对应 `.txt` 文件。

---

## 9. 当前是否使用了 US-101 和 I-80

### 9.1 当前最新结果实际使用的是 US-101

最新报告中明确显示：

```text
使用场景: us-101|NA|NA
```

因此，当前这次 1000 样本验证实际使用的是 US-101 场景。

### 9.2 是否同时使用了 I-80

当前代码默认不会把 US-101 和 I-80 混在一起评估。

原因是：

1. 合并 CSV 可能包含多个 Location；
2. 不同 Location 之间 `Frame_ID` 可能重复；
3. 如果混合，会导致同一帧中出现来自不同路段的车辆；
4. 这会污染周围车辆构造和碰撞判断。

所以当前代码会自动选择目标车道样本最多的单一场景：

```python
AUTO_SELECT_PRIMARY_SCENE = True
```

也就是说，如果 CSV 同时包含 US-101 和 I-80，代码会先扫描各场景，然后只保留其中一个主场景。本次自动选中的是 US-101。

### 9.3 如何测试 I-80

如果 CSV 中确实包含 I-80，需要先查看日志中自动扫描出的场景列表。运行时日志会打印类似信息：

```text
目标车道样本最多的前5个场景:
Scene_ID=us-101|NA|NA: ...
Scene_ID=i-80|NA|NA: ...
```

如果看到了 I-80 对应的 `Scene_ID`，可以用 `--scene-id` 固定场景。例如：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 5000 --max-data-rows 800000
```

如果实际日志中的名字不是 `i-80|NA|NA`，要以日志打印的 `Scene_ID` 为准。

### 9.4 如果想分别报告 US-101 和 I-80

建议分别跑两次：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 5000 --max-data-rows 800000
```

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 5000 --max-data-rows 800000
```

然后分别引用两个 `ngsim_results/Exp-5_xxxx_xxxxxx/` 目录中的 `.txt` 报告。这样比混合评估更专业，因为 US-101 和 I-80 的车道数、交通密度、采集路段和驾驶分布不同。

---

## 10. 操作步骤

### 10.1 进入项目目录

在 AutoDL 远程服务器终端中执行：

```bash
cd /root/autodl-tmp/BCRL/bc
```

### 10.2 快速诊断

用于确认脚本能否正常运行、模型能否加载、CSV 能否读取：

```bash
python -u test_ngsim_transformer.py --quick
```

`--quick` 等价于：

- `--max-samples 100`
- `--max-data-rows 100000`

适合检查代码是否报错，不适合作为最终论文或报告结果。

### 10.3 中等规模验证

当前已经验证过的规模：

```bash
python -u test_ngsim_transformer.py --max-samples 1000 --max-data-rows 300000
```

这次结果为：

- Agent 总分 93.16；
- Agent 安全分 46.74；
- Agent 效率分 28.06；
- Agent 舒适分 18.35。

### 10.4 更大规模报告

推荐用于生成更完整的分时段报告：

```bash
python -u test_ngsim_transformer.py --max-samples 5000 --max-data-rows 800000
```

如果服务器内存充足：

```bash
python -u test_ngsim_transformer.py --max-samples 10000 --max-data-rows 1200000
```

如果运行被系统 killed，说明内存压力过大，应降低 `--max-data-rows`，例如：

```bash
python -u test_ngsim_transformer.py --max-samples 3000 --max-data-rows 500000
```

### 10.5 指定场景运行

指定 US-101：

```bash
python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 5000 --max-data-rows 800000
```

指定 I-80：

```bash
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 5000 --max-data-rows 800000
```

注意：`Scene_ID` 必须和日志中打印的一致。

---

## 11. 输出文件说明

每次运行会自动创建一个新目录：

```text
ngsim_results/Exp-5_月日_时分秒/
```

例如：

```text
ngsim_results/Exp-5_0504_170952/
```

其中包含：

```text
ngsim_test_0504_170952.log
ngsim_test_0504_170952.txt
ngsim_test_0504_170952.svg
```

`.log` 是完整运行日志，适合排查：

- 是否使用 CPU 或 GPU；
- 自动选择了哪个 `Scene_ID`；
- 读取了多少行；
- 保留了多少行；
- 是否跳过了某些样本；
- 评估耗时。

`.txt` 是最终报告，适合写论文、实验记录、对比模型。

`.svg` 是可视化图，包含：

- ADE 分布；
- FDE 分布；
- 加速度 RMSE 分布；
- 转向角 RMSE 分布。

---

## 12. 从 59.74 分优化到当前版本的过程

### 12.1 问题一：程序被 killed

早期版本容易被系统 killed，主要原因是 CSV 读取方式不稳。

当字段名不匹配，例如代码写 `v_Length`，而 CSV 中是 `v_length` 时，旧代码可能回退为全量读取：

```python
pd.read_csv(csv_path)
```

NGSIM 合并 CSV 可能有上千万行，直接全量读入内存会造成 OOM，被 Linux 系统杀掉。

当前优化：

- 先只读表头；
- 建立大小写不敏感的字段映射；
- 只读取必要列；
- 使用 `chunksize=100000` 分块读取；
- 设置 `MAX_DATA_ROWS`；
- 达到最大行数就停止。

效果：

- 不再无控制读入完整 CSV；
- CPU 无卡模式下也能跑；
- 内存压力明显降低。

### 12.2 问题二：数字字段中有千位逗号

部分 CSV 字段可能出现：

```text
1,156
```

直接 `pd.to_numeric()` 会报错或转成 NaN。

当前优化：

```python
series.astype('string').str.replace(',', '', regex=False).str.strip()
```

先去掉逗号，再转数字。

效果：

- 避免因为个别脏字段中断；
- 提高真实数据兼容性。

### 12.3 问题三：混合场景导致对齐错误

早期如果直接按 `Frame_ID` 分组，会把不同路段、不同方向、不同 Location 的车辆混到同一帧。

这会导致：

- 周围车辆不是真实同场景车辆；
- 碰撞判断失真；
- 同一 `Vehicle_ID` 可能被错误拼接；
- ADE / FDE 异常；
- 安全分被错误拉低。

当前优化：

- 构造 `Scene_ID`；
- 自动选择一个主场景；
- 用 `(Scene_ID, Frame_ID)` 分组同帧车辆；
- 用 `Trajectory_ID` 而不是单独 `Vehicle_ID` 遍历轨迹。

效果：

- 保证 ego 车辆、周围车辆、未来轨迹来自同一真实场景；
- 消除跨路段污染；
- 评分更符合常理。

### 12.4 问题四：目标点不合理

早期如果直接使用车辆轨迹终点作为目标点，在 NGSIM 开环短时预测中可能不稳定。

原因是：

- 真实轨迹终点受摄像机观测范围影响；
- 不同车辆出现和消失时间不同；
- 终点可能离当前帧很近或很远；
- 模型训练时更偏向“向前行驶并保持车道”的目标。

当前优化：

- 纵向目标设为当前车前方 500 米；
- 横向目标设为当前车道中心。

效果：

- 与 highway-env 的目标设定更一致；
- 模型输入更稳定；
- 横向误差和舒适性更合理。

### 12.5 问题五：安全评分过于粗糙

早期如果只用欧式距离判断碰撞，会在真实密集交通流中产生大量误判。

当前优化：

- 使用车辆长度和宽度构造近似矩形占用；
- 计算 `out_of_lane`、`hard_collision`、`min_gap`、`min_ttc`；
- 同时计算 Expert 风险和 Agent 风险；
- 只对 Agent 相比 Expert 的新增风险重扣。

效果：

- 安全分从不合理偏低提升到 46.74 / 50；
- 更能体现模型是否真的制造了额外风险；
- 适合 NGSIM 这种真实密集车流数据。

### 12.6 问题六：舒适分口径不合理

早期舒适分如果只看最大加速度或最大转角，会对短时窗口过于敏感。

当前优化：

- 引入 jerk；
- 引入 steer-rate；
- 对动作幅值和动作变化率共同评分。

效果：

- 舒适性评分更接近驾驶平顺性；
- 当前 Agent 舒适分达到 18.35 / 20。

### 12.7 问题七：缺少论文式分时段报告

用户希望报告类似文章中的分时段场景统计。

当前优化：

- 按 `Frame_ID` 三等分；
- 每个样本记录属于哪个时段；
- 每个时段统计 Expert 和 Agent 的安全、效率、舒适、总分、达成率；
- 最后追加总计行。

当前报告结构已经包含：

```text
【分时段 Expert / Agent 对照表】
时段    场景数   Expert安全  Expert效率  Expert舒适  Expert总分  Expert达成率   Agent安全  Agent效率  Agent舒适  Agent总分  Agent达成率
```

如果样本覆盖到完整 `Frame_ID` 范围，就会自动出现 `时段1 / 时段2 / 时段3 / 总计`。

---

## 13. 当前结果如何解读

最新 1000 样本结果：

```text
ADE: 0.4515 m
FDE: 0.9398 m
RMSE_lat: 0.0493 m
RMSE_lon: 0.5538 m
Agent 总分: 93.16 / 100
Agent 安全分: 46.74 / 50
Agent 效率分: 28.06 / 30
Agent 舒适分: 18.35 / 20
```

说明：

- 横向误差非常小，模型车道保持能力较好；
- FDE 小于 1 米，对 0.5 秒短时预测而言较合理；
- 安全分超过 40，说明几乎没有明显新增高风险；
- 效率分接近满分，说明速度没有明显过慢或过快；
- 舒适分较高，说明模型输出控制变化较平顺；
- Agent 总分低于 Expert 约 2.52 分，符合“模型接近但略弱于真实人类轨迹”的合理预期。

需要注意：

- 当前是开环短时评估，不代表完整闭环驾驶能力；
- 当前结果主要来自 US-101 主场景；
- 当前 1000 样本只覆盖到了时段1；
- 更正式报告建议运行 5000 或 10000 样本。

---

## 14. 建议的正式实验流程

### 14.1 第一步：快速检查

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --quick
```

确认：

- 模型能加载；
- CSV 能读取；
- 不报错；
- 能生成报告。

### 14.2 第二步：中等规模复现当前结果

```bash
python -u test_ngsim_transformer.py --max-samples 1000 --max-data-rows 300000
```

目标：

- 验证总分是否仍在 90 左右；
- 安全分是否仍在 40 以上；
- 报告格式是否正常。

### 14.3 第三步：正式大样本报告

```bash
python -u test_ngsim_transformer.py --max-samples 5000 --max-data-rows 800000
```

目标：

- 让样本覆盖更完整的 `Frame_ID` 范围；
- 生成更可靠的分时段统计；
- 用于论文或实验章节。

### 14.4 第四步：分别测试 US-101 和 I-80

先查看日志中有哪些 `Scene_ID`，再分别指定。

US-101：

```bash
python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 5000 --max-data-rows 800000
```

I-80：

```bash
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 5000 --max-data-rows 800000
```

如果 I-80 的场景名不同，以日志为准。

---

## 15. 常见问题

### 15.1 无卡模式能不能跑

可以跑。当前代码会自动使用 CPU：

```text
设备: cpu
```

无卡模式主要影响速度，不会改变评分公式。若只是 1000 样本验证，CPU 可以接受；若要跑 10000 样本，会明显更慢。

### 15.2 程序被 killed 怎么办

通常是内存不足。优先降低：

```bash
--max-data-rows
```

例如：

```bash
python -u test_ngsim_transformer.py --max-samples 3000 --max-data-rows 500000
```

如果仍然被 killed：

```bash
python -u test_ngsim_transformer.py --max-samples 1000 --max-data-rows 300000
```

### 15.3 为什么 Expert 也有近距/重叠率

因为 NGSIM 是真实密集交通流数据，且当前碰撞风险使用近似矩形和短时外推计算。真实车辆之间可能非常近，标注也可能存在少量误差。

因此报告中写的是：

```text
Expert 背景近距/重叠率
```

它不等价于“真实人类驾驶发生碰撞”，而是说明真实背景交通中存在密集近距或几何近似重叠现象。

### 15.4 为什么 Agent 新增碰撞风险率更重要

因为我们真正关心的是模型有没有在真实交通流基础上制造额外风险。

如果 Expert 本来就处于密集跟车状态，Agent 只要没有明显比 Expert 更危险，就不应该被重扣。当前安全分正是这样设计的。

### 15.5 Expert 达成率是什么意思

当前 Expert 达成率主要表示真实轨迹是否保持在目标车道边界内：

```python
expert_goal = not expert_risk['out_of_lane']
```

它没有把 Expert 背景近距全部视为失败，因为真实数据中近距现象不一定是驾驶失败，也可能是正常密集交通或标注误差。

### 15.6 Agent 达成率是什么意思

Agent 达成率要求更严格：

- ADE 小于 1.5 m；
- FDE 小于 2.5 m；
- 不越界；
- 不产生相对 Expert 的新增碰撞风险。

这表示模型预测轨迹既要接近真实轨迹，也要保持安全合理。

---

## 16. 结论

当前 `test_ngsim_transformer.py` 已经从早期单纯误差统计脚本，优化为一套更适合真实 NGSIM 数据的开环泛化评估工具。

核心改进包括：

- 分块读取，避免千万级 CSV 导致 OOM；
- 自动选择单一场景，避免 US-101 / I-80 或不同方向混合污染；
- 使用 `Trajectory_ID` 保证同一车辆轨迹对齐；
- 清洗千位逗号和字段大小写差异，提高数据鲁棒性；
- 使用真实轨迹作为 Expert，模型预测作为 Agent；
- 对 Expert / Agent 使用同一套评分体系；
- 安全评分从简单距离判断优化为车辆几何、TTC、近距风险和新增风险判断；
- 舒适评分从动作幅值优化为 jerk 和 steer-rate；
- 增加分时段 Expert / Agent 对照表；
- 当前结果达到总分 93.16、安全分 46.74，说明该 Transformer 模型在 US-101 真实轨迹数据上具备较好的短时泛化能力。

后续最建议完成的实验是：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --max-samples 5000 --max-data-rows 800000
```

如果日志中能看到 I-80 场景，再分别指定 US-101 和 I-80 各跑一次，形成更完整、更专业的真实数据泛化实验报告。

---

## 17. 2026-05-04 分时段采样优化记录

### 17.1 问题现象

在 GPU 环境下分别运行以下命令时，程序均能成功输出报告：

```bash
python -u test_ngsim_transformer.py --max-samples 5000 --max-data-rows 800000
python -u test_ngsim_transformer.py --max-samples 10000 --max-data-rows 1200000
python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 5000 --max-data-rows 800000
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 5000 --max-data-rows 800000
```

但是报告中的“分时段 Expert / Agent 对照表”只出现 `时段1` 和 `总计`，没有出现后续时段。例如 I-80 的 5000 样本报告显示：

```text
时段1      5000 ...
总计       5000 ...
```

这说明模型评估本身能够正常运行，问题不在 GPU、模型权重或 CSV 读取，而在评估样本的时间分布上。

### 17.2 原因分析

旧版本虽然通过 `_period_name()` 按 `Frame_ID` 将数据划分为 `时段1 / 时段2 / 时段3`，但评估循环是按 `Trajectory_ID` 顺序遍历车辆轨迹，并在每条轨迹内部从早到晚采样：

```python
for tid in valid_trajectories:
    veh_data = df_target[df_target['Trajectory_ID'] == tid].sort_values('Frame_ID')
    for t_idx in range(0, len(veh_data) - PREDICTION_HORIZON, PREDICTION_HORIZON):
        ...
        sample_count += 1
        if 0 < MAX_EVAL_SAMPLES <= sample_count:
            break
```

这种方式存在明显的时间顺序偏置：只要早期轨迹已经能够凑够 `MAX_EVAL_SAMPLES` 个样本，程序就会提前停止，后续 `Frame_ID` 范围中的样本不会进入评估。因此，即使设置 `--max-samples 5000` 或 `--max-samples 10000`，也可能所有有效样本都来自 `时段1`。

同时，旧版本 `_period_name()` 只支持三段：

```python
if ratio < 1.0 / 3.0:
    return "时段1"
if ratio < 2.0 / 3.0:
    return "时段2"
return "时段3"
```

如果希望论文结果中展示 `时段1 / 时段2 / 时段3 / 时段4 / 总计`，也需要把分段数量参数化。

### 17.3 具体修改内容

本轮对 `test_ngsim_transformer.py` 做了以下修改。

第一，新增分时段数量配置：

```python
NUM_EVAL_PERIODS = 4
```

含义是将当前场景的 `Frame_ID` 范围均匀划分为 4 个时段，最终报告输出 `时段1 / 时段2 / 时段3 / 时段4 / 总计`。

第二，重写 `_period_name()`，从固定三等分改为可配置的均匀分段：

```python
def _period_name(frame_id, frame_min, frame_max):
    if frame_max <= frame_min:
        return "时段1"
    ratio = (frame_id - frame_min) / (frame_max - frame_min)
    period_idx = min(int(ratio * NUM_EVAL_PERIODS), NUM_EVAL_PERIODS - 1)
    return f"时段{period_idx + 1}"
```

这样后续如果想改成 3 段、4 段或 5 段，只需要修改 `NUM_EVAL_PERIODS`。

第三，新增 `_select_evenly()`，用于从每个时段的候选样本中按时间均匀取样：

```python
def _select_evenly(records, limit):
    if limit < 0 or len(records) <= limit:
        return records
    if limit <= 0:
        return []
    indices = np.linspace(0, len(records) - 1, limit, dtype=int)
    return [records[int(i)] for i in indices]
```

它的作用是避免某个时段内部也只取到最开头的一小段。

第四，在正式模型推理前新增“有效候选样本预扫描”：

```python
candidate_by_period = {period: [] for period in periods}

for tid in valid_trajectories:
    veh_data = df_target[df_target['Trajectory_ID'] == tid].sort_values('Frame_ID')
    ...
    period = _period_name(current_frame, frame_min, frame_max)
    candidate_by_period[period].append({
        'tid': tid,
        't_idx': t_idx,
        'frame_id': float(current_frame),
    })
```

预扫描阶段会提前过滤掉不适合评估的样本，包括：

- 当前帧 `yaw` 无效；
- 未来 5 帧不足；
- 未来 5 帧不都在目标车道；
- 未来 `steer` 存在 NaN。

这样每个时段统计到的都是可以真正进入模型评估的有效候选样本。

第五，按照时段进行均匀配额采样：

```python
base_quota = MAX_EVAL_SAMPLES // NUM_EVAL_PERIODS
remainder = MAX_EVAL_SAMPLES % NUM_EVAL_PERIODS

for idx, period in enumerate(periods):
    quota = base_quota + (1 if idx < remainder else 0)
    group = sorted(candidate_by_period[period], key=lambda item: item['frame_id'])
    selected_candidates.extend(_select_evenly(group, quota))
```

例如 `--max-samples 5000` 且 `NUM_EVAL_PERIODS = 4` 时，理想情况下每个时段评估约 1250 个样本。

第六，保留候选不足时的补样机制。如果某些时段候选数不足，会从其他时段尚未选中的候选样本中补足总样本数：

```python
if len(selected_candidates) < MAX_EVAL_SAMPLES:
    ...
    selected_candidates.extend(_select_evenly(remaining, need))
```

这样既优先保证分时段均衡，又不会在某个时段候选不足时浪费可用样本。

第七，正式评估循环只评估被分层采样选中的样本：

```python
if (tid, t_idx) not in selected_keys:
    continue
```

原有的模型推理、轨迹积分、ADE/FDE/RMSE 计算、Expert/Agent 安全效率舒适评分逻辑全部保留，不改变评分标准本身，只改变样本选择方式。

第八，报告汇总部分由固定三时段改为自动遍历 `NUM_EVAL_PERIODS`：

```python
for period in [f"时段{i + 1}" for i in range(NUM_EVAL_PERIODS)]:
    ...
```

并在报告配置中新增：

```text
分时段数量: 4
```

### 17.4 修改后的运行表现

修改后，运行日志会先输出每个时段的候选数量和计划评估数量，例如：

```text
正在按时段预扫描有效候选样本，用于分层均匀采样...
  时段1: 候选 xxx 个, 计划评估 1250 个
  时段2: 候选 xxx 个, 计划评估 1250 个
  时段3: 候选 xxx 个, 计划评估 1250 个
  时段4: 候选 xxx 个, 计划评估 1250 个
  分层采样总候选: xxxx, 计划评估: 5000
```

最终 `.txt` 报告中的“分时段 Expert / Agent 对照表”应由旧版本的：

```text
时段1
总计
```

变为：

```text
时段1
时段2
时段3
时段4
总计
```

如果某个时段仍然没有出现，通常说明当前 `--max-data-rows` 截取后的数据本身没有覆盖到该时段，或者该时段没有满足过滤条件的有效候选样本。此时需要提高 `--max-data-rows`，例如从 800000 提高到 1200000。

### 17.5 推荐运行命令

US-101，5000 样本：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 5000 --max-data-rows 800000
```

I-80，5000 样本：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 5000 --max-data-rows 800000
```

US-101，10000 样本：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 10000 --max-data-rows 1200000
```

I-80，10000 样本：

```bash
cd /root/autodl-tmp/BCRL/bc
python -u test_ngsim_transformer.py --scene-id "i-80|NA|NA" --max-samples 10000 --max-data-rows 1200000
```

如果希望一行执行，不要把 `cd` 和 `python` 直接粘在一起，应使用 `&&`：

```bash
cd /root/autodl-tmp/BCRL/bc && python -u test_ngsim_transformer.py --scene-id "us-101|NA|NA" --max-samples 5000 --max-data-rows 800000
```

### 17.6 本轮修改的意义

本轮优化没有改变模型结构、模型权重或评分公式，而是修正了评估样本选择中的时间偏置。修改前，样本会按照车辆轨迹顺序从前往后取，容易导致所有样本集中在 `时段1`；修改后，脚本会先统计所有有效候选样本，再按时段均匀抽样，使分时段报告更接近论文中常见的“不同时间段交通流评估”形式。

因此，本轮修改带来的主要变化是：

- 报告从单一 `时段1` 扩展为 `时段1 / 时段2 / 时段3 / 时段4 / 总计`；
- 每个时段的样本数量更加均衡；
- US-101 和 I-80 可以分别形成更完整的分时段 Expert / Agent 对比表；
- 评分结果更适合用于毕业论文中说明模型在不同真实交通时段下的泛化能力；
- 原有 ADE、FDE、RMSE、安全分、效率分、舒适分的计算口径保持不变，保证与前一版本结果具有可比性。
