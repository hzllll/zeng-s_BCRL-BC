# NGSIM真实数据集泛化测试详细介绍与操作文档

## 1. 文档目的

本文档用于说明：在本项目完成 Transformer 行为克隆模型训练之后，如何使用 NGSIM 真实高速公路轨迹数据集对模型进行泛化能力测试。

本项目主线模型来自 OnSite 高速场景专家策略数据，训练阶段主要关注监督学习损失，OnSite 闭环阶段主要关注官方评分：

- 安全性 Safety：满分 50
- 效率性 Efficiency：满分 30
- 舒适性 Comfort：满分 20
- 总分：100

NGSIM 测试的目标是进一步回答一个问题：

> 模型是否只是学会了 OnSite 场景中的驾驶模式，还是在真实高速公路交通流数据上也能保持合理的预测能力和驾驶行为质量？

因此，NGSIM 测试在本项目中承担的是**真实数据域泛化验证**角色。它不是替代 OnSite 闭环测试，而是对 OnSite 和 highway-env 测试的补充。

---

## 2. NGSIM测试在项目中的定位

### 2.1 与 OnSite 测试的关系

OnSite 测试是闭环测试。模型输出控制量之后，仿真环境会真实执行动作，车辆位置、速度、周围交通状态都会被模型上一时刻的动作影响。因此 OnSite 更接近“让模型自己开车考试”。

OnSite 的结果通常用官方评分体系评价：

- 是否碰撞，是否违规，体现安全性；
- 是否高效完成任务，体现效率性；
- 加速度和转向是否平顺，体现舒适性。

### 2.2 与 highway-env 测试的关系

`test_highway_env_transformer_v2.py` 同样用于泛化能力测试。它把模型放到 highway-env 随机生成的高速环境中执行闭环控制，并参考 OnSite 设计了类似的 100 分评分体系。

`highway_env_results_v2/0410_104509/test_report_0410_104509.txt` 中的报告结构非常适合作为 NGSIM 报告优化参考，例如：

- 总测试集数；
- 安全完成率；
- 平均总分；
- 平均安全分、效率分、舒适分；
- 分场景汇总；
- 代表性样本标记；
- 详细 episode 结果。

### 2.3 NGSIM测试的特殊性

NGSIM 与 OnSite、highway-env 最大不同在于：

> NGSIM 是真实轨迹数据集，不是可交互仿真环境。

NGSIM 没有 `env.step()`，也没有实时碰撞反馈。它只能提供真实车辆在每一帧的位置、速度、加速度、车道编号等记录。

所以当前 `test_ngsim_transformer.py` 的测试方式是**开环测试**：

1. 回放某一帧真实交通状态；
2. 构造与训练阶段一致的模型输入；
3. 模型预测未来 5 步加速度和转向角；
4. 用简化车辆运动学积分出预测轨迹；
5. 与 NGSIM 中真实未来轨迹进行对比；
6. 统计 ADE、FDE、RMSE 等误差指标；
7. 进一步映射出安全性、效率性、舒适性评分。

因此，NGSIM 分数应理解为**真实数据开环代理评分**，不能完全等同于 OnSite 闭环官方分，但可以作为模型泛化能力的重要证据。

---

## 3. 测试代码文件

主测试脚本：

```text
test_ngsim_transformer.py
```

该文件负责完成从模型加载、NGSIM 数据读取、特征构造、模型推理、轨迹积分、误差评估、综合评分到结果保存的完整流程。

当前脚本核心模块包括：

- 全局配置区；
- Transformer 模型定义；
- NGSIM CSV 加载与预处理；
- 航向角和转向角计算；
- 车道边界统计；
- 54 维模型输入构造；
- NGSIM 开环评估主循环；
- 结果保存与图表绘制；
- `main()` 主入口。

---

## 4. 代码整体运行流程

### 4.1 步骤1：加载模型

脚本首先根据 `MODEL_PATH` 加载训练好的 `.pth` 文件。

关键配置位于文件开头附近：

```python
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_xxx.pth"
)
```

模型结构由以下参数决定：

```python
D_MODEL = 256
FFN_DIM = 4 * D_MODEL
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8
```

加载流程在 `main()` 中执行：

```python
model = TransformerTrajectoryPredictor(...).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
```

如果模型路径不存在，会直接报错退出。如果模型结构参数与训练时不一致，则会在 `load_state_dict` 阶段报参数尺寸不匹配。

### 4.2 步骤2：读取NGSIM CSV

NGSIM 数据路径由下面配置控制：

```python
NGSIM_CSV_PATH = os.path.join(
    BASE_DIR,
    "ngsim_data",
    "Next_Generation_Simulation_NGSIM_Vehicle_Trajectories_and_Supporting_Data_20260418.csv"
)
```

脚本会优先尝试按指定字段读取 CSV。如果字段名不完全匹配，例如 `v_Length` 写成了 `v_length`，会退回到读取全部列，并打印实际列名。

### 4.3 步骤3：数据清洗与单位转换

NGSIM 原始数据通常使用英制单位：

- 位置：英尺 ft；
- 速度：ft/s；
- 加速度：ft/s²；
- 时间：毫秒 ms。

项目训练数据使用国际单位制，因此必须转换：

```python
FT_TO_M = 0.3048
```

转换关系：

- `Local_X *= 0.3048`
- `Local_Y *= 0.3048`
- `v_Length *= 0.3048`
- `v_Width *= 0.3048`
- `v_Vel *= 0.3048`
- `v_Acc *= 0.3048`
- `Global_Time /= 1000.0`

我们之前遇到的报错：

```text
TypeError: can't multiply sequence by non-int of type 'float'
```

原因是 CSV 中部分数值列被 Pandas 识别成了字符串。修复方法是：

```python
numeric_cols = [
    'Local_X', 'Local_Y', 'v_Length', 'v_Width',
    'v_Vel', 'v_Acc', 'Space_Headway', 'Lane_ID'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['Local_X', 'Local_Y', 'v_Vel', 'Lane_ID'], inplace=True)
```

同时要兼容列名大小写：

```python
if 'v_length' in df.columns and 'v_Length' not in df.columns:
    df.rename(columns={'v_length': 'v_Length'}, inplace=True)
```

### 4.4 步骤4：计算 yaw 和 steer

NGSIM 原始数据不直接提供模型需要的航向角 `yaw` 和转向角 `steer`，所以脚本会根据连续轨迹点进行估计。

处理逻辑：

1. 对每辆车按照 `Frame_ID` 排序；
2. 使用 `Local_Y` 作为模型纵向位置 `x`；
3. 使用 `Local_X` 作为模型横向位置 `y`；
4. 用 Savitzky-Golay 滤波器平滑轨迹；
5. 对平滑后位置求导；
6. 根据 `arctan2(dy, dx)` 得到航向角；
7. 根据自行车模型近似反推转向角。

这一步非常重要，因为真实轨迹数据存在摄像机提取噪声。若不平滑，差分得到的 `yaw` 和 `steer` 会剧烈抖动，导致舒适性评价失真。

### 4.5 步骤5：计算车道边界

脚本根据目标车道 `TARGET_LANES` 中车辆的横向位置统计车道中心，再根据标准车道宽度估计车道边界：

```python
TARGET_LANES = [2, 3, 4]
LANE_WIDTH = 3.66
```

输出变量：

- `map_info`
- `lane1`
- `lane2`
- `lane3`
- `lane4`

这些变量用于构造模型输入，也用于后续判断预测轨迹是否越界。

### 4.6 步骤6：逐样本开环评估

在 `run_openloop_evaluation()` 中，脚本会：

1. 只保留目标车道车辆；
2. 筛选轨迹长度大于 `MIN_TRAJECTORY_LENGTH` 的车辆；
3. 对每辆车每隔 5 帧采样一次；
4. 检查未来 5 帧是否完整；
5. 排除未来发生换道的样本；
6. 构造模型输入；
7. 模型预测未来 5 步动作；
8. 计算误差指标和综合评分。

---

## 5. NGSIM数据输入要求

### 5.1 数据文件位置

建议将 NGSIM CSV 放到：

```text
ngsim_data/
```

脚本默认读取：

```text
ngsim_data/Next_Generation_Simulation_NGSIM_Vehicle_Trajectories_and_Supporting_Data_20260418.csv
```

如果文件名不同，需要修改 `NGSIM_CSV_PATH`。

### 5.2 必要字段

建议 CSV 至少包含：

- `Vehicle_ID`
- `Frame_ID`
- `Total_Frames`
- `Global_Time`
- `Local_X`
- `Local_Y`
- `v_Length` 或 `v_length`
- `v_Width`
- `v_Vel`
- `v_Acc`
- `Lane_ID`
- `Preceding`
- `Following`
- `Space_Headway`
- `Time_Headway`

其中真正影响模型输入和评估的关键字段是：

- `Vehicle_ID`
- `Frame_ID`
- `Local_X`
- `Local_Y`
- `v_Length`
- `v_Vel`
- `v_Acc`
- `Lane_ID`

---

## 6. 如何运行NGSIM测试

### 6.1 运行前检查

运行前建议逐项检查：

1. 模型 `.pth` 是否存在；
2. NGSIM CSV 是否存在；
3. 模型超参数是否与训练一致；
4. 数据列名是否兼容；
5. `TARGET_LANES` 是否适合当前数据集；
6. GPU/CPU 环境是否正常。

### 6.2 运行命令

在项目根目录执行：

```bash
python -u test_ngsim_transformer.py
```

如果希望在服务器后台运行，可使用 `screen`：

```bash
screen -S NGSIM
python -u test_ngsim_transformer.py
```

### 6.3 正常运行时的日志阶段

正常运行会依次出现：

```text
[步骤 1/5] 加载 Transformer 模型...
[步骤 2/5] 加载并预处理 NGSIM 数据...
[步骤 3/5] 计算航向角(yaw)和转向角(steer)...
[步骤 4/5] 计算车道边界...
[步骤 5/5] 执行开环评估...
```

如果看到评估进度：

```text
已评估 500 样本
已评估 1000 样本
...
```

说明程序已经进入正式评估阶段。

---

## 7. 模型输入54维特征逻辑

这一节只保留必要理解。NGSIM 测试脚本必须把真实交通流数据整理成和训练阶段一致的输入格式：

```text
54 = 主车6维 + 8辆周围车 × 每车6维
```

主车 6 维主要描述当前速度、航向、目标点相对位置和车道边界相对位置；周围车 6 维主要描述相对位置、相对速度、航向和所在车道边界。脚本最多选择距离主车最近的 8 辆周围车，不足 8 辆时用固定占位特征补齐。

这里不需要背每一维的具体顺序，但必须记住一个原则：

> NGSIM 构造出来的输入维度、顺序、单位和语义必须与训练数据完全一致。只要特征顺序或含义错位，即使代码能运行，测试结果也没有可信度。

---

## 8. 如果更换测试模型，需要修改哪些地方

这一节是实际操作中最重要的部分。更换模型不是简单把 `.pth` 文件名换掉，而是要把**权重文件、模型结构、输入特征、输出动作、归一化规则、评估步长、报告记录**全部对齐。

可以把更换模型分成三种情况：

1. **同结构 Transformer，仅换另一个训练轮次或实验权重**：通常只需要改 `MODEL_PATH`，再核对超参数。
2. **仍是 Transformer，但训练结构或输入输出发生变化**：需要同步修改模型类、超参数、输入构造、动作反归一化和评估逻辑。
3. **换成 GRU、LSTM、vanilla RNN、AR Transformer 等其他模型**：不能直接复用当前 `TransformerTrajectoryPredictor`，必须替换模型定义和推理方式。

### 8.1 第一处：修改模型权重路径 `MODEL_PATH`

当前脚本通过 `MODEL_PATH` 指定待测试模型：

```python
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "Tf_trajectory_model_xxx.pth"
)
```

如果要测试新的 Transformer 权重，例如某个 0422 训练结果，需要把第三行文件名替换成真实存在的 `.pth` 文件名。

必须注意：

- `MODEL_PATH` 必须指向具体 `.pth` 文件，而不是目录；
- 文件名必须完整，包含 `.pth` 后缀；
- 如果权重不在 `Transformer_checkpoints/`，需要同步修改目录；
- 不要连续写两个未注释字符串，否则 Python 会自动拼接。

错误示例：

```python
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "model_a.pth"
    "model_b.pth"
)
```

Python 会把它理解成：

```text
model_a.pthmodel_b.pth
```

正确做法是只保留一个权重文件：

```python
MODEL_PATH = os.path.join(
    BASE_DIR, "Transformer_checkpoints",
    "model_b.pth"
)
```

### 8.2 第二处：核对 Transformer 结构超参数

当前脚本使用以下结构参数创建模型：

```python
D_MODEL = 256
FFN_DIM = 4 * D_MODEL
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8
```

这些值必须与训练该 `.pth` 文件时完全一致。`.pth` 保存的是参数张量，不保存完整 Python 模型定义；测试时会先按这些参数创建一个空模型，再把 `.pth` 里的参数加载进去。因此只要结构不同，就会加载失败或语义错误。

### 8.3 `D_MODEL`：隐藏维度必须一致

`D_MODEL` 是 Transformer token 的 embedding 维度。当前模型是：

```python
D_MODEL = 256
```

如果新模型训练时使用 `D_MODEL=128` 或 `D_MODEL=512`，测试脚本必须同步修改。否则通常会在加载权重时报：

```text
size mismatch for embedding_main_target.weight
size mismatch for embedding_vehicle.weight
size mismatch for encoder.layers...
```

判断方法：

- 看训练脚本中的 `D_MODEL`；
- 看训练日志文件名中是否包含 `256dmodel`、`128dmodel` 等信息；
- 看训练日志开头打印的模型配置。

### 8.4 `FFN_DIM`：前馈网络维度必须一致

`FFN_DIM` 是 Transformer encoder/decoder 内部前馈网络的隐藏层维度。当前写法是：

```python
FFN_DIM = 4 * D_MODEL
```

在当前配置下等于 1024。但更稳妥的做法是以训练时真实值为准。如果训练日志写的是：

```text
1024FFNdim
```

测试脚本就必须保证：

```python
FFN_DIM = 1024
```

不要机械地认为所有模型都是 `4 * D_MODEL`。如果某次实验使用 `D_MODEL=256` 但 `FFN_DIM=512`，测试脚本仍写 `4 * D_MODEL` 就会加载失败。

### 8.5 Encoder / Decoder 层数必须一致

当前脚本配置：

```python
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
```

如果训练权重来自 `enc3_dec3`，这里就是 3 和 3。如果训练权重来自 `enc2_dec2`、`enc4_dec4`，这里必须同步修改。

层数不一致时常见报错：

```text
Missing key(s) in state_dict
Unexpected key(s) in state_dict
```

专业判断方式：

- 文件名中常见 `enc3_dec3`；
- 训练日志中通常会记录 encoder/decoder 层数；
- 如果不确定，应回到对应训练脚本确认模型创建参数。

### 8.6 `SEQ_LENGTH` 与 `PREDICTION_HORIZON` 必须一致

当前模型输出未来 5 步：

```python
SEQ_LENGTH = 5
PREDICTION_HORIZON = 5
NGSIM_DT = 0.1
```

含义是预测未来 0.5 秒。训练标签中最后 10 维对应：

```text
5步 × 每步(acc, steer) = 10维
```

如果新模型改成预测 10 步，那么必须同时修改：

- `SEQ_LENGTH`；
- `PREDICTION_HORIZON`；
- `run_openloop_evaluation()` 中未来帧截取逻辑；
- 预测轨迹积分循环；
- ADE/FDE/RMSE 计算；
- 评分逻辑中预测碰撞、越界、速度统计的长度；
- 报告中“预测步长”和“预测时长”的文字说明。

如果只改 `SEQ_LENGTH` 不改 `PREDICTION_HORIZON`，会出现模型输出长度和真实未来轨迹长度不一致的问题。如果只改评估步长不改模型输出，则会索引越界或丢弃部分输出。

### 8.7 `CAR_NUM` 与输入维度必须一致

当前脚本最多考虑 8 辆周围车：

```python
CAR_NUM = 8
```

因此输入维度是：

```text
6 + 8 × 6 = 54
```

如果新模型训练时改为 6 辆车或 10 辆车，就必须同步修改：

- `CAR_NUM`；
- `build_model_input()` 里选择周围车的数量；
- 不足车辆时的补齐逻辑；
- `forward()` 中 `reshape(-1, self.car_num, 6)` 的语义；
- 输入维度检查；
- 报告中对输入维度的描述。

例如 `CAR_NUM=10` 时，输入维度应变为：

```text
6 + 10 × 6 = 66
```

如果只改 `CAR_NUM`，但 `build_model_input()` 仍构造 54 维，模型会 reshape 失败；如果刚好维度能对上但车辆排序规则不同，结果会更隐蔽地失真。

### 8.8 模型类定义必须与训练脚本一致

当前测试脚本中的模型类是：

```python
class TransformerTrajectoryPredictor(nn.Module):
```

它需要与训练脚本中的模型结构保持一致。需要核对的内容包括：

- `embedding_main_target` 和 `embedding_vehicle` 是否一致；
- 主车 token 和周围车 token 拼接顺序是否一致；
- encoder/decoder 是否都是 `norm_first=True`；
- 激活函数是否是 `gelu`；
- `future_queries` 是否仍用于一次性预测未来 5 步；
- 输出 head 是否仍以 `Tanh()` 结尾；
- dropout 层位置是否改变；
- 是否新增 LayerNorm、位置编码或其他模块。

如果训练脚本中模型结构改过，但测试脚本仍旧使用旧类，有两种情况：

- **严格加载失败**：参数名或形状不匹配，直接报错；
- **勉强加载或部分加载成功**：代码能跑，但输出语义错误，评估结果不可信。

不要使用 `strict=False` 来强行加载不匹配权重，除非你明确知道缺失和多余参数的原因，并在报告中说明该模型不是完整同结构评估。

### 8.9 动作反归一化参数必须与训练一致

当前脚本假设训练阶段动作标签做了如下归一化：

```python
acc_norm = acc / 6.0
steer_norm = steer / 0.15
```

所以测试时反归一化为：

```python
pred_acc = pred[:, 0] * 6.0
pred_steer = pred[:, 1] * 0.15
```

如果新模型训练时改过动作缩放，例如：

- 加速度除以 4、5 或 8；
- 转角除以 0.10、0.20；
- 标签没有归一化；
- 输出层没有 `Tanh()`；
- 输出动作顺序从 `[acc, steer]` 改成 `[steer, acc]`；

则必须同步修改 NGSIM 测试脚本。这个问题尤其危险，因为它通常不会导致程序崩溃，但会导致轨迹积分、ADE/FDE、舒适分和安全分全部失真。

例子：

训练时如果使用：

```python
acc_norm = acc / 4.0
steer_norm = steer / 0.10
```

测试时必须写成：

```python
pred_acc = pred[:, 0] * 4.0
pred_steer = pred[:, 1] * 0.10
```

### 8.10 输入特征构造逻辑是否需要重写

如果新模型仍然使用当前训练数据格式：

```text
主车6维 + 8车×6维
```

并且特征顺序、单位、坐标含义都没有变化，那么 `build_model_input()` 可以继续使用。

但只要训练时发生以下任一变化，就必须同步修改 `build_model_input()`：

- 增加历史帧，例如 `his5` / `Hist5`；
- 增加主车历史速度、历史动作或历史轨迹；
- 增加车道曲率、道路 heading、目标车道编号；
- 改变主车 6 维内部顺序；
- 改变障碍车 6 维内部顺序；
- 改变周围车排序规则，例如从最近 8 车改为前后分车道选择；
- 改变周围车占位值；
- 改变左右方向统一或坐标镜像规则；
- 输入从单帧状态改成时序张量。

必须强调：

> 输入维度一致不代表输入语义一致。

例如同样是 54 维，如果训练时第 3 维是 `goal_x - x_ego`，测试时却填成 `goal_y - y_ego`，模型仍能正常推理，但结果没有实验意义。

### 8.11 目标点构造方式是否需要调整

当前 NGSIM 测试把车辆轨迹终点作为目标点：

```python
goal_x = veh_data.iloc[-1]['Local_Y']
goal_y = veh_data.iloc[-1]['Local_X']
```

这是为了在 NGSIM 开环数据中构造一个近似的“行驶目标”。但它和 OnSite 中的任务目标区域不是完全等价的。

如果更换的模型对目标点特别敏感，或者训练时目标点定义发生过变化，需要考虑同步修改目标点构造方式，例如：

- 使用当前车道前方固定距离作为 `goal_x`；
- 使用目标车道中心作为 `goal_y`；
- 根据车辆最终 `Lane_ID` 推断目标车道；
- 对换道样本和非换道样本分别定义目标点；
- 对只做跟车的测试样本弱化目标点横向影响。

如果目标点定义不一致，常见表现是：ADE/FDE 可能仍然不算太差，但转向输出、横向误差和安全评分会异常。

### 8.12 如果换成 GRU、LSTM、RNN 或 AR Transformer

当前 `test_ngsim_transformer.py` 只适用于当前非自回归 Transformer BC 模型，不能直接把其他网络的 `.pth` 填进 `MODEL_PATH`。

#### 换成 GRU

需要修改：

- 模型类定义；
- hidden size；
- layer num；
- dropout；
- checkpoint 目录；
- `forward()` 输入形状；
- 是否需要显式构造时间维；
- 输出层结构；
- 动作反归一化。

#### 换成 LSTM

需要修改：

- LSTM 模型定义；
- hidden state / cell state 初始化；
- 输入张量 shape；
- layer num 和 hidden dim；
- checkpoint 路径；
- 推理输出解析。

#### 换成 vanilla RNN

需要修改：

- RNN 模型类；
- 输入是否为单步或序列；
- hidden state 初始化；
- 输出层和归一化逻辑。

#### 换成 AR Transformer

如果测试自回归 Transformer，需要重点修改：

- 模型类；
- decoder 输入；
- 起始 token 或上一时刻动作输入；
- 推理方式从“一次性输出 5 步”改为“逐步生成未来动作”；
- 误差与评分统计仍可复用，但预测结果组装方式要改。

### 8.13 输出目录和报告命名建议同步修改

为了避免多个模型结果混在一起，建议在测试不同模型时修改输出目录或实验名。

当前写法：

```python
SAVE_DIR = os.path.join(BASE_DIR, "ngsim_results")
```

建议改成带实验名的结构：

```python
EXP_NAME = "Tf_0422_1915"
SAVE_DIR = os.path.join(BASE_DIR, "ngsim_results", EXP_NAME + "_" + TIMESTAMP)
```

这样每个模型会生成独立目录，里面包含：

- `.log` 运行日志；
- `.txt` 测试报告；
- `.svg` 可视化图。

如果暂时不改目录结构，也至少要保证 `.txt` 报告里完整记录：

- `MODEL_PATH`；
- `D_MODEL`；
- `FFN_DIM`；
- encoder/decoder 层数；
- `SEQ_LENGTH`；
- `CAR_NUM`；
- 动作缩放参数；
- NGSIM 数据文件名。

### 8.14 更换模型后的最小验证流程

每次换模型后，建议不要直接跑完整大样本，而是按以下顺序验证：

1. 先把 `MAX_EVAL_SAMPLES` 暂时设小，例如 50 或 100；
2. 运行脚本，确认模型能成功加载；
3. 检查日志中的参数量是否合理；
4. 检查输入维度是否仍为预期值；
5. 检查 `pred_acc` 和 `pred_steer` 是否在合理范围；
6. 检查 ADE/FDE 是否不是 NaN 或极端离谱；
7. 检查安全、效率、舒适分是否能正常输出；
8. 再恢复到 5000 或全部样本进行正式评估。

这样可以避免因为路径、结构、归一化或输入语义错误，浪费大量时间跑出无效报告。

### 8.15 更换模型前最终核对清单

正式运行前建议逐项核对：

```text
[ ] MODEL_PATH 是否指向正确 .pth 文件
[ ] 权重文件是否来自当前测试脚本支持的模型类型
[ ] D_MODEL 是否与训练一致
[ ] FFN_DIM 是否与训练一致
[ ] NUM_ENCODER_LAYERS 是否与训练一致
[ ] NUM_DECODER_LAYERS 是否与训练一致
[ ] SEQ_LENGTH 是否与训练一致
[ ] PREDICTION_HORIZON 是否与 SEQ_LENGTH 一致
[ ] CAR_NUM 是否与训练一致
[ ] 输入维度是否与训练一致
[ ] 输入特征顺序和语义是否与训练一致
[ ] 目标点构造方式是否适合该模型
[ ] 动作输出顺序是否仍为 [acc, steer]
[ ] 动作反归一化比例是否与训练一致
[ ] 模型类定义是否与训练脚本一致
[ ] 是否误把 GRU/LSTM/RNN 权重放进 Transformer 测试脚本
[ ] SAVE_DIR 或报告中是否能区分不同模型结果
[ ] 小样本试跑是否通过
```

---

## 9. 模型输出与轨迹积分逻辑

当前模型输出：

```text
(5, 2)
```

含义：

- 未来 5 个时间步；
- 每步输出 2 个控制量；
- 第 1 维是加速度；
- 第 2 维是转向角。

模型输出经过 `Tanh()`，所以范围约为 `[-1, 1]`。测试时需要反归一化：

```python
pred_acc = pred[:, 0] * 6.0
pred_steer = pred[:, 1] * 0.15
```

随后脚本使用简化车辆运动学进行积分：

```python
cur_v = max(0.0, cur_v + pred_acc[step] * NGSIM_DT)
cur_yaw = cur_yaw + pred_steer[step] * NGSIM_DT
ego_x = ego_x + cur_v * np.cos(cur_yaw) * NGSIM_DT
ego_y = ego_y + cur_v * np.sin(cur_yaw) * NGSIM_DT
```

这里要注意，当前转向角积分方式是简化近似，并不是完整自行车模型。它适合做短时 0.5 秒开环评估，但不应过度解释为精确动力学仿真。

---

## 10. 原始误差指标解释

当前 NGSIM 脚本会输出以下基础指标。

### 10.1 ADE

ADE 是 Average Displacement Error，平均位移误差。

含义：

```text
未来5步预测轨迹与真实轨迹之间的平均欧式距离
```

单位是米。越小越好。

ADE 低说明模型整体预测轨迹接近真实车辆行为。

### 10.2 FDE

FDE 是 Final Displacement Error，终点位移误差。

含义：

```text
第5步预测位置与第5步真实位置之间的欧式距离
```

单位是米。越小越好。

FDE 更关注预测末端误差，能体现误差是否随时间累积。

### 10.3 RMSE_acc

预测加速度与真实加速度之间的均方根误差。

它体现模型纵向控制能力。如果该指标高，说明模型加减速策略与真实驾驶差异较大。

### 10.4 RMSE_steer

预测转向角与估计真实转向角之间的均方根误差。

它体现横向控制能力。需要注意，NGSIM 的真实转向角不是原始字段，而是由轨迹反推得到，因此会受轨迹噪声和平滑参数影响。

### 10.5 RMSE_lat 和 RMSE_lon

- `RMSE_lat`：横向位置误差；
- `RMSE_lon`：纵向位置误差。

横向误差更接近车道保持能力，纵向误差更接近速度控制和跟车能力。

### 10.6 推理耗时

包括：

- 平均推理耗时；
- P50 推理耗时；
- P95 推理耗时。

这部分用于判断模型是否具备实时部署潜力。

---

## 11. OnSite风格100分评分体系设计

为了使 NGSIM 测试结果与 OnSite、highway-env 的报告口径统一，建议在 NGSIM 测试中加入：

```text
安全性 Safety：50分
效率性 Efficiency：30分
舒适性 Comfort：20分
总分 Total：100分
```

但需要明确：

> NGSIM 是开环数据集，没有真实仿真碰撞结果。因此这里的安全、效率、舒适分是根据预测轨迹和真实交通流构造出来的代理评分。

### 11.1 安全性评分

安全性建议满分 50。

主要扣分项：

1. 预测轨迹越界；
2. 预测轨迹与周围车辆距离过近；
3. 预测碰撞；
4. 预测 TTC 过低。

基础逻辑：

```text
初始安全分 = 50
如果预测轨迹越出 lane1/lane4，扣分
如果预测轨迹与周围车未来位置重叠或过近，扣分
如果最小距离低于碰撞阈值，视为预测碰撞
最终安全分限制在 [0, 50]
```

可输出统计：

- 平均安全分；
- 预测碰撞率；
- 预测越界率；
- 最小 TTC；
- 最小车距均值。

### 11.2 效率性评分

效率性建议满分 30。

NGSIM 中没有任务完成时间，所以效率不能直接用 OnSite 的“是否快速到达终点”来算。建议用真实交通流速度作为参考。

主要扣分项：

1. 预测速度明显低于真实速度；
2. 预测速度异常过高；
3. 纵向误差过大；
4. 过度保守导致低效率。

基础逻辑：

```text
初始效率分 = 30
比较预测速度与真实未来速度
速度过低扣分
速度过高扣分
纵向误差过大扣分
最终效率分限制在 [0, 30]
```

可输出统计：

- 平均效率分；
- 平均预测速度；
- 平均真实速度；
- 速度误差；
- 纵向误差。

### 11.3 舒适性评分

舒适性建议满分 20。

主要扣分项：

1. 预测加速度绝对值过大；
2. 急刹车；
3. 预测转向角过大；
4. 加速度变化率过大；
5. 转向变化率过大。

基础逻辑：

```text
初始舒适分 = 20
大加速度扣分
急刹车扣分
大转角扣分
jerk过大扣分
steer rate过大扣分
最终舒适分限制在 [0, 20]
```

可输出统计：

- 平均舒适分；
- 最大加速度；
- 最大减速度；
- 最大转向角；
- 平均 jerk；
- 平均 steer rate。

### 11.4 总分

总分计算：

```text
Total = Safety + Efficiency + Comfort
```

输出格式建议：

```text
平均总分:   xx.xx / 100
平均安全分: xx.xx / 50
平均效率分: xx.xx / 30
平均舒适分: xx.xx / 20
```

这样可以与 OnSite 和 highway-env 的报告形成统一表达。

---

## 12. 输出文件说明

每次运行 `test_ngsim_transformer.py` 后，会在 `ngsim_results/` 中生成文件。

### 12.1 `.log`日志文件

示例：

```text
ngsim_test_0423_180548.log
```

作用：

- 记录运行过程；
- 记录模型路径；
- 记录数据路径；
- 记录数据加载与清洗；
- 记录有效样本数；
- 记录评估进度；
- 用于排错。

如果程序中途失败，通常只有 `.log` 文件，没有完整 `.txt` 和 `.svg`。

### 12.2 `.txt`结果报告

示例：

```text
ngsim_test_0423_180548.txt
```

作用：

- 这是最重要的结果文件；
- 用于记录最终指标；
- 可直接放入实验记录、论文附录或答辩材料中整理。

建议报告内容包括：

- 测试配置；
- 模型参数；
- 数据集信息；
- 有效样本数量；
- 跳过样本统计；
- ADE/FDE/RMSE；
- 安全性、效率性、舒适性评分；
- 预测碰撞率；
- 越界率；
- 推理耗时；
- 指标解释。

### 12.3 `.svg`图表

示例：

```text
ngsim_test_0423_180548.svg
```

当前图表通常包含：

- ADE 分布；
- FDE 分布；
- 加速度 RMSE 分布；
- 转向角 RMSE 分布。

后续可扩展加入：

- Safety 分布；
- Efficiency 分布；
- Comfort 分布；
- Total score 分布；
- 预测碰撞样本可视化；
- 最差样本轨迹对比图。

---

## 13. 建议升级后的报告格式

参考 highway-env v2 报告，NGSIM 报告建议升级为以下结构：

```text
================================================================================
NGSIM Transformer BC 真实数据集泛化测试报告
生成时间:
模型:
设备:
数据集:
目标车道:
最大评估样本:
================================================================================

总评估样本数:
有效样本数:
跳过 NaN:
跳过未来帧不足:
跳过换道样本:

平均总分:   xx.xx / 100
平均安全分: xx.xx / 50
平均效率分: xx.xx / 30
平均舒适分: xx.xx / 20

ADE:
FDE:
RMSE_acc:
RMSE_steer:
RMSE_lat:
RMSE_lon:

预测碰撞率:
预测越界率:
平均预测速度:
平均真实速度:
推理耗时:
```

建议增加分车道汇总：

```text
--------------------------------------------------------------------------------
分车道汇总:
Lane    样本数    Safety    Effic    Comfort    Total    ADE    FDE
--------------------------------------------------------------------------------
2       ...
3       ...
4       ...
--------------------------------------------------------------------------------
```

建议增加代表性样本标记：

```text
代表性样本标记:
  ★ [最佳] Vehicle_ID xx Frame xx | S=50.0 E=30.0 C=20.0 T=100.0
  ★ [最差] Vehicle_ID xx Frame xx | S=... E=... C=... T=...
  ★ [高风险] Vehicle_ID xx Frame xx | 预测碰撞 / 越界 / 低TTC
```

这会使 NGSIM 报告与 highway-env v2 报告保持风格统一，也更适合写进毕设、论文和项目答辩材料。

---

## 14. 结果如何解读

### 14.1 如果ADE/FDE低，总分也高

说明模型在真实高速轨迹数据上具有较好的泛化能力。

可以说明：

- 模型预测轨迹接近真实驾驶轨迹；
- 安全性风险较低；
- 速度控制没有明显过保守；
- 控制输出较平滑。

这是最理想结果。

### 14.2 如果ADE/FDE低，但安全分低

说明模型整体轨迹误差不大，但局部存在高风险行为。

可能原因：

- 预测轨迹距离周围车辆过近；
- 横向误差虽然不大，但刚好靠近车道线或邻车；
- 短时动作有危险尖峰；
- 误差指标无法完全反映安全约束。

此时应重点看预测碰撞率、越界率和最差样本。

### 14.3 如果ADE/FDE高，但安全分不低

说明模型预测与真实驾驶轨迹有差异，但不一定危险。

可能原因：

- 模型比真实驾驶更保守；
- 真实驾驶存在换道或激进行为；
- 当前目标点设定与真实意图不完全一致；
- NGSIM 与 OnSite 数据分布差异较大。

这类结果需要结合效率分和纵向误差判断。

### 14.4 如果效率分低

说明模型预测速度偏离真实交通流。

可能表现：

- 预测速度偏低，模型过度保守；
- 预测速度偏高，模型不符合真实车流节奏；
- 纵向控制误差大。

如果效率分低但安全分高，可能说明模型倾向于保守驾驶。

### 14.5 如果舒适分低

说明模型控制输出不平滑。

可能表现：

- 急加速；
- 急刹车；
- 大转角；
- 转向抖动；
- 加速度变化率过大。

这通常说明模型对真实数据噪声的鲁棒性不足，或者动作平滑约束不足。

---

## 15. 常见问题与排错

### 15.1 找不到模型文件

报错通常类似：

```text
找不到模型文件: xxx.pth
```

检查：

- `MODEL_PATH` 是否写错；
- `.pth` 是否真的存在；
- 文件名是否多写或少写；
- 是否误把两个字符串拼接在一起。

### 15.2 找不到NGSIM CSV

检查：

- CSV 是否放在 `ngsim_data/`；
- 文件名是否与 `NGSIM_CSV_PATH` 一致；
- 是否使用了压缩文件而不是解压后的 `.csv`。

### 15.3 `TypeError: can't multiply sequence by non-int of type 'float'`

原因：

- 数值列被读成字符串；
- CSV 中存在脏数据或混合类型。

解决：

- 使用 `pd.to_numeric(..., errors='coerce')`；
- 删除关键字段为 NaN 的行；
- 设置 `low_memory=False` 也可减少 dtype 警告。

### 15.4 `KeyError: v_Length`

原因：

- CSV 中列名是 `v_length`；
- 代码中使用 `v_Length`。

解决：

```python
if 'v_length' in df.columns and 'v_Length' not in df.columns:
    df.rename(columns={'v_length': 'v_Length'}, inplace=True)
```

### 15.5 `load_state_dict`参数不匹配

原因：

- 模型结构参数与训练时不同；
- 测试脚本模型类与训练脚本模型类不同；
- 用 Transformer 脚本加载了 GRU/LSTM/RNN 权重。

解决：

- 核对训练日志；
- 核对 checkpoint 文件名；
- 核对模型类；
- 核对 `D_MODEL`、`FFN_DIM`、层数等。

### 15.6 有效样本数为0

可能原因：

- `TARGET_LANES` 设置不适合当前数据集；
- `MIN_TRAJECTORY_LENGTH` 太大；
- 未来 5 帧不连续；
- 过滤换道样本过严；
- `Lane_ID` 数据类型不正确。

解决：

- 打印 Lane_ID 分布；
- 降低 `MIN_TRAJECTORY_LENGTH`；
- 检查 `Frame_ID` 是否连续；
- 检查 CSV 是否来自正确路段。

### 15.7 评估速度很慢

原因：

- NGSIM 数据量很大；
- 当前脚本按车辆和帧循环；
- 周围车辆查找消耗较高；
- 每个样本逐次模型推理。

优化方向：

- 降低 `MAX_EVAL_SAMPLES`；
- 缓存 frame group；
- 批量推理；
- 只测试目标时间段；
- 只测试部分车辆。

---

## 16. 推荐实验记录方式

每次 NGSIM 测试建议记录：

```text
实验编号:
模型文件:
训练脚本:
训练数据:
训练超参数:
NGSIM CSV:
TARGET_LANES:
MAX_EVAL_SAMPLES:
ADE:
FDE:
RMSE_acc:
RMSE_steer:
Safety:
Efficiency:
Comfort:
Total:
备注:
```

如果后续同时比较 Transformer、GRU、LSTM、RNN，建议建立统一表格记录不同模型的：

- OnSite 分数；
- highway-env 分数；
- NGSIM 分数；
- ADE/FDE；
- 推理耗时。

这样可以形成一条完整实验链：

```text
训练损失 → OnSite闭环能力 → highway-env泛化能力 → NGSIM真实数据泛化能力
```

---

## 17. 总结

NGSIM 测试是本项目泛化能力验证中非常重要的一环。它的核心价值不在于替代 OnSite 官方闭环评分，而在于将模型放到真实高速公路轨迹数据上，检查它是否仍能输出合理的短时控制动作和轨迹预测。

当前 `test_ngsim_transformer.py` 已经具备以下能力：

- 读取真实 NGSIM CSV；
- 清洗混合类型脏数据；
- 兼容 `v_length` 与 `v_Length` 列名；
- 将英制单位转换为国际单位；
- 构造与训练阶段一致的模型输入；
- 加载 Transformer `.pth` 权重；
- 预测未来 5 步加速度和转向角；
- 积分得到预测轨迹；
- 输出 ADE、FDE、RMSE 等开环误差指标；
- 生成 `.log`、`.txt`、`.svg` 结果文件。

后续最推荐的优化方向是：

1. 在 NGSIM 报告中加入 OnSite 风格 100 分评分；
2. 输出安全、效率、舒适三个子分；
3. 增加分车道汇总；
4. 增加代表性高风险样本；
5. 将结果报告格式对齐 `highway_env_results_v2`；
6. 在更换模型时严格同步模型结构、动作归一化和输入特征语义。

如果这些优化完成，NGSIM 测试就可以成为本项目中证明 Transformer 行为克隆模型泛化能力的关键实验章节。
