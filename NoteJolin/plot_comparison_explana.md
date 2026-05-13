## 改图例的位置
你当前 gap 图这里是：

```python
ax_gap.legend(fontsize=11, loc='upper right', frameon=False)
```

改成：

```python
ax_gap.legend(fontsize=11, loc='lower right', frameon=False)
```

如果右下角仍然压住曲线，可以进一步让图例半透明或放到图外：

```python
ax_gap.legend(fontsize=11, loc='lower right', frameon=True, framealpha=0.85)
```

或者图外右侧：

```python
ax_gap.legend(
    fontsize=11,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
)
plt.tight_layout()
```

但你现在的需求是移动到右下角，所以优先用第一种。

## 两张图的整体思路

你现在画的是两类图：

第一张图：`All_Exps_Loss_Comparison_Clean.svg`

它包含两个子图：

- 左图：7 组实验的 `train loss`
- 右图：7 组实验的 `val loss`

这张图回答的问题是：

> 不同实验配置下，模型训练损失和验证损失分别如何变化？哪个模型最终 loss 更低？哪个模型收敛更快？

第二张图：`Train_Val_Loss_Gap_Comparison.svg`

它画的是：

```python
gap = val_loss - train_loss
```

这张图回答的问题是：

> 每个模型训练集和验证集之间的差距有多大？是否出现明显过拟合？dropout=0.3 的泛化差距是否更稳定？

如果你之后改成绝对值：

```python
gap = [abs(v - t) for t, v in zip(t_loss, v_loss)]
```

那它回答的问题就变成：

> train loss 和 val loss 的接近程度如何？哪个模型泛化差距最小？

## 代码逐步解释

### 1. 实验文件映射

你的 `log_files` 本质上是一个“实验名称 -> 日志路径”的字典：

```python
log_files = {
    "Exp-6:Dropout=0.30": "../train_log/xxx.txt",
}
```

作用是告诉程序：

- 图例上显示什么名字；
- 到哪个日志文件里读取 loss；
- 各实验的显示顺序。

这里的顺序很重要。Python 字典会保持插入顺序，所以你把 Exp-1 到 Exp-7 按顺序写进去，图例和绘图顺序也会按这个顺序来。

### 2. 从日志中解析 loss

函数：

```python
def parse_log_file(filepath):
```

会逐行读取日志文件，然后用正则表达式匹配这一类内容：

```text
Epoch [1/512] | 训练损失: 0.171930 | 验证损失: 0.129931 | 学习率: 0.000500
```

核心正则是：

```python
pattern = re.compile(r"训练损失:\s*([\d.]+)\s*\|\s*验证损失:\s*([\d.]+)")
```

它会提取两个数字：

- 第一个：训练损失 `train loss`
- 第二个：验证损失 `val loss`

然后分别放进：

```python
train_losses.append(...)
val_losses.append(...)
```

最后返回：

```python
return train_losses, val_losses
```

所以每个实验都会得到两个列表，例如：

```python
train = [0.171930, 0.134334, 0.124411, ...]
val = [0.129931, 0.118093, 0.120925, ...]
```

### 3. 保存所有实验数据

这一段：

```python
data = {}
for exp_name, filepath in log_files.items():
    if filepath:
        t_loss, v_loss = parse_log_file(filepath)
        if t_loss:
            data[exp_name] = {"train": t_loss, "val": v_loss}
```

会把所有实验整理成统一结构：

```python
data = {
    "Exp-1:Underfitting": {
        "train": [...],
        "val": [...],
    },
    "Exp-6:Dropout=0.30": {
        "train": [...],
        "val": [...],
    },
}
```

后面两张图都复用这个 `data`，所以只要前面数据读对了，后面可以画任意派生图。

### 4. 第一张图：train loss / val loss 对比

第一张图通过：

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
```

创建一行两列：

- `ax1`：左边图，画 train loss；
- `ax2`：右边图，画 val loss。

每个实验进入循环：

```python
for exp_name, losses in data.items():
    t_loss = losses["train"][:200]
    v_loss = losses["val"][:200]
    epochs = range(1, len(t_loss) + 1)
```

这里 `[:200]` 的意思是只取前 200 轮。比如原始有 300 轮，也只画第 1 到 200 轮。

然后判断是否突出 dropout=0.3：

```python
if "Dropout=0.30" in exp_name:
    color = '#E63946'
    linewidth = 3.2
    alpha = 1.0
    zorder = 10
```

含义是：

- `color`：红色；
- `linewidth`：更粗；
- `alpha`：不透明；
- `zorder=10`：画在最上层，避免被其他线盖住。

普通实验则用浅色、细线：

```python
linewidth = 1.5
alpha = 0.8
zorder = 1
```

最后分别画到两个子图：

```python
ax1.plot(epochs, t_loss, ...)
ax2.plot(epochs, v_loss, ...)
```

### 5. 第一张图的美化

这部分控制坐标轴样式：

```python
ax.set_xlim(-6, 200)
```

表示横轴从 `-6` 到 `200`。虽然 epoch 从 1 开始，但左侧留出一点空白，视觉上不会贴边。

```python
ax.set_ylim(0.01, 0.15)
```

限制纵轴范围，让图聚焦在主要 loss 区间。如果 Exp-7 开头 loss 超过 0.15，会被截掉，但后期趋势会更清楚。

```python
ax.grid(False)
ax.set_facecolor('white')
```

关闭网格，底色纯白。

```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

隐藏上边框和右边框，让图更干净。

```python
ax.tick_params(axis='both', direction='out', labelsize=12)
```

刻度线向外，字号为 12。

### 6. 第二张图：train-val gap 对比

第二张图是单独一张图：

```python
fig_gap, ax_gap = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
```

它不是左右子图，而是只有一个坐标轴 `ax_gap`。

核心计算是：

```python
gap = [v - t for t, v in zip(t_loss, v_loss)]
```

含义是每一轮：

```text
gap(epoch) = val_loss(epoch) - train_loss(epoch)
```

如果你想画绝对差距，就改成：

```python
gap = [abs(v - t) for t, v in zip(t_loss, v_loss)]
```

这张图里还有一条参考线：

```python
ax_gap.axhline(y=0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
```

它表示 `val_loss = train_loss` 的位置。

如果 gap 曲线在 0 上方，说明验证损失高于训练损失；如果长期越来越高，通常说明泛化差距变大。

### 7. 修改 gap 图图例位置

你当前这里：

```python
ax_gap.legend(fontsize=11, loc='upper right', frameon=False)
```

改成：

```python
ax_gap.legend(fontsize=11, loc='lower right', frameon=False)
```

如果使用绝对值 gap，建议同时改标题和纵轴：

```python
ax_gap.set_title('Absolute Train-Val Loss Gap Comparison', fontsize=16, fontweight='bold')
ax_gap.set_ylabel('|Validation Loss - Training Loss|', fontsize=14)
```

这样图的含义会更准确。