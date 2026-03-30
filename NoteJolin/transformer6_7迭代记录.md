 # `get_clone_learning_Transformer6_7.py`

 ## - d_model = 256,Batch_size = 1024, ffn_dim = 1024,LEARNING_RATE = 5e-4,early_stopping_PATIENCE = 40

## 一、 3.26前：`线性预热（Linear Warmup） + 单次余弦衰减（Cosine Annealing）`

你目前使用的是一种非常经典的深度学习调度策略：**线性预热（Linear Warmup） + 单次余弦衰减（Cosine Annealing）**。

具体代码（第387-403行）：
```python
num_warmup_epochs = max(1, int(0.05 * EPOCHS))  # 5%的预热期，即 512 * 0.05 ≈ 25轮

def lr_lambda(current_epoch: int):
    if current_epoch < num_warmup_epochs:
        # 线性 warmup：从 0 -> 1
        return float(current_epoch + 1) / float(num_warmup_epochs)
    # 余弦衰减：从 1 -> 0
    progress = float(current_epoch - num_warmup_epochs) / max(1, EPOCHS - num_warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

**这个策略的运行轨迹是这样的：**
1. **预热期（第1~25轮）**：学习率从接近 `0` 线性爬升到你设置的最大值 `LEARNING_RATE = 5e-4`。
2. **衰减期（第26~512轮）**：学习率按照余弦曲线（像半个钟罩一样），从 `5e-4` 极其平滑、缓慢地下降，直到第512轮时降为 `0`。

**3.25断电重启后,效果变好了:**
    当模型跑到约270轮时，余弦曲线已经走过了一半，此时的学习率已经衰减得非常小了（模型陷入了某个局部最优解的坑底）。
因为设备关机，你重新运行了代码。代码虽然加载了第270轮保存的**模型权重（Model Weights）**，但**优化器和调度器的状态（Epoch计数）被清零了**！
于是，代码又从第1轮开始算起，触发了前25轮的“线性预热”，学习率在短短25轮内迅速飙升回了 `5e-4`。这个巨大的学习率突变，直接把模型从原来的坑底“踹”了出来，由于模型本身已经具备了很好的特征提取能力（继承了270轮的权重），它很快就在高学习率的震荡下找到了一个更平坦、泛化能力更好的最优解（Loss从0.044降到0.040）。

---

## 二、 3.26计划修改为修改为“余弦退火伴随热重启（SGDR）”：

 `CosineAnnealingWarmRestarts` 

为了将这种“断电重启”的红利固化到代码逻辑中，我们可以直接使用 PyTorch 内置的 `CosineAnnealingWarmRestarts`。

#### 具体的代码修改步骤：

**1. 替换调度器定义**
找到代码的第387行到第403行（即 `=== 学习率调度：epoch 级 warmup + cosine ===` 这一大段），将它们**全部删除或注释掉（即之前的代码，之前的线性预热+余弦衰减）**，替换为以下代码：

```python
    # === 替换为：余弦退火伴随热重启 (Cosine Annealing Warm Restarts) ===
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    # T_0: 第一次重启发生前的Epoch数（比如设为 50 或 64）
    # T_mult: 每次重启后，周期的放大倍数（设为 2，则周期依次为 50, 100, 200...）
    # eta_min: 学习率衰减的下限（不能完全为0，给一个很小的值即可）
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=64,           # 建议值：64轮进行第一次重启
        T_mult=2,         # 建议值：周期翻倍
        eta_min=1e-6      # 学习率最小降到 1e-6
    )
```

**参数设计思路（以 `EPOCHS=512` 为例）：**
* 第1个周期：1 ~ 64 轮（学习率从 5e-4 降到 1e-6）
* 第64轮结束时：**瞬间重启**，学习率突变回 5e-4！
* 第2个周期：65 ~ 256 轮（长度为 64*2=128 轮，学习率再次缓慢下降）
* 第256轮结束时：**再次瞬间重启**，学习率突变回 5e-4！
* 第3个周期：257 ~ 768 轮（长度为 256 轮，覆盖到你设定的最大512轮）

这种设置完美模拟了你“跑了200多轮后突然重置学习率”的过程，而且是有规律、周期性地去寻找更平坦的最优解。


### 0326
- 设置了early_stopping = 40 +  `CosineAnnealingWarmRestarts` ，太小了，导致他在64轮时train loss和val loss均突增。在 102 轮就结束了，且最优模型是epoch 64
- ` /root/autodl-tmp/BCRL/bc/train_log_0326TF_1024BSIZE_256DMODEL_1024FFNdim_40ESTOP_CoAnWarmRest_zDATASET:es40太小没处理好fail.txt`
- `/root/autodl-tmp/BCRL/bc/Transformer_plots/Tf_loss_curve_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_40es_CoAnWarmRest_zDATASET0326.svg`
- /`root/autodl-tmp/BCRL/bc/Transformer_checkpoints/Tf_trajectory_model_1024BSIZE_256dmodel_1024FFNdim_enc3_dec3_40es_CoAnWarmRest_zDATASET0326.pth`

