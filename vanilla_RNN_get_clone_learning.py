import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
import math
from datetime import datetime

# ======================== 全局配置 ========================
torch.manual_seed(2025)
np.random.seed(2025)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 数据路径（与已有代码保持一致）
NPY_FILE_PATHS = [
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_0.npy"),
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_1.npy"),
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_out.npy"),
]

# ======================== 模型超参数 ========================
EMBED_DIM   = 128   # 嵌入维度
HIDDEN_DIM  = 256   # RNN 隐藏层维度
NUM_LAYERS  = 2     # RNN 层数（vanilla RNN 梯度更容易消失，层数不宜过深）
OUTPUT_DIM  = 2     # 输出维度（加速度 + 转角）
SEQ_LENGTH  = 5     # 预测序列长度
CAR_NUM     = 8     # 障碍车数量
DROPOUT     = 0.2   # dropout 概率

# ======================== 训练超参数 ========================
LEARNING_RATE          = 5e-4
BATCH_SIZE             = 1024
EPOCHS                 = 512
EARLY_STOPPING_PATIENCE = 20
ACCELERATION_WEIGHT    = 1
STEERING_WEIGHT        = 5
TEST_SIZE              = 0.1

# ======================== 输出路径 ========================
checkpoints_dir = os.path.join(BASE_DIR, "RNN_checkpoints")
plots_dir       = os.path.join(BASE_DIR, "RNN_plots")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

RUN_DATE_TIME = datetime.now().strftime("%m%d")
_tag = f"{RUN_DATE_TIME}_{BATCH_SIZE}BSIZE_{HIDDEN_DIM}HDdim_{NUM_LAYERS}layers_{EARLY_STOPPING_PATIENCE}es_CoAnWarmRest_zDATASET"
MODEL_SAVE_PATH     = os.path.join(checkpoints_dir, f"rnn_trajectory_model_{_tag}.pth")
LOSS_PLOT_SAVE_PATH = os.path.join(plots_dir,       f"rnn_loss_curve_{_tag}.svg")


# ======================== 模型定义（Vanilla RNN 编码器-解码器）========================
class VanillaRNNTrajectoryPredictor(nn.Module):
    """
    以 Vanilla RNN（nn.RNN）为骨干的轨迹预测器。
    结构：
      Embedding → tanh → RNN Encoder → RNN Decoder（自回归展开）→ FC Head → Tanh
    与 GRU 版本的核心区别：
      - 使用 nn.RNN（tanh 门控）替代 nn.GRU
      - 梯度更容易消失，因此建议层数 ≤ 2，并搭配梯度裁剪
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_length: int,
        car_num: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        nonlinearity: str = 'tanh',   # 'tanh' 或 'relu'
    ):
        super().__init__()
        self.car_num    = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # --- 嵌入层（与 GRU / Transformer 版对齐，分别处理主车和障碍车）---
        self.embedding_main_target = nn.Linear(6, embed_dim)
        self.embedding_vehicle     = nn.Linear(6, embed_dim)
        self.dropout_embed         = nn.Dropout(dropout)

        # --- Vanilla RNN 编码器 ---
        # nn.RNN dropout 参数只在 num_layers > 1 时层间生效
        self.encoder_rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity=nonlinearity,
        )
        self.dropout_encoder = nn.Dropout(dropout)

        # --- Vanilla RNN 解码器 ---
        self.decoder_rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity=nonlinearity,
        )

        # --- 输出头（与 Transformer 版相同的双层 FC + Tanh）---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),   # 输出归一化到 [-1, 1]，与标签一致
        )

    def forward(self, x):
        """
        x: (batch_size, 6 + car_num * 6)
           前 6 维：主车 + 目标点特征
           后 car_num*6 维：障碍车特征
        返回: (batch_size, seq_length, output_dim)
        """
        B = x.size(0)

        # 1. 拆分输入
        main_target_feat = x[:, 0:6]                                          # (B, 6)
        vehicle_feats    = x[:, 6:6 + self.car_num * 6].reshape(B, self.car_num, 6)  # (B, 8, 6)

        # 2. 随机打乱障碍车顺序（增强位置不变性，对 RNN 尤其重要）
        rand_idx     = torch.randperm(self.car_num, device=x.device)
        vehicle_feats = vehicle_feats[:, rand_idx, :]

        # 3. 嵌入 + dropout
        main_emb = self.dropout_embed(
            self.embedding_main_target(main_target_feat).unsqueeze(1)   # (B, 1, embed_dim)
        )
        veh_emb  = self.dropout_embed(
            self.embedding_vehicle(vehicle_feats)                        # (B, 8, embed_dim)
        )

        # 4. 拼接序列：[障碍车×8, 主车×1] → (B, 9, embed_dim)
        seq_input = torch.cat([veh_emb, main_emb], dim=1)
        seq_input = torch.tanh(seq_input)   # 与 GRU/Transformer 版保持一致

        # 5. RNN 编码器
        enc_out, hidden = self.encoder_rnn(seq_input)     # enc_out: (B, 9, hidden_dim)
        enc_out = self.dropout_encoder(enc_out)
        hidden  = self.dropout_encoder(hidden)            # hidden: (num_layers, B, hidden_dim)

        # 6. RNN 解码器（自回归展开，共 seq_length 步）
        #    初始输入：编码器最后一个时间步的输出
        decoder_input = enc_out[:, -1, :].unsqueeze(1)   # (B, 1, hidden_dim)
        pred_seq = torch.zeros(B, self.seq_length, self.output_dim, device=x.device)

        for t in range(self.seq_length):
            dec_out, hidden = self.decoder_rnn(decoder_input, hidden)  # (B, 1, hidden_dim)
            decoder_input   = dec_out                                   # 下一步以当前输出为输入
            pred_seq[:, t, :] = self.head(dec_out).squeeze(1)          # (B, output_dim)

        return pred_seq  # (B, seq_length, output_dim)


# ======================== 数据加载与预处理 ========================
def load_and_merge_npy(file_paths):
    dataset_list = []
    for path in file_paths:
        try:
            data = np.load(path)
            dataset_list.append(data)
            print(f"成功加载: {path} | shape: {data.shape}")
        except Exception as e:
            print(f"加载失败 {path}: {e}")
    if not dataset_list:
        raise ValueError("未加载到任何有效 npy 文件！")
    merged = np.concatenate(dataset_list, axis=0)
    print(f"\n合并完成 | 总量: {merged.shape[0]} | 特征维度: {merged.shape[1]}")
    return merged


def preprocess_data(dataset):
    states  = dataset[:, :-10]
    actions = dataset[:, -10:].copy()
    actions[:, 0:10:2] = np.clip(actions[:, 0:10:2] / 6,    -1, 1)   # 加速度
    actions[:, 1:10:2] = np.clip(actions[:, 1:10:2] / 0.15, -1, 1)   # 转角
    print(f"预处理完成 | states: {states.shape} | actions: {actions.shape}")
    return states, actions


# ======================== 训练与验证 ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    train_losses, val_losses = [], []
    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # scheduler 在 epoch 开头 step（与 Transformer 版保持一致）
        scheduler.step()

        # ---- 训练 ----
        model.train()
        train_epoch_loss = 0.0

        for batch_states, batch_actions in tqdm(train_loader, leave=False):
            batch_states   = batch_states.to(device)
            batch_actions  = batch_actions.to(device)

            outputs = model(batch_states).reshape(-1, 10)

            weighted_outputs = outputs       * weights
            weighted_actions = batch_actions * weights
            loss = criterion(weighted_outputs, weighted_actions)

            optimizer.zero_grad()
            loss.backward()
            # Vanilla RNN 梯度消失/爆炸比 GRU 更严重，裁剪阈值可适当收紧
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- 验证 ----
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for val_states, val_actions in val_loader:
                val_states  = val_states.to(device)
                val_actions = val_actions.to(device)

                val_out = model(val_states).reshape(-1, 10)
                val_loss = criterion(val_out * weights, val_actions * weights)
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"训练损失: {avg_train_loss:.6f} | "
            f"验证损失: {avg_val_loss:.6f} | "
            f"学习率: {scheduler.get_last_lr()[0]:.6f}"
        )
        plot_loss_curve(train_losses, val_losses, LOSS_PLOT_SAVE_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"验证损失下降，保存最优模型 → {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！连续 {patience} 轮未下降，最优验证损失: {best_val_loss:.6f}")
                break

    return train_losses, val_losses


# ======================== 损失曲线绘制 ========================
def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue',  linewidth=2)
    plt.plot(val_losses,   label='Val Loss',   color='red',   linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss',  fontsize=12)
    plt.title('Vanilla RNN — Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================== 主函数 ========================
if __name__ == "__main__":

    # 1. 数据加载与预处理
    dataset        = load_and_merge_npy(NPY_FILE_PATHS)
    states, actions = preprocess_data(dataset)

    # 2. 转 Tensor
    states_tensor  = torch.tensor(states,  dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    # 3. 划分训练 / 验证集
    train_states, val_states, train_actions, val_actions = train_test_split(
        states_tensor, actions_tensor, test_size=TEST_SIZE, random_state=1
    )

    # 4. DataLoader
    train_loader = DataLoader(
        TensorDataset(train_states, train_actions),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_states, val_actions),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # 5. 模型初始化
    model = VanillaRNNTrajectoryPredictor(
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        output_dim  = OUTPUT_DIM,
        seq_length  = SEQ_LENGTH,
        car_num     = CAR_NUM,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        nonlinearity='tanh',
    ).to(device)

    # 续跑：检测已有模型
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"检测到已有模型，加载: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("未检测到已有模型，使用随机初始化。")

    # 6. 损失函数 + 优化器 + 调度器
    weights   = torch.tensor(
        [ACCELERATION_WEIGHT, STEERING_WEIGHT] * SEQ_LENGTH,
        dtype=torch.float32, device=device
    )
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=64,       # 第一个重启周期长度
        T_mult=2,     # 每次重启后周期翻倍
        eta_min=1e-6,
    )

    # 7. 训练
    print("\n开始训练 Vanilla RNN 模型...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
    )

    # 8. 最终保存损失曲线
    plot_loss_curve(train_losses, val_losses, LOSS_PLOT_SAVE_PATH)
    print("\n训练流程全部完成！")