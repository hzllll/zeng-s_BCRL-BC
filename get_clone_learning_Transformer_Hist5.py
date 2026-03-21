import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import math

# ======================== 全局配置 ========================
torch.manual_seed(2025)
np.random.seed(2025)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 文件路径配置（请修改为你实际的文件夹路径）
DATA_DIR = os.path.join(BASE_DIR, "history5_123_history_traj_acc_steer")
NPY_FILE_PATHS = [
    os.path.join(DATA_DIR, "history5_closest8_3_10000_1_history_traj_acc_steer.npy"), # 请替换为实际的文件名
    os.path.join(DATA_DIR, "history5_closest8_3_10000_2_history_traj_acc_steer.npy"),
    os.path.join(DATA_DIR, "history5_closest8_3_10000_3_history_traj_acc_steer.npy"),
    # os.path.join(DATA_DIR, "your_dataset_file_2.npy"),
]

# ======================== 模型超参数配置 ========================
HIST_SEQ_LENGTH = 5     # 历史帧数
PRED_SEQ_LENGTH = 5     # 预测帧数
FRAME_DIM = 74          # 每一帧的特征维度 (自车10 + 8*8)
OUTPUT_DIM = 2          # 输出维度（加速度+转角）

# D_MODEL = 128           # Transformer 隐藏维度 (推荐128，防止在74维输入上过拟合)
D_MODEL = 256
FFN_DIM = 4 * D_MODEL   # 前馈层维度
num_encoder_layers = 3  # encoder层数
num_decoder_layers = 3  # decoder层数

# ======================== 训练超参数配置 ========================
# LEARNING_RATE = 5e-4
LEARNING_RATE = 3e-4
BATCH_SIZE = 1024 # 批次大小
EPOCHS = 512 # 最大训练轮数
EARLY_STOPPING_PATIENCE = 25 # 早停
ACCELERATION_WEIGHT = 1.0  # 加速度损失权重
STEERING_WEIGHT = 5.0      # 转角损失权重
LAMBDA_SMOOTH = 0.2        # 【新增】平滑损失权重，惩罚动作剧烈抖动
TEST_SIZE = 0.1

# 确保输出目录存在
checkpoints_dir = os.path.join(BASE_DIR, "Transformer_checkpoints_his5")
plots_dir = os.path.join(BASE_DIR, "Transformer_plots_his5")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(checkpoints_dir, f"TF_Hist5_{BATCH_SIZE}B_{D_MODEL}d_enc{num_encoder_layers}.pth")
LOSS_PLOT_SAVE_PATH = os.path.join(plots_dir, f"TF_Hist5_{BATCH_SIZE}B_{D_MODEL}d_enc{num_encoder_layers}.svg")

# ======================== 位置编码模块 ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

# ======================== 模型定义 ========================
class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, d_model, output_dim, hist_seq_len, pred_seq_len, frame_dim,
                 nhead=8, num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.10):
        super().__init__()
        self.hist_seq_len = hist_seq_len
        self.pred_seq_len = pred_seq_len
        
        # 1. 帧特征嵌入：将每一帧的 74 维映射到 d_model
        self.frame_embedding = nn.Linear(frame_dim, d_model)
        
        # 2. 时间位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=hist_seq_len)
        self.dropout_embed = nn.Dropout(dropout)

        # 3. Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # 4. Transformer 解码器
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # 5. 未来查询向量 (Query Tokens)
        self.future_queries = nn.Parameter(torch.randn(pred_seq_len, d_model))

        # 6. 输出头
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # x shape: (Batch_Size, 370)
        B = x.size(0)
        
        # --- 1. 重塑为时间序列 ---
        # 变成 (Batch, 5帧, 74维)
        x_seq = x.view(B, self.hist_seq_len, -1) 
        
        # --- 2. 嵌入与位置编码 ---
        tokens = self.frame_embedding(x_seq)      # (B, 5, d_model)
        tokens = torch.tanh(tokens)
        tokens = self.pos_encoder(tokens)         # 注入时间顺序信息
        tokens = self.dropout_embed(tokens)

        # --- 3. 编码器 ---
        memory = self.encoder(tokens)             # (B, 5, d_model)

        # --- 4. 解码器 ---
        tgt = self.future_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 5, d_model)
        dec_out = self.decoder(tgt=tgt, memory=memory)            # (B, 5, d_model)

        # --- 5. 输出 ---
        pred = self.head(dec_out)                 # (B, 5, 2)
        return pred

# ======================== 数据加载与预处理 ========================
def load_and_merge_npy(file_paths):
    """
    加载并合并多个npy文件
    file_paths: npy文件路径列表
    return: 合并后的完整数据集
    """
    dataset_list = []
    for path in file_paths:
        try:
            data = np.load(path)
            dataset_list.append(data)
            print(f"成功加载: {path} | 形状: {data.shape}")
        except Exception as e:
            print(f"加载失败 {path}: {str(e)}")
    merged_dataset = np.concatenate(dataset_list, axis=0)
    print(f"\n数据合并完成 | 总量: {merged_dataset.shape[0]} | 维度: {merged_dataset.shape[1]}")
    return merged_dataset

def preprocess_data(dataset):
    """
    数据预处理：拆分特征/标签，归一化标签
    dataset: 合并后的原始数据集
    return: 预处理后的特征(states)和标签(actions)
    """
    # 前370列为特征，后10列为标签
    states = dataset[:, :370]  
    actions = dataset[:, 370:] 

    # 标签归一化
    actions[:, 0:10:2] = np.clip(actions[:, 0:10:2] / 6, -1, 1)    # 加速度
    actions[:, 1:10:2] = np.clip(actions[:, 1:10:2] / 0.15, -1, 1) # 转角

    print(f"预处理完成 | 特征: {states.shape} | 标签: {actions.shape}")
    return states, actions

# ======================== 训练与验证 ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        scheduler.step()
        model.train()
        train_epoch_loss = 0.0

        for batch_states, batch_actions in tqdm(train_loader, leave=False):
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            outputs = model(batch_states)  # (B, 5, 2)
            
            # --- 计算 MSE Loss ---
            outputs_flat = outputs.reshape(-1, 10)
            weighted_outputs = outputs_flat * weights
            weighted_actions = batch_actions * weights
            mse_loss = criterion(weighted_outputs, weighted_actions)

            # --- 【新增】计算平滑损失 (Smoothness Loss) ---
            # 惩罚相邻帧之间动作的剧烈变化 (t+1 减 t)
            diff = outputs[:, 1:, :] - outputs[:, :-1, :]  # shape: (B, 4, 2)
            smoothness_loss = torch.mean(diff ** 2)

            # 总损失
            loss = mse_loss + LAMBDA_SMOOTH * smoothness_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_batch_states, val_batch_actions in val_loader:
                val_batch_states = val_batch_states.to(device)
                val_batch_actions = val_batch_actions.to(device)

                val_outputs = model(val_batch_states)
                
                val_outputs_flat = val_outputs.reshape(-1, 10)
                val_weighted_outputs = val_outputs_flat * weights
                val_weighted_actions = val_batch_actions * weights
                val_mse_loss = criterion(val_weighted_outputs, val_weighted_actions)

                val_diff = val_outputs[:, 1:, :] - val_outputs[:, :-1, :]
                val_smoothness_loss = torch.mean(val_diff ** 2)

                val_loss = val_mse_loss + LAMBDA_SMOOTH * val_smoothness_loss
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f} | 学习率: {scheduler.get_last_lr()[0]:.6f}")
        plot_loss_curve(train_losses, val_losses, LOSS_PLOT_SAVE_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"验证损失下降，保存最优模型到: {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！连续{patience}轮验证损失未下降，最优验证损失: {best_val_loss:.6f}")
                break

    return train_losses, val_losses

# ======================== 损失曲线绘制与保存 ========================
def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    dataset = load_and_merge_npy(NPY_FILE_PATHS)
    states, actions = preprocess_data(dataset)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    train_states, val_states, train_actions, val_actions = train_test_split(
        states_tensor, actions_tensor, test_size=TEST_SIZE, random_state=1
    )

    train_dataset = TensorDataset(train_states, train_actions)
    val_dataset = TensorDataset(val_states, val_actions)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerTrajectoryPredictor(
        d_model=D_MODEL,
        output_dim=OUTPUT_DIM,
        hist_seq_len=HIST_SEQ_LENGTH,
        pred_seq_len=PRED_SEQ_LENGTH,
        frame_dim=FRAME_DIM,
        nhead=8, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=FFN_DIM,
        dropout=0.05
    ).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"检测到已有模型文件，加载预训练模型: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("未检测到已有模型文件，使用随机初始化模型参数。")


    weights = torch.tensor([ACCELERATION_WEIGHT, STEERING_WEIGHT] * PRED_SEQ_LENGTH, 
                            dtype=torch.float32, device=device)
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    num_warmup_epochs = max(1, int(0.05 * EPOCHS))
    def lr_lambda(current_epoch: int):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch + 1) / float(num_warmup_epochs)
        progress = float(current_epoch - num_warmup_epochs) / max(1, EPOCHS - num_warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print("\n开始训练模型...")
    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE
    )
    print("\n训练流程全部完成！")