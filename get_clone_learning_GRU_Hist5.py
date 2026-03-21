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

# 文件路径配置（请与Transformer代码保持一致）
DATA_DIR = os.path.join(BASE_DIR, "history5_123_history_traj_acc_steer")
NPY_FILE_PATHS = [
    os.path.join(DATA_DIR, "history5_closest8_3_10000_1_history_traj_acc_steer.npy"),
    os.path.join(DATA_DIR, "history5_closest8_3_10000_2_history_traj_acc_steer.npy"),
    os.path.join(DATA_DIR, "history5_closest8_3_10000_3_history_traj_acc_steer.npy"),
]

# ======================== 模型超参数配置 ========================
HIST_SEQ_LENGTH = 5     # 历史帧数
PRED_SEQ_LENGTH = 5     # 预测帧数
FRAME_DIM = 74          # 每一帧的特征维度 (自车10 + 8*8)
OUTPUT_DIM = 2          # 输出维度（加速度+转角）

EMBED_DIM = 128         # 帧特征嵌入维度 (与Transformer的D_MODEL保持一致)
HIDDEN_DIM = 256        # GRU隐藏层维度 
NUM_LAYERS = 2          # GRU层数

# ======================== 训练超参数配置 ========================
LEARNING_RATE = 5e-4
BATCH_SIZE = 1024
EPOCHS = 512
EARLY_STOPPING_PATIENCE = 25
ACCELERATION_WEIGHT = 1.0
STEERING_WEIGHT = 5.0
LAMBDA_SMOOTH = 0.2     # 平滑损失权重（与Transformer保持一致）
TEST_SIZE = 0.1

# 确保输出目录存在
checkpoints_dir = os.path.join(BASE_DIR, "GRU_checkpoints_his5")
plots_dir = os.path.join(BASE_DIR, "GRU_plots_his5")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(checkpoints_dir, f"GRU_Hist5_{BATCH_SIZE}B_{HIDDEN_DIM}H_{NUM_LAYERS}L.pth")
LOSS_PLOT_SAVE_PATH = os.path.join(plots_dir, f"GRU_Hist5_{BATCH_SIZE}B_{HIDDEN_DIM}H_{NUM_LAYERS}L.svg")

# ======================== 模型定义 (Seq2Seq GRU) ========================
class GRUTrajectoryPredictor(nn.Module):
    def __init__(self, frame_dim, embed_dim, hidden_dim, output_dim, hist_seq_len, pred_seq_len, num_layers=2, dropout=0.15):
        super(GRUTrajectoryPredictor, self).__init__()
        self.hist_seq_len = hist_seq_len
        self.pred_seq_len = pred_seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. 帧特征嵌入：将每一帧的 74 维映射到 embed_dim
        self.frame_embedding = nn.Linear(frame_dim, embed_dim)
        self.dropout_embed = nn.Dropout(dropout)

        # 2. GRU编码器：处理历史5帧的时间序列
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 3. GRU解码器：自回归生成未来5帧
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # 4. 输出预测头
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch_Size, 370)
        B = x.size(0)
        
        # --- 1. 重塑为时间序列 ---
        # 变成 (Batch, 5帧, 74维)
        x_seq = x.view(B, self.hist_seq_len, -1)
        
        # --- 2. 帧特征嵌入 ---
        emb = self.frame_embedding(x_seq)  # (B, 5, embed_dim)
        emb = torch.tanh(emb)
        emb = self.dropout_embed(emb)

        # --- 3. 编码器前向传播 ---
        # encoder_output: (B, 5, hidden_dim)
        # hidden_state: (num_layers, B, hidden_dim)
        encoder_output, hidden_state = self.encoder_gru(emb)

        # --- 4. 解码器准备 ---
        # 使用编码器最后一个时间步的输出作为解码器的初始输入
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)  # (B, 1, hidden_dim)
        
        pred_seq = torch.zeros(B, self.pred_seq_len, OUTPUT_DIM).to(x.device)

        # --- 5. 逐步解码预测 ---
        for t in range(self.pred_seq_len):
            decoder_output, hidden_state = self.decoder_gru(decoder_input, hidden_state)
            
            # 预测当前步的动作
            fc_out = self.fc1(decoder_output)
            fc_out = self.relu(fc_out)
            fc_out = self.dropout_fc(fc_out)
            fc_out = torch.tanh(self.fc2(fc_out))  # (B, 1, 2)
            
            pred_seq[:, t, :] = fc_out.squeeze(1)
            
            # 将当前步的输出特征作为下一步的输入 (自回归)
            decoder_input = decoder_output

        return pred_seq

# ======================== 数据加载与预处理 ========================
def load_and_merge_npy(file_paths):
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
    states = dataset[:, :370]  
    actions = dataset[:, 370:] 
    actions[:, 0:10:2] = np.clip(actions[:, 0:10:2] / 6, -1, 1)    
    actions[:, 1:10:2] = np.clip(actions[:, 1:10:2] / 0.15, -1, 1) 
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

            # --- 计算平滑损失 (Smoothness Loss) ---
            diff = outputs[:, 1:, :] - outputs[:, :-1, :]  # shape: (B, 4, 2)
            smoothness_loss = torch.mean(diff ** 2)

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

    model = GRUTrajectoryPredictor(
        frame_dim=FRAME_DIM,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        hist_seq_len=HIST_SEQ_LENGTH,
        pred_seq_len=PRED_SEQ_LENGTH,
        num_layers=NUM_LAYERS,
        dropout=0.15
    ).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"检测到已有模型文件，加载预训练模型: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("未检测到已有模型文件，使用随机初始化模型参数。")

    weights = torch.tensor([ACCELERATION_WEIGHT, STEERING_WEIGHT] * PRED_SEQ_LENGTH, 
                            dtype=torch.float32, device=device)
    criterion = nn.MSELoss()

    # 优化器与学习率调度器 (与Transformer保持一致，便于对比)
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