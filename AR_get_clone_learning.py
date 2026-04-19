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
from datetime import datetime

# ======================== 全局配置 ========================
torch.manual_seed(2025)
np.random.seed(2025)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 数据集路径 (请确保与你原代码一致)
NPY_FILE_PATHS = [
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_0.npy"),
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_1.npy"),
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_out.npy"),
]

EXP_NAME = "Exp-AR-Baseline"  # 标记为自回归基线实验

# 模型与训练超参数
D_MODEL = 256
FFN_DIM = 4 * D_MODEL
num_encoder_layers = 3
num_decoder_layers = 3
OUTPUT_DIM = 2
SEQ_LENGTH = 5
CAR_NUM = 8

LEARNING_RATE = 5e-4
BATCH_SIZE = 1024
EPOCHS = 512
EARLY_STOPPING_PATIENCE = 250
ACCELERATION_WEIGHT = 1
STEERING_WEIGHT = 5
TEST_SIZE = 0.1

checkpoints_dir = os.path.join(BASE_DIR, "Transformer_checkpoints")
plots_dir = os.path.join(BASE_DIR, "Transformer_plots")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
RUN_DATE_TIME = datetime.now().strftime("%m%d_%H")

MODEL_SAVE_PATH = os.path.join(checkpoints_dir, f"{EXP_NAME}_AR_model_{RUN_DATE_TIME}.pth")
LOSS_PLOT_SAVE_PATH = os.path.join(plots_dir, f"{EXP_NAME}_AR_loss_{RUN_DATE_TIME}.svg")

# ======================== 传统自回归 Transformer 模型 ========================
class AutoregressiveTransformerPredictor(nn.Module):
    def __init__(self, d_model, output_dim, seq_length, car_num, nhead=8, 
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.car_num = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim

        # --- Encoder 侧 (与原模型一致) ---
        self.embedding_main_target = nn.Linear(6, d_model)
        self.embedding_vehicle = nn.Linear(6, d_model)
        self.dropout_embed = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # --- Decoder 侧 (自回归特有) ---
        # 动作嵌入：将上一时刻的动作(2维)映射为d_model维，作为下一步的输入
        self.action_embedding = nn.Linear(output_dim, d_model)
        # SOS Token：序列起始标志 (Start Of Sequence)
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # --- 输出头 ---
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh(),
        )

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码，防止解码时看到未来信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, target_actions=None):
        """
        x: (B, 54)
        target_actions: (B, 5, 2) 仅在训练时提供，用于 Teacher Forcing
        """
        B = x.size(0)

        # 1. Encoder 提取全局特征
        main_target_feat = x[:, 0:6]
        vehicle_feats = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)

        main_emb = self.embedding_main_target(main_target_feat).unsqueeze(1)
        veh_emb = self.embedding_vehicle(vehicle_feats)
        
        tokens = torch.cat([veh_emb, main_emb], dim=1)
        tokens = torch.tanh(tokens)
        tokens = self.dropout_embed(tokens)

        memory = self.encoder(tokens)  # (B, 9, d_model)

        # 2. Decoder 解码
        if self.training and target_actions is not None:
            # 【训练模式：Teacher Forcing】
            # 输入序列为 [SOS, action_1, action_2, action_3, action_4]
            action_embs = self.action_embedding(target_actions)  # (B, 5, d_model)
            sos = self.sos_token.expand(B, -1, -1)               # (B, 1, d_model)
            
            tgt = torch.cat([sos, action_embs[:, :-1, :]], dim=1) # (B, 5, d_model)
            tgt_mask = self.generate_square_subsequent_mask(self.seq_length).to(x.device)
            
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            pred = self.head(dec_out)  # (B, 5, 2)
            return pred
        else:
            # 【推理/验证模式：逐步自回归生成】
            sos = self.sos_token.expand(B, -1, -1)
            tgt = sos
            preds = []

            for i in range(self.seq_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(x.device)
                dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                
                # 取当前序列最后一个时间步的输出作为最新预测
                step_out = dec_out[:, -1, :]  # (B, d_model)
                step_pred = self.head(step_out)  # (B, 2)
                preds.append(step_pred.unsqueeze(1))
                
                # 将预测结果嵌入后拼接到 tgt，用于预测下一步
                if i < self.seq_length - 1:
                    step_pred_emb = self.action_embedding(step_pred).unsqueeze(1)
                    tgt = torch.cat([tgt, step_pred_emb], dim=1)

            return torch.cat(preds, dim=1)  # (B, 5, 2)

# ======================== 数据加载与预处理 ========================
# (与原代码完全相同，保持不变)
def load_and_merge_npy(file_paths):
    dataset_list = []
    for path in file_paths:
        data = np.load(path)
        dataset_list.append(data)
    return np.concatenate(dataset_list, axis=0)

def preprocess_data(dataset):
    states = dataset[:, :-10]
    actions = dataset[:, -10:]
    actions[:, 0:10:2] = np.clip(actions[:, 0:10:2] / 6, -1, 1)
    actions[:, 1:10:2] = np.clip(actions[:, 1:10:2] / 0.15, -1, 1)
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

            # 【注意这里的区别：训练时需要传入 target_actions】
            outputs = model(batch_states, target_actions=batch_actions.reshape(-1, 5, 2))
            outputs = outputs.reshape(-1, 10)

            weighted_outputs = outputs * weights
            weighted_actions = batch_actions * weights
            loss = criterion(weighted_outputs, weighted_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_batch_states, val_batch_actions in val_loader:
                val_batch_states = val_batch_states.to(device)
                val_batch_actions = val_batch_actions.to(device)

                # 【验证时：不传 target_actions，触发自回归推理】
                val_outputs = model(val_batch_states)
                val_outputs = val_outputs.reshape(-1, 10)

                val_weighted_outputs = val_outputs * weights
                val_weighted_actions = val_batch_actions * weights
                val_loss = criterion(val_weighted_outputs, val_weighted_actions)
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发！")
                break

    return train_losses, val_losses

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val loss', color='red')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    dataset = load_and_merge_npy(NPY_FILE_PATHS)
    states, actions = preprocess_data(dataset)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    train_states, val_states, train_actions, val_actions = train_test_split(
        states_tensor, actions_tensor, test_size=TEST_SIZE, random_state=1
    )

    train_loader = DataLoader(TensorDataset(train_states, train_actions), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_states, val_actions), batch_size=BATCH_SIZE, shuffle=False)

    model = AutoregressiveTransformerPredictor(
        d_model=D_MODEL, output_dim=OUTPUT_DIM, seq_length=SEQ_LENGTH, car_num=CAR_NUM,
        nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        dim_feedforward=FFN_DIM, 
        # dropout=0.1
        dropout=0.15
    ).to(device)

    weights = torch.tensor([ACCELERATION_WEIGHT, STEERING_WEIGHT] * SEQ_LENGTH, dtype=torch.float32, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=64, T_mult=2, eta_min=1e-6)

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, EARLY_STOPPING_PATIENCE
    )
    plot_loss_curve(train_losses, val_losses, LOSS_PLOT_SAVE_PATH)