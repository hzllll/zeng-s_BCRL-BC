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

# ======================== 全局配置（可直接调整）========================
# 随机种子固定（保证实验可复现）
torch.manual_seed(2025)
np.random.seed(2025)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备配置（自动检测GPU/CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 文件路径配置（按需修改）
# NPY_FILE_PATHS = [
#     # r"F:\WCY\00.code\曾润林毕业资料\模仿学习\dataset\GRU_choose_closest8_properRL_0.npy",
#     # r"F:\WCY\00.code\data\npy_test\choose_closest8_15000_1_0.1rad_len_wid_milane.npy",  
#     # r"F:\WCY\00.code\data\npy_test\choose_closest8_15000_2_0.1rad_len_wid_milane.npy",
#     # r"F:\WCY\00.code\data\npy_test\choose_closest8_15000_3_0.1rad_len_wid_milane.npy",    
#     # r"F:\WCY\00.code\data\npy_test\choose_closest8_3_10000_1_0.1rad_len_wid_milane.npy"
#     # r"E:\aaaproject\clone_learning\data0206\choose_closest8_15000_1_0.1rad.npy",
#     # r"E:\aaaproject\clone_learning\data0206\choose_closest8_15000_2_0.1rad.npy",
#     # r"E:\aaaproject\clone_learning\data0206\choose_closest8_15000_3_0.1rad.npy",
#     r"/root/autodl-tmp/clone_learning/data0206/choose_closest8_15000_1_0.1rad_len_wid_milane.npy",
#     r"/root/autodl-tmp/clone_learning/data0206/choose_closest8_15000_2_0.1rad_len_wid_milane.npy",
#     r"/root/autodl-tmp/clone_learning/data0206/choose_closest8_15000_3_0.1rad_len_wid_milane.npy",
# ]

# 文件路径配置(改为相对路径)
# NPY_FILE_PATHS = [
#     os.path.join(BASE_DIR, "data0206", "choose_closest8_15000_1_0.1rad_len_wid_milane.npy"),
#     os.path.join(BASE_DIR, "data0206", "choose_closest8_15000_2_0.1rad_len_wid_milane.npy"),
#     os.path.join(BASE_DIR, "data0206", "choose_closest8_15000_3_0.1rad_len_wid_milane.npy"),
# ]

NPY_FILE_PATHS = [
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_0.npy"),
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_1.npy"),
    os.path.join(BASE_DIR, "dataset", "GRU_choose_closest8_properRL_out.npy"),
]


# 模型超参数配置
D_MODEL = 256           # Transformer 的隐藏维度
FFN_DIM = 4 * D_MODEL   # 前馈层维度，经验上 4×d_model 比较常见
num_encoder_layers = 3  # encoder层数 (2 -> 3)
num_decoder_layers = 3  # decoder层数 (2 -> 3)
OUTPUT_DIM = 2  # 输出维度（加速度+转角）
SEQ_LENGTH = 5  # 预测序列长度（5个时间步）
CAR_NUM = 8  # 障碍车数量

# 训练超参数配置
# LEARNING_RATE = 3e-4  # 初始学习率
LEARNING_RATE = 5e-4
# LEARNING_RATE = 0.001
# BATCH_SIZE = 128  # 批次大小
BATCH_SIZE = 1024
EPOCHS = 512  # 最大训练轮数
# EARLY_STOPPING_PATIENCE = 20  # 早停耐心值（连续多少轮验证集loss不下降则停止）
EARLY_STOPPING_PATIENCE = 500  # 早停耐心值（连续多少轮验证集loss不下降则停止）
ACCELERATION_WEIGHT = 1  # 加速度损失权重
STEERING_WEIGHT = 5  # 转角损失权重（转角对轨迹影响更大，权重更高）
TEST_SIZE = 0.1  # 验证集比例（10%数据用于验证）



# 确保输出目录存在
checkpoints_dir = os.path.join(BASE_DIR, "Transformer_checkpoints")
plots_dir = os.path.join(BASE_DIR, "Transformer_plots")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
RUN_DATE_TIME = datetime.now().strftime("%m%d")


MODEL_SAVE_PATH = os.path.join(checkpoints_dir, f"Tf_trajectory_model_{RUN_DATE_TIME}_{str(BATCH_SIZE)}BSIZE_{str(D_MODEL)}dmodel_{str(FFN_DIM)}FFNdim_enc{num_encoder_layers}_dec{num_decoder_layers}_{EARLY_STOPPING_PATIENCE}es_CoAnWarmRest_zDATASET.pth") # 模型保存路径
LOSS_PLOT_SAVE_PATH = os.path.join(plots_dir, f"Tf_loss_curve_{RUN_DATE_TIME}_{str(BATCH_SIZE)}BSIZE_{str(D_MODEL)}dmodel_{str(FFN_DIM)}FFNdim_enc{num_encoder_layers}_dec{num_decoder_layers}_{EARLY_STOPPING_PATIENCE}es_CoAnWarmRest_zDATASET.svg") # 损失曲线

# ======================== 模型定义（transformer编码器-解码器结构）========================
class TransformerTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        seq_length: int,
        car_num: int,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.car_num = car_num
        self.seq_length = seq_length
        self.output_dim = output_dim

        # 保留“主车/目标点”和“障碍车”分别embedding（等价于token-type区分）
        self.embedding_main_target = nn.Linear(6, d_model)
        self.embedding_vehicle = nn.Linear(6, d_model)

        self.dropout_embed = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,          # 如果torch太老不支持，见下方兼容说明
            norm_first=True,  # 关键：输入形状为 (batch, seq, feature)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # “未来5步查询向量”：每个时间步一个query token
        self.future_queries = nn.Parameter(torch.randn(seq_length, d_model))

        # 输出头：d_model -> hidden -> 2
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Tanh(),   # 你当前标签就是clip到[-1,1]，保持一致
        )

    def forward(self, x):
        # x shape: (Batch_Size, 6 + car_num * 6)

        # --- 1. 数据切片 (适配 6 维输入) ---
        main_target_feat = x[:, 0:6]  # (B,6)
        vehicle_feats = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)  # (B,car=8,6)

        # 如果你不加位置编码，Transformer对token顺序天然“等变/近似无序”
        # 所以随机打乱基本没意义；你也可以只在训练时打乱：
        # if self.training: ...

        # --- 2. 嵌入 (Embedding) ---
        main_emb = self.embedding_main_target(main_target_feat).unsqueeze(1)  # (B,1,d)
        veh_emb = self.embedding_vehicle(vehicle_feats)                       # (B,car,d)
        
        # 拼接所有对象特征作为 Encoder 输入序列
        # 序列长度 = 1 (主车) + 8 (障碍车) = 9
        tokens = torch.cat([veh_emb, main_emb], dim=1)                        # (B,car+1=9,d_model)
        tokens = torch.tanh(tokens)
        tokens = self.dropout_embed(tokens)

        # --- 3. Transformer Encoder ---
        memory = self.encoder(tokens)                                         # (B,car+1,d)

        # 构造decoder输入：把(SEQ_LENGTH,d)扩成(B,SEQ_LENGTH,d)
        tgt = self.future_queries.unsqueeze(0).expand(x.size(0), -1, -1)      # (B,T,d)
        dec_out = self.decoder(tgt=tgt, memory=memory)                         # (B,T,d)

        pred = self.head(dec_out)                                             # (B,T=5,2)
        return pred

    
# ======================== 数据加载与预处理函数 ========================
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
            print(f"成功加载文件: {path} | 数据形状: {data.shape}")
        except Exception as e:
            print(f"加载文件失败 {path}: {str(e)}")
    
    if not dataset_list:
        raise ValueError("未加载到任何有效npy文件！")
    
    # 合并所有npy数据
    merged_dataset = np.concatenate(dataset_list, axis=0)
    print(f"\n数据合并完成 | 总数据量: {merged_dataset.shape[0]} | 特征维度: {merged_dataset.shape[1]}")
    return merged_dataset

def preprocess_data(dataset):
    """
    数据预处理：拆分特征/标签，归一化标签
    dataset: 合并后的原始数据集
    return: 预处理后的特征(states)和标签(actions)
    """
    # 拆分特征（前n-10列）和标签（后10列：5个时间步的加速度+转角）
    states = dataset[:, :-10]  # 模型输入特征（主车+障碍车+目标点）
    actions = dataset[:, -10:]  # 模型输出标签（5×2：加速度/转角）

    # 标签归一化：加速度除以10，转角除以0.68，限制在[-1,1]
    actions[:, 0:10:2] = np.clip(actions[:, 0:10:2] / 6, -1, 1)  # 加速度列（0,2,4,6,8）
    actions[:, 1:10:2] = np.clip(actions[:, 1:10:2] / 0.15, -1, 1)  # 转角列（1,3,5,7,9）

    print(f"数据预处理完成 | 特征形状: {states.shape} | 标签形状: {actions.shape}")
    return states, actions

# ======================== 训练与验证函数 ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    """
    模型训练与验证，包含早停机制
    return: 训练损失列表，验证损失列表
    """
    train_losses = []  # 记录每轮训练损失
    val_losses = []    # 记录每轮验证损失
    best_val_loss = float('inf')  # 最优验证损失（初始化为无穷大）
    patience_counter = 0  # 早停计数器

    for epoch in range(epochs):
        scheduler.step()  # 每轮更新学习率，确保每个 epoch 里的所有 batch 都用同一个、已经更新好的 lr
        # ---------------------- 训练阶段 ----------------------
        model.train()  # 切换到训练模式（启用Dropout/BatchNorm）
        train_epoch_loss = 0.0

        for batch_states, batch_actions in tqdm(train_loader,leave=False):
            # 数据迁移到设备（GPU/CPU）
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            # 前向传播
            outputs = model(batch_states)  # 模型预测输出 (batch_size, seq_length, 2)
            outputs = outputs.reshape(-1, 10)  # 展平为(batch_size, 10)，与标签对齐

            # 加权损失计算（对加速度和转角赋予不同权重）
            weighted_outputs = outputs * weights
            weighted_actions = batch_actions * weights
            loss = criterion(weighted_outputs, weighted_actions)

            # 反向传播与优化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播计算梯度

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪：防止 Transformer 在个别 batch 上爆梯度
            optimizer.step()       # 更新参数

            train_epoch_loss += loss.item()  # 累加批次损失

        # 计算本轮平均训练损失
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------------------- 验证阶段 ----------------------
        model.eval()  # 切换到验证模式（禁用Dropout/BatchNorm）
        val_epoch_loss = 0.0

        with torch.no_grad():  # 禁用梯度计算（节省内存，加速计算）
            for val_batch_states, val_batch_actions in val_loader:
                val_batch_states = val_batch_states.to(device)
                val_batch_actions = val_batch_actions.to(device)

                val_outputs = model(val_batch_states)
                val_outputs = val_outputs.reshape(-1, 10)

                # 加权验证损失计算
                val_weighted_outputs = val_outputs * weights
                val_weighted_actions = val_batch_actions * weights
                val_loss = criterion(val_weighted_outputs, val_weighted_actions)

                val_epoch_loss += val_loss.item()

        # 计算本轮平均验证损失
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # ---------------------- 学习率衰减 + 早停判断 ----------------------
        # scheduler.step()  # 每轮更新学习率

        # 打印本轮训练/验证损失
        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f} | 学习率: {scheduler.get_last_lr()[0]:.6f}")
        plot_loss_curve(train_losses, val_losses, LOSS_PLOT_SAVE_PATH)

        # 保存最优模型（验证损失下降时）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"验证损失下降，保存最优模型到: {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1  # 验证损失未下降，计数器+1
            if patience_counter >= patience:
                print(f"早停触发！连续{patience}轮验证损失未下降，最优验证损失: {best_val_loss:.6f}")
                break

    return train_losses, val_losses

# ======================== 损失曲线绘制与保存 ========================
def plot_loss_curve(train_losses, val_losses, save_path):
    """
    绘制训练/验证损失曲线并保存
    train_losses: 训练损失列表
    val_losses: 验证损失列表
    save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
    plt.close()
    # print(f"损失曲线已保存到: {save_path}")

# ======================== 主函数（程序入口）========================
if __name__ == "__main__":
    # 1. 加载并合并npy数据
    dataset = load_and_merge_npy(NPY_FILE_PATHS)

    # 2. 数据预处理
    states, actions = preprocess_data(dataset)

    # 3. 转换为PyTorch张量
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    # 4. 划分训练集/验证集
    train_states, val_states, train_actions, val_actions = train_test_split(
        states_tensor, actions_tensor, test_size=TEST_SIZE, random_state=1
    )

    # 5. 创建DataLoader（批次加载数据）
    train_dataset = TensorDataset(train_states, train_actions)
    val_dataset = TensorDataset(val_states, val_actions)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 训练集打乱
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)     # 验证集不打乱

    # 6. 初始化 Transformer 模型、损失函数、优化器
    model = TransformerTrajectoryPredictor(
        d_model = D_MODEL,
        output_dim = OUTPUT_DIM,
        seq_length = SEQ_LENGTH,
        car_num = CAR_NUM,
        nhead=8, 
        num_encoder_layers=num_encoder_layers,       # 3 层 encoder
        num_decoder_layers=num_decoder_layers,       # 3 层 decoder
        dim_feedforward=FFN_DIM,    # 4*d_model 
        # dropout=0.1                 # 轻微正则，后续可调到 0.15
        dropout=0.15
    ).to(device)  # 模型迁移到设备

    # ================== 新增：加载已有模型逻辑 ==================
    # 如果文件开头没有import os，这里需要引用，或者最好加到文件最开头
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"检测到已有模型文件，加载预训练模型: {MODEL_SAVE_PATH}")
        # 加载参数（weights_only=True 是新版PyTorch的安全建议，如果报错可去掉）
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("未检测到已有模型文件，使用随机初始化模型参数。")

    # 续跑时注意学习率要从上次中断的地方开始，需要改（终端一般会打印出来）
    # ==========================================================

    # 加权MSE损失（加速度和转角分别加权）
    weights = torch.tensor([ACCELERATION_WEIGHT, STEERING_WEIGHT] * SEQ_LENGTH, 
                            dtype=torch.float32, device=device)
    criterion = nn.MSELoss()  # 均方误差损失

   
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RAπTE) # Adam优化器（带学习率衰减）
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1-LR_DECAY)
    # from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=32, T_mult=2, eta_min=1e-6)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2) # 后续保持或略增weight_decay

    # === 学习率调度：epoch 级 warmup + cosine ===
    # 例如前 10% 的 epoch 做 warmup，后 90% 做余弦衰减
    # num_warmup_epochs = max(1, int(0.1 * EPOCHS))
    # num_warmup_epochs = max(1, int(0.05 * EPOCHS))  # 10% → 5%


    # def lr_lambda(current_epoch: int):
    #     # current_epoch 从 0 开始计数
    #     if current_epoch < num_warmup_epochs:
    #         # 线性 warmup：从 0 -> 1
    #         return float(current_epoch + 1) / float(num_warmup_epochs)
    #     # 余弦衰减：从 1 -> 0
    #     progress = float(current_epoch - num_warmup_epochs) / max(
    #         1, EPOCHS - num_warmup_epochs
    #     )
    #     return 0.5 * (1.0 + math.cos(math.pi * progress))
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    
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


    # 7. 模型训练
    print("\n开始训练模型...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE
    )

    # 8. 绘制并保存损失曲线
    plot_loss_curve(train_losses, val_losses, LOSS_PLOT_SAVE_PATH)

    print("\n训练流程全部完成！")