import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)
# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class NetWithEncoderDecoderGRU_properRL(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, seq_length):
        super(NetWithEncoderDecoderGRU_properRL, self).__init__()

        # 定义嵌入层
        self.embedding_main_target = nn.Linear(6, embed_dim)  # 主车及目标点
        self.embedding_vehicle = nn.Linear(6, embed_dim)  # 每辆车

        # GRU 层
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 前向全连接层，带激活函数
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.seq_length = seq_length
        self.car_num = 8

    def forward(self, x):
        # 拆分输入为多个子序列
        main_target = x[:, 0:6]  # 主车及目标点信息
        vehicles = x[:, 6:6 + self.car_num * 6].reshape(-1, self.car_num, 6)  # (batch_size, self.car_num, 6)

        # 生成随机排列的索引
        random_indices = torch.randperm(self.car_num)
        vehicles = vehicles[:, random_indices, :]

        # 对每个子序列进行嵌入
        main_target_emb = self.embedding_main_target(main_target).unsqueeze(1)  # (batch_size, 1, embed_dim)
        vehicles_emb = self.embedding_vehicle(vehicles)  # (batch_size, 6, embed_dim)

        # 将所有嵌入拼接成一个序列
        sequence = torch.cat([vehicles_emb, main_target_emb], dim=1)  # (batch_size, 7, embed_dim)
        # 激活函数
        sequence = torch.tanh(sequence)

        # 使用 GRU 编码器进行处理
        encoder_output, hidden_state = self.encoder_gru(sequence)  # (batch_size, 7, hidden_dim)

        # 解码器的输入是编码器的最后一个输出
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)

        out = torch.zeros(x.size(0), self.seq_length, 2).to(device)
        # 解码器的输出
        for i in range(self.seq_length):
            decoder_output, hidden_state = self.decoder_gru(decoder_input, hidden_state)
            decoder_input = decoder_output

            # 通过全连接层并使用激活函数
            output = self.fc1(decoder_output)
            output = self.relu(output)
            output = torch.tanh(self.fc2(output))
            out[:, i, :] = output.squeeze(1)

        return out

dataset0 = np.load('D:\C\linz\BS\onsite\dataset\proper_RL\GRU_choose_closest8_properRL_0.npy')
dataset1 = np.load('D:\C\linz\BS\onsite\dataset\proper_RL\GRU_choose_closest8_properRL_1.npy')
dataset2 = np.load('D:\C\linz\BS\onsite\dataset\proper_RL\GRU_choose_closest8_properRL_out.npy')

dataset = np.concatenate((dataset0, dataset1, dataset2), axis=0)
print(dataset.shape)
# action是后10列
states = dataset[:, :-10]
actions = dataset[:, -10:]
# 清除内存
# del dataset0, dataset1, dataset
del dataset0, dataset1, dataset2, dataset

# # steer的最大值和最小值
# print(np.max(actions[:, 1]), np.min(actions[:, 1]))
# 0,2,4,6,8为加速度，1,3,5,7,9为转角，对这些列进行归一化
actions[:, 0:10:2] = np.clip(actions[:, 0:10:2] / 10, -1, 1)
actions[:, 1:10:2] = np.clip(actions[:, 1:10:2] / 0.15, -1, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
# 定义模型参数和学习率
learning_rate = 0.001
lr_decay = 0.025  # 通常取值范围是 0.01 到 0.1 0.022相当于每100个epoch衰减到0.1 0.015相当于每150个epoch衰减到0.1
batch_size = 2048
# 早停技术参数
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0
# 假设加速度的权重为1，转角的权重为50
acceleration_weight = 1
steering_weight = 5
# 创建权重向量，形状是 (10,) 对应于 5 个时间步，每个时间步的加速度和转角的权重
weights = torch.tensor([acceleration_weight, steering_weight] * 5, dtype=torch.float32).to(device)

# 将数据划分为训练集和验证集
train_states, val_states, train_actions, val_actions = train_test_split(states, actions, test_size=0.1, random_state=1)
# 转换数据为PyTorch张量并迁移到GPU
train_states = torch.tensor(train_states, dtype=torch.float32).to(device)
train_actions = torch.tensor(train_actions, dtype=torch.float32).to(device)
val_states = torch.tensor(val_states, dtype=torch.float32).to(device)
val_actions = torch.tensor(val_actions, dtype=torch.float32).to(device)
# 创建DataLoader
train_dataset = TensorDataset(train_states, train_actions)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_states, val_actions)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 创建一个列表来保存每个epoch的损失值
train_losses = []
val_losses = []

# 创建模型并迁移到GPU
model = NetWithEncoderDecoderGRU_properRL(32, 64, 2, 5).to(device)

# 创建 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1-lr_decay)

epo = 200
# 训练模型
for epoch in range(epo):
    model.train()
    train_epoch_loss = 0.0

    for batch_states, batch_actions in train_dataloader:
        # 前向传播
        outputs = model(batch_states)
        outputs = outputs.reshape(-1, 10)

        # 将权重应用于 outputs 和 batch_actions
        weighted_outputs = outputs * weights
        weighted_batch_actions = batch_actions * weights

        # loss = criterion(outputs, batch_actions)
        loss = criterion(weighted_outputs, weighted_batch_actions)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()

    scheduler.step()
    # 打印学习率
    # print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())

    average_train_loss = train_epoch_loss / len(train_dataloader)
    train_losses.append(average_train_loss)

    # 验证模型
    model.eval()
    val_epoch_loss = 0.0
    with torch.no_grad():
        for val_batch_states, val_batch_actions in val_dataloader:
            val_outputs = model(val_batch_states)
            val_outputs = val_outputs.reshape(-1, 10)

            # 将权重应用于 outputs 和 batch_actions
            val_weighted_outputs = val_outputs * weights
            val_weighted_batch_actions = val_batch_actions * weights

            # val_loss = criterion(val_outputs, val_batch_actions)
            val_loss = criterion(val_weighted_outputs, val_weighted_batch_actions)

            val_epoch_loss += val_loss.item()

    average_val_loss = val_epoch_loss / len(val_dataloader)
    val_losses.append(average_val_loss)

    print(f'Epoch {epoch + 1}, Train Loss: {average_train_loss:.6f}, Val Loss: {average_val_loss:.6f}')

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'model_GRU_closest8_sim_tanh_propeRL_01o.pth')
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print('Early stopping at epoch:', epoch + 1)
        break

# 绘制训练和验证损失图
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
# plt.show()
# plt.savefig('sLSTM_Multi1_choose_closest8_sim_tanh_01o' + '.svg')
plt.savefig('GRU_choose_closest8_sim_tanh_properRL_01o' + '.svg')