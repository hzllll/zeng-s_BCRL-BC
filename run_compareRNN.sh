#!/bin/bash

# 步骤：
# 创建screen窗口：'screen -S myrun'
# 在screen中运行脚本：
# chmod +x run_all.sh # 第一次运行授权，后续不用再输
# ./run_all.sh # 运行脚本
# 在screen中退出：Ctrl+a d
# 查看screen窗口：screen -ls
# 重新进入screen窗口：screen -r myrun
# 'nvidia-smi'看gpu、显卡状态
# 'tail -f train_log_0414TFexp-2_1024BSIZE_512DMODEL_xxx.txt'看实时训练日志
# 提前终止所有任务（整个脚本）：pkill -f run_all.sh
# 只杀exp-2:
# ps aux | grep TF_exp-2.py # 查找exp-2进程 ->会输出“root  12345 …………”
# kill 12345 # 输入kill + PID进程号（数字，⚠️ 把 12345 换成你查到的真实 PID！）

# 确保 train_log 文件夹存在，如果不存在则自动创建
mkdir -p train_log

echo "========== 开始执行第一批实验：vanilla_RNN 和 LSTM =========="

# 后台运行 vanilla_RNN
# $(date +"%m%d_%H%M") 会在这一行执行的瞬间，自动替换为当前时间，如 0414_1930
python -u vanilla_RNN_get_clone_learning.py | tee train_log/train_log_vanilla_RNN_$(date +"%m%d_%H%M")_1024BSIZE_128emb_256hid_2lay_150ESTOP_0.20drop_CoAnWarmRest_zDATASET.txt &

# 后台运行 LSTM
python -u LSTM_get_clone_learning.py | tee train_log/train_log_LSTM_$(date +"%m%d_%H%M")_1024BSIZE_128emb_256hid_2lay_150ESTOP_0.20drop_CoAnWarmRest_zDATASET.txt &

# 等待第一批实验全部结束
wait
echo "========== 第一批实验 (vanilla_RNN, LSTM) 已全部完成 =========="


echo "========== 开始执行第二批实验：AR_TF 和 GRU =========="

# 后台运行 AR_TF
# 注意：这里的 $(date +"%m%d_%H%M") 是在第一批跑完后才获取的时间，所以它记录的是第二批真正开跑的时间！
python -u AR_get_clone_learning.py | tee train_log/train_log_AR_TF_$(date +"%m%d_%H%M")_1024BSIZE_256DMODEL_1024FFNdim_0.15drop_250ESTOP_CoAnWarmRest_zDATASET.txt &

# 后台运行 GRU
python -u get_clone_learning_lw_mi.py | tee train_log/train_log_GRU_$(date +"%m%d_%H%M")_1024BSIZE_128emb_256hid_2lay_250ESTOP_0.20drop_CoAnWarmRest_zDATASET.txt &

# 再次等待第二批实验全部结束
wait
echo "========== 第二批实验 (AR_TF, GRU) 已全部完成 =========="


echo "========== 所有实验运行完毕，准备关机 =========="
# 执行关机命令
shutdown