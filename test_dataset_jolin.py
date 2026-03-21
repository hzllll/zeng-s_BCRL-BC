import numpy as np
import os

def check_dataset_dim(path):
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return

    try:
        # 使用 mmap_mode 避免大文件占满内存
        data = np.load(path, mmap_mode='r')
        total_cols = data.shape[1]
        print(f"\n✅ 成功加载数据: {os.path.basename(path)}")
        print(f"📊 数据总行数: {data.shape[0]}")
        print(f"📏 数据总列数: {total_cols}")
        
        # 预期的维度
        expected_cols_6dim = 6 + 8*6 + 10  # = 64
        expected_cols_7dim = 7 + 8*7 + 10  # = 73
        expected_cols_380 = 370 + 10       # = 380 (历史5帧 + 目标控制量[acc, steer]*5)
        expected_cols_385 = 370 + 15       # = 385 (历史5帧 + 目标控制量[x, y, v]*5)
        
        # 判断逻辑
        if total_cols == expected_cols_6dim:
            print("🟢 判定结果: 6维特征格式 (单帧 64列)")
            print("   - 主车: 6维\n   - 障碍车: 8*6=48维\n   - 预测目标: 10维 (5步 * 2控制量)")
        elif total_cols == expected_cols_7dim:
            print("🟠 判定结果: 7维特征格式 (单帧 73列)")
            print("   - 主车: 7维\n   - 障碍车: 8*7=56维\n   - 预测目标: 10维 (5步 * 2控制量)")
            print("   ⚠️ 注意: 当前代码可能需要修改切片逻辑！")
        elif total_cols == expected_cols_380:
            print("🔵 判定结果: 历史轨迹特征格式 (多帧 380列)")
            print("   - 输入序列: 370维 (5帧历史 * (主车10维 + 8障碍车*8维))")
            print("   - 预测目标: 10维 (未来5帧的 [加速度, 转角])")
        elif total_cols == expected_cols_385:
            print("🟣 判定结果: 历史轨迹特征格式 (多帧 385列)")
            print("   - 输入序列: 370维 (5帧历史 * (主车10维 + 8障碍车*8维))")
            print("   - 预测目标: 15维 (未来5帧的 [x, y, v])")
        else:
            print(f"❓ 未知格式: 总列数 {total_cols} 不匹配任何已知预期。")
            
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")

if __name__ == "__main__":
    # 支持检查多个文件，把你想检查的路径加到这里
    paths_to_check = [
        "dataset/GRU_choose_closest8_properRL_0.npy",  # 原先的 64维数据
        # "history5_closest8_1234_history_traj_relvel/history5_closest8_3_10000_1_history_traj_obs_relvel.npy"
        # "history5_closest8_1234_history_traj_relvel/history5_closest8_3_10000_2_history_traj_obs_relvel.npy"
        "history5_123_history_traj_acc_steer/history5_closest8_3_10000_1_history_traj_acc_steer.npy"
    ]
    
    for p in paths_to_check:
        check_dataset_dim(p)


# import numpy as np
# import os

# # 替换为你实际的一个 npy 文件路径
# # path = "dataset/GRU_choose_closest8_properRL_0.npy" 
# path = "/history5_closest8_1234_history_traj_relvel/history5_closest8_3_10000_1_history_traj_obs_relvel.npy"

# if os.path.exists(path):
#     data = np.load(path)
#     print(f"数据总列数: {data.shape[1]}")
    
#     # 验证
#     expected_cols_6dim = 6 + 8*6 + 10  # = 64
#     expected_cols_7dim = 7 + 8*7 + 10  # = 73
    
#     if data.shape[1] == expected_cols_6dim:
#         print("✅ 数据格式正确 (6维)，代码可以直接运行！")
#     elif data.shape[1] == expected_cols_7dim:
#         print("❌ 数据格式为 7维，代码需要修改切片逻辑！")
#     else:
#         print(f"⚠️ 数据列数异常，请检查数据生成过程。")
# else:
#     print("文件不存在，请检查路径。")