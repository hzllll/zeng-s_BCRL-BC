import numpy as np
import math
import matplotlib.pyplot as plt

def rotate(x, y, yaw):
    """逆时针旋转点 (x, y) 对应于原点旋转角度为 yaw 的点"""
    # x_rot = x * math.cos(yaw) - y * math.sin(yaw)
    # y_rot = x * math.sin(yaw) + y * math.cos(yaw)
    x_rot = x * math.cos(yaw) + y * math.sin(yaw)
    y_rot = -x * math.sin(yaw) + y * math.cos(yaw)
    return x_rot, y_rot

def path(sx, sy, gx, gy, yaw0):
    ky = 0
    if abs(sy - gy) > 25:
        # 将坐标轴逆时针旋转yaw0度，得到旋转后的sx, sy, gx, gy
        sx, sy = rotate(sx, sy, yaw0)
        gx, gy = rotate(gx, gy, yaw0)
        ky = 1

    # 先换道
    if gx > sx:
        gx = gx + 5
        x1 = sx + 25
        y1 = gy
    else:
        gx = gx - 5
        x1 = sx - 25
        y1 = gy

    # 在点(x1, y1)与(sx, sy)之间进行线性插值
    num_points1 = int(abs(sx - x1)) + 1  # 插值点的数量，根据距离差进行计算
    x_values1 = np.linspace(sx, x1, num_points1)  # 在x轴方向上进行线性插值
    y_values1 = np.linspace(sy, y1, num_points1)  # 在y轴方向上进行线性插值

    # 在点(x1, y1)与(gx, gy)之间进行线性插值
    num_points2 = int(abs(gx - x1)) + 1  # 插值点的数量，根据距离差进行计算
    x_values2 = np.linspace(x1, gx, num_points2)  # 在x轴方向上进行线性插值
    y_values2 = np.linspace(y1, gy, num_points2)  # 在y轴方向上进行线性插值

    # 合并路径点，确保路径点从起点sx到终点gx有顺序
    x_values = np.concatenate((x_values1, x_values2[1:]))
    y_values = np.concatenate((y_values1, y_values2[1:]))
    x_values = np.flip(x_values)
    y_values = np.flip(y_values)

    # if gx > sx:
    #     a = int((gx - sx) / 3)
    #     gx = gx + 5
    #     x1 = sx + a
    #     y1 = sy
    #     x2 = gx - a
    #     y2 = gy
    # else:
    #     a = int((gx - sx) / 3)
    #     gx = gx - 5
    #     x1 = sx - a
    #     y1 = sy
    #     x2 = gx + a
    #     y2 = gy
    #
    # # 在点(x1, y1)与(x2, y2)之间进行线性插值
    # num_points1 = int(abs(x2 - x1)) + 1  # 插值点的数量，根据距离差进行计算
    # x_values1 = np.linspace(x1, x2, num_points1)  # 在x轴方向上进行线性插值
    # y_values1 = np.linspace(y1, y2, num_points1)  # 在y轴方向上进行线性插值
    #
    # # 在点(x1, y1)与(sx, sy)之间进行线性插值
    # num_points2 = int(abs(sx - x1)) + 1  # 插值点的数量，根据距离差进行计算
    # x_values2 = np.linspace(x1, sx, num_points2)  # 在x轴方向上进行线性插值
    # y_values2 = np.linspace(y1, sy, num_points2)  # 在y轴方向上进行线性插值
    #
    # # 在点(x2, y2)与(gx, gy)之间进行线性插值
    # num_points3 = int(abs(gx - x2)) + 1  # 插值点的数量，根据距离差进行计算
    # x_values3 = np.linspace(x2, gx, num_points3)  # 在x轴方向上进行线性插值
    # y_values3 = np.linspace(y2, gy, num_points3)  # 在y轴方向上进行线性插值
    #
    # # 合并路径点，确保路径点从起点sx到终点gx有顺序
    # x_values = np.concatenate((x_values1, x_values2[1:], x_values3[1:]))
    # y_values = np.concatenate((y_values1, y_values2[1:], y_values3[1:]))
    # x_values = np.flip(x_values)
    # y_values = np.flip(y_values)

    # 将坐标轴顺时针旋转yaw0度，得到旋转后的x_values, y_values
    if ky == 1:
        x_values, y_values = rotate(x_values, y_values, -yaw0)

    return x_values, y_values, ky

# x_values, y_values, ky = path(0, 0, 100, 100, 0)
# plt.plot(x_values, y_values, 'r-')
# plt.show()