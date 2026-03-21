import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import time
# from multi_path_plan import path_test, path_hi_merge, path_turn_right, path_stri, path_turn_left, path_circle


# 多场景多障碍物求解
# 障碍物新增自行车、行人，且车辆长宽参数可调
# 加入规划路径长度限制，加入路径规划起点设置
# 全局路径自由输入
# 加入障碍车速度，并用作预测
# 加入障碍车航向角

class LocalPathPlan:
    def __init__(self):
        self.ROAD_CONFIG = {
            'width': 3.5,
            'lane_num': 2,
            's_max': 120.0,
            'n_points': 50
        }
        self.VEHICLE_CONFIG = {
            'min_turn_radius': 5.0,
            'width': 2.0,
            'safe_margin': 2.0,
            'max_heading_rate': 0.2
        }

    def init(self):
        self.ROAD_CONFIG = {
            'width': 3.5,
            'lane_num': 2,
            's_max': 120.0,
            'n_points': 50
        }
        self.VEHICLE_CONFIG = {
            'min_turn_radius': 5.0,
            'width': 2.0,
            'safe_margin': 2.0,
            'max_heading_rate': 0.2
        }

    def path_planning(self, obstacles, weights, node_num, fixed_distance, ref_path, road_model=None, init_heading=None):
        n = node_num
        delta = np.mean(np.linalg.norm(np.diff(ref_path, axis=0), axis=1))
        w_track, w_smooth, w_curvature, w_obstacle = weights

        x_ref, y_ref = ref_path[:, 0], ref_path[:, 1]
        if len(x_ref) > n:
            x_ref, y_ref = x_ref[:n], y_ref[:n]
        elif len(x_ref) < n:
            pad_x = np.full(n - len(x_ref), x_ref[-1])
            pad_y = np.full(n - len(y_ref), y_ref[-1])
            x_ref, y_ref = np.concatenate([x_ref, pad_x]), np.concatenate([y_ref, pad_y])

        x0 = np.concatenate([x_ref, y_ref])

        def objective(x):
            x_pts, y_pts = x[:n], x[n:]
            track_cost = w_track * np.mean((x_pts - x_ref) ** 2 + (y_pts - y_ref) ** 2)
            smooth_cost = w_smooth * np.sum(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2)
            ddx = x_pts[2:] - 2 * x_pts[1:-1] + x_pts[:-2]
            ddy = y_pts[2:] - 2 * y_pts[1:-1] + y_pts[:-2]
            curvature_cost = w_curvature * np.sum(ddx ** 2 + ddy ** 2) / (delta ** 4)

            # 避障代价：使用时间动态预测障碍物位置
            obstacle_cost = 0.0
            # for obs in obstacles:
            #     dxo = x_pts - obs.x_pos
            #     dyo = y_pts - obs.y_pos
            #     dist = np.hypot(dxo, dyo)
            #     thres = obs.width / 2 + VEHICLE_CONFIG['width'] / 2 + VEHICLE_CONFIG['safe_margin']
            #     mask = dist < thres
            #     if np.any(mask):
            #         obstacle_cost += w_obstacle * np.sum((1 - dist[mask] / thres) ** 2)

            for obs in obstacles:
                # dxo = x_pts - obs.x_pos
                # dyo = y_pts - obs.y_pos
                # dist = np.hypot(dxo, dyo)
                # thres = obs.width / 2 + VEHICLE_CONFIG['width'] / 2 + VEHICLE_CONFIG['safe_margin']
                # mask = dist < thres
                # if np.any(mask):
                #     obstacle_cost += w_obstacle * np.sum((1 - dist[mask] / thres) ** 2)

                t_series = np.linspace(0, (n * delta / obs.speed), round(n / 8))
                # print(len(t_series))
                x_pred, y_pred = obs.predict_position(t_series)
                for i in range(len(x_pred)):
                    dx = x_pts - x_pred[i]
                    dy = y_pts - y_pred[i]
                    dist = np.hypot(dx, dy)
                    # print(dist)
                    thres = obs.width / 2 + self.VEHICLE_CONFIG['width'] / 2 + self.VEHICLE_CONFIG['safe_margin']
                    # print(thres)
                    mask = dist < thres
                    if np.any(mask):
                        # print(mask)
                        obstacle_cost += w_obstacle * np.sum((1 - dist[mask] / thres) ** 2)
            # print(obstacle_cost)
            return track_cost + smooth_cost + curvature_cost + obstacle_cost

        k_fixed = min(int(fixed_distance / delta), n)
        constraints = []
        for i in range(k_fixed):
            constraints += [
                {'type': 'eq', 'fun': lambda x, i=i: x[i] - x_ref[i]},
                {'type': 'eq', 'fun': lambda x, i=i: x[n + i] - y_ref[i]},
            ]

        # # 航向角初始约束（仅在初始点定义）
        # if init_heading is not None and n >= 2:
        #     def heading_constraint(x):
        #         dx0 = x[1] - x[0]
        #         dy0 = x[n + 1] - x[n]
        #         return np.arctan2(dy0, dx0) - init_heading
        #
        #     constraints.append({'type': 'eq', 'fun': heading_constraint})

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-3, 'disp': False}
        )

        return result.x[:n], result.x[n:]

    def get_rotated_box_corners(cx, cy, length, width, heading):
        """获取绕中心旋转后的矩形四角坐标"""
        # 车辆半长宽
        dx = length / 2
        dy = width / 2

        # 局部坐标系下的四个点（逆时针）
        corners = np.array([
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy]
        ])

        # 旋转矩阵
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        # 平移 + 旋转
        rotated = (corners @ rot.T) + np.array([cx, cy])
        return rotated

    def visualize_xy_trajectories(self, trajectories, obstacles, road_model):
        plt.figure(figsize=(15, 8))
        plt.plot(road_model.global_path[:, 0], road_model.global_path[:, 1], 'k--', label='ref_path')

        colors = ['r', 'b', 'g', 'm', 'c']
        for idx, (x, y, label) in enumerate(trajectories):
            plt.plot(x, y, color=colors[idx % 5], linewidth=2, label=label)

        ind = 0
        for obs_group in obstacles:
            color = colors[ind % 5]
            ind += 1
            for obs in obs_group:
                # 当前矩形表示
                corners = self.get_rotated_box_corners(obs.x_pos, obs.y_pos, obs.length, obs.width, obs.heading)
                poly = Polygon(corners, color=color, alpha=0.3)
                plt.gca().add_patch(poly)

                # 航向角箭头
                arrow_len = 2.0
                dx = arrow_len * np.cos(obs.heading)
                dy = arrow_len * np.sin(obs.heading)
                plt.arrow(obs.x_pos, obs.y_pos, dx, dy, color=color, head_width=0.5)

                # 未来轨迹点预测
                t_series = np.linspace(0, 3, 10)
                x_pred, y_pred = obs.predict_position(t_series)
                plt.plot(x_pred, y_pred, linestyle=':', marker='o', color=color, alpha=0.7, label=f'Obs Future {ind}')

        plt.axis('equal')
        plt.title("Trajectory Planning with Dynamic Obstacle Prediction")
        plt.legend()
        plt.show()


class RoadModel:
    def __init__(self, local_path, global_path):
        self.global_path = np.array(global_path)
        self.local_path = local_path
        self.width = self.local_path.ROAD_CONFIG['width']
        self.lane_num = self.local_path.ROAD_CONFIG['lane_num']
        self.lane_centers = [i * self.width - (self.lane_num - 1) * self.width / 2
                             for i in range(self.lane_num)]

    def get_reference_points(self, start_idx=0, path_length=None, n_points=None):
        n_pts = n_points if n_points else self.local_path.ROAD_CONFIG['n_points']
        max_len = path_length if path_length else self.local_path.ROAD_CONFIG['s_max']

        total_len = 0
        ref_pts = [self.global_path[start_idx]]
        for i in range(start_idx + 1, len(self.global_path)):
            dx = self.global_path[i][0] - ref_pts[-1][0]
            dy = self.global_path[i][1] - ref_pts[-1][1]
            dist = np.hypot(dx, dy)
            total_len += dist
            if total_len > max_len:
                break
            ref_pts.append(self.global_path[i])

        if len(ref_pts) < n_pts:
            pad = np.repeat(ref_pts[-1][np.newaxis, :], n_pts - len(ref_pts), axis=0)
            ref_pts = np.vstack([ref_pts, pad])

        return np.array(ref_pts)


class DynamicObstacle:
    TYPE_CONFIG = {
        'pedestrian': {'length': 0.5, 'width': 0.5, 'speed': 1.2},
        'bicycle': {'length': 1.8, 'width': 0.6, 'speed': 4.0},
        'car': {'length': 4.8, 'width': 2.0, 'speed': 10.0}
    }

    def __init__(self, x_pos, y_pos, type='car', length=None, width=None, speed=None, heading=None):
        config = self.TYPE_CONFIG.get(type, self.TYPE_CONFIG['car'])
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.length = length if length else config['length']
        self.width = width if width else config['width']
        self.speed = speed if speed is not None else config['speed']
        self.heading = heading if heading is not None else 0  # 可选：加入障碍物朝向

    # def predict_position(self, t):
    #     dx = self.speed * t * np.cos(self.heading)
    #     dy = self.speed * t * np.sin(self.heading)
    #     return self.x_pos + dx, self.y_pos + dy

    def predict_position(self, t):
        dx = self.speed * t * np.cos(self.heading)
        dy = self.speed * t * np.sin(self.heading)
        return self.x_pos + dx, self.y_pos + dy


if __name__ == "__main__":
    start_time = time.time()

    path = [[s, 3.0 * np.sin(0.05 * s)] for s in np.linspace(0, 120, 200)]
    local_path_plan = LocalPathPlan
    local_path_plan.init(LocalPathPlan)
    # print(local_path_plan)
    # road = RoadModel(path)
    road = RoadModel(local_path_plan, path_circle)
    # road = RoadModel(path_turn_left)
    # plt.plot(road.global_path[:, 0], road.global_path[:, 1], 'k--', label='ref_path')
    # plt.show()

    start_idx = 65
    path_length = 20
    ref_path_length = 50
    ref = road.get_reference_points(start_idx=start_idx, path_length=ref_path_length)
    delta = np.mean(np.linalg.norm(np.diff(ref, axis=0), axis=1))
    node_num = round(path_length / delta)

    # 初始航向角（弧度）
    # init_heading = np.arctan2(ref[1, 1] - ref[0, 1], ref[1, 0] - ref[0, 0])
    init_heading = 0

    # obs_list = [
    #     DynamicObstacle(x_pos=45, y_pos=2.0, type='car', speed=8.0)
    # ]
    # obs_list = [
    #     DynamicObstacle(x_pos=48, y_pos=3.0, type='car', speed=8.0),
    #     DynamicObstacle(x_pos=50, y_pos=1.0, type='pedestrian'),
    #     DynamicObstacle(x_pos=55, y_pos=1.5, type='bicycle')
    # ]
    obs_list = [
        DynamicObstacle(x_pos=100.82, y_pos=62.99, type='car', speed=2.06, heading=6.106),
        DynamicObstacle(x_pos=122.53, y_pos=85.68, type='car', speed=4.69, heading=3.778),
        DynamicObstacle(x_pos=92.89, y_pos=63.44, type='car', speed=0.92, heading=6.283),
        DynamicObstacle(x_pos=140.21, y_pos=116.16, type='car', speed=4.95, heading=4.34),
        # DynamicObstacle(x_pos=175, y_pos=-4.5, type='car'),
        # DynamicObstacle(x_pos=180, y_pos=-4, type='bicycle')
    ]

    x_opt, y_opt = local_path_plan.path_planning(LocalPathPlan, obs_list, (50, 100, 2000, 1000), node_num,
                                                 fixed_distance=5.0, ref_path=ref,
                                                 road_model=road, init_heading=init_heading)

    obs_list_2 = [
        DynamicObstacle(x_pos=45, y_pos=2.0, type='car', speed=6.0),
        DynamicObstacle(x_pos=52, y_pos=1.8, type='pedestrian', speed=1.5),
        DynamicObstacle(x_pos=56, y_pos=-0.5, type='bicycle', speed=3.0)
    ]

    x_opt_2, y_opt_2 = local_path_plan.path_planning(LocalPathPlan, obs_list_2, (50, 100, 2000, 1000), node_num,
                                                     fixed_distance=5.0, ref_path=ref,
                                                     road_model=road, init_heading=init_heading)

    print('cost time:', round(time.time() - start_time, 2), 's')

    local_path_plan.visualize_xy_trajectories(LocalPathPlan, [(x_opt, y_opt, 'Trajectory for Scenario1'),
                                                              (x_opt_2, y_opt_2, 'Trajectory for Scenario2')
                                                              ],
                                              [obs_list, obs_list_2],
                                              road)
