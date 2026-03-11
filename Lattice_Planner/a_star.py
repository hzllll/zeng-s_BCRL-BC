import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from path import path
import JCK


# show_animation = True

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                # print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # # show graph
            # if show_animation:  # pragma: no cover
            #     plt.plot(self.calc_grid_position(current.x, self.min_x),
            #              self.calc_grid_position(current.y, self.min_y), "xc")
            #     # for stopping simulation with the esc key.
            #     plt.gcf().canvas.mpl_connect('key_release_event',
            #                                  lambda event: [exit(
            #                                      0) if event.key == 'escape' else None])
            #     if len(closed_set.keys()) % 10 == 0:
            #         plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        dx = abs(n1.x - n2.x)
        dy = abs(n1.y - n2.y)
        d = dx + dy
        return d
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width+1)]
                             for _ in range(self.x_width+1)]
        for iox, ioy in zip(ox, oy):
            ix = int((iox - self.min_x)/self.resolution)
            iy = int((ioy - self.min_y)/self.resolution)
            self.obstacle_map[ix][iy] = True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

def decrease(a):
    if a > 0:
        a = int(a)
    else:
        a = int(a) - 1
    return a
def add(a):
    if a > 0:
        a = int(a) + 1
    else:
        a = int(a)
    return a

def path_generate(sx, sy, gx, gy, lanes, grid_size = 1.0, rr = 1.0):
    # print(__file__ + " start!!")
    # Find the minimum and maximum x and y coordinates in the lanes array
    min_x = decrease(np.min(lanes[:, 0]))
    max_x = add(np.max(lanes[:, 0]))
    min_y = decrease(np.min(lanes[:, 1]))
    max_y = add(np.max(lanes[:, 1]))

    ox, oy = [], []
    # Add the boundary points of the rectangle to the obstacle list
    for x in np.arange(min_x, max_x + grid_size, grid_size):
        oy.append(min_y)
        oy.append(max_y)
        ox.append(x)
        ox.append(x)
    for y in np.arange(min_y, max_y + grid_size, grid_size):
        ox.append(min_x)
        ox.append(max_x)
        oy.append(y)
        oy.append(y)

    # 将现有的障碍物坐标点添加到 grid_points 集合中
    grid_points = set(zip(ox, oy))
    # 将已有的障碍物坐标点添加到 obstacles 列表中
    obstacles = list(grid_points)
    num_lanes = len(lanes)
    for i in range(num_lanes):
        x, y = lanes[i]
        x_idx = (x - min_x) // grid_size
        y_idx = (y - min_y) // grid_size
        grid_points_to_add = [(min_x + x_idx * grid_size, min_y + y_idx * grid_size),
                              (min_x + (x_idx + 1) * grid_size, min_y + y_idx * grid_size),
                              (min_x + x_idx * grid_size, min_y + (y_idx + 1) * grid_size),
                              (min_x + (x_idx + 1) * grid_size, min_y + (y_idx + 1) * grid_size)]
        for point in grid_points_to_add:
            if point not in grid_points:
                obstacles.append(point)
                grid_points.add(point)
    # 将新的障碍物坐标点转换为数组形式
    ox, oy = zip(*obstacles)

    # if show_animation:  # pragma: no cover
    #     plt.plot(ox, oy, ".k")
    #     plt.plot(sx, sy, "og")
    #     plt.plot(gx, gy, "xb")
    #     plt.grid(True)
    #     plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, rr)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    return rx, ry

class Spline2D:

    def __init__(self, x, y, kind="cubic"):
        self.s = self.__calc_s(x, y)
        self.sx = interpolate.interp1d(self.s, x, kind=kind)
        self.sy = interpolate.interp1d(self.s, y, kind=kind)

    def __calc_s(self, x, y):
        self.ds = np.hypot(np.diff(x), np.diff(y))
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        x = self.sx(s)
        y = self.sy(s)
        return x, y

def cubic_spline(x, y):
    ds = 0.1  # [m] distance of each interpolated points
    for (kind, label) in [
                          # ("quadratic", "C0 & C1 (Quadratic spline)"),
                          ("cubic", "C0 & C1 & C2 (Cubic spline)")
                          ]:
        # 等间隔采样x, y，并保留起始点和终点
        x = [x[0]] + x[1:-1:5] + [x[-1]]
        y = [y[0]] + y[1:-1:5] + [y[-1]]
        rx, ry = [], []
        sp = Spline2D(x, y, kind=kind)
        s = np.arange(0, sp.s[-1], ds)
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
    return rx, ry

def a_star_global_path(start_goal, lanes, yaw0):
    sx, sy, gx, gy = [start_goal[i] for i in range(4)]
    # grid_size = 1
    # rr = 1.5
    # # 清除以起点和终点为中心边长为2.5的正方形区域内的障碍物
    # square_size = 4
    # square_min_x1 = sx - square_size/2
    # square_max_x1 = sx + square_size/2
    # square_min_y1 = sy - square_size/2
    # square_max_y1 = sy + square_size/2
    # # 过滤掉在正方形区域内的障碍物点
    # lanes = [point for point in lanes if not(square_min_x1 <= point[0] <= square_max_x1 and square_min_y1 <= point[1] <= square_max_y1)]
    # square_min_x2 = gx - square_size/2
    # square_max_x2 = gx + square_size/2
    # square_min_y2 = gy - square_size/2
    # square_max_y2 = gy + square_size/2
    # # 过滤掉在正方形区域内的障碍物点
    # lanes = [point for point in lanes if not(square_min_x2 <= point[0] <= square_max_x2 and square_min_y2 <= point[1] <= square_max_y2)]
    # lanes = np.array(lanes)
    #
    # path_x, path_y = path_generate(sx, sy, gx, gy, lanes, grid_size, rr)
    # dx = gx - path_x[0]; dy = gy - path_y[0]
    # path_x = [x + dx for x in path_x]; path_y = [y + dy for y in path_y]
    # path_x[-1] = sx; path_y[-1] = sy
    # try:
    #     path_x, path_y = cubic_spline(path_x, path_y)
    # except:
    #     lanes = JCK.JCK_point(lanes)
    #     # 过滤掉在正方形区域内的障碍物点
    #     lanes = [point for point in lanes if not (square_min_x1 <= point[0] <= square_max_x1 and square_min_y1 <= point[1] <= square_max_y1)]
    #     lanes = [point for point in lanes if not (square_min_x2 <= point[0] <= square_max_x2 and square_min_y2 <= point[1] <= square_max_y2)]
    #     lanes = np.array(lanes)
    #
    #     path_x, path_y = path_generate(sx, sy, gx, gy, lanes, grid_size)
    #     dx = gx - path_x[0]; dy = gy - path_y[0]
    #     path_x = [x + dx for x in path_x]; path_y = [y + dy for y in path_y]
    #     path_x[-1] = sx; path_y[-1] = sy
    #     try:
    #         path_x, path_y = cubic_spline(path_x, path_y)
    #     except:
    #         # 在sx,sy和gx,gy之间进行线性插值
    #         path_x = np.linspace(sx, gx, 100)
    #         path_y = np.linspace(sy, gy, 100)
    
    path_x, path_y, ky = path(sx, sy, gx, gy, yaw0)

    return path_x, path_y, ky

class global_path():

    def __init__(self, obs, scenario):


        self.start = {
            'x' : obs['vehicle_info']['ego']['x'],
            'y' : obs['vehicle_info']['ego']['y'],
            'yaw' : obs['vehicle_info']['ego']['yaw']
        }
        self.goal = obs['test_setting']['goal']
        self.hyperparameters = {
            # 'max_v': 50.0,
            'max_v': 40.0,
            # 'max_v': 35.0,
            'goal_extend': 0.0
        }

    def global_path_planning(self, road_info, yaw0):

        # start_goal = np.array(
        #     [self.start['x'], self.start['y'], \
        #      np.mean(self.goal['x']), np.mean(self.goal['y'])])

        # 多50m
        if np.mean(self.goal['x']) > self.start['x']:
            start_goal = np.array(
                [self.start['x'], self.start['y'], \
                 np.mean(self.goal['x']) + 50, np.mean(self.goal['y'])])
        else:
            start_goal = np.array(
                [self.start['x'], self.start['y'], \
                 np.mean(self.goal['x']) - 50, np.mean(self.goal['y'])])

        all_points_np = self.get_discretelanes(road_info)
        # all_points_np = 0
        x_global, y_global, ky = a_star_global_path(start_goal, all_points_np, yaw0)
        x_global = x_global[::-1]
        y_global = y_global[::-1]
        # plt.plot(x_global, y_global, 'r')
        # yaw_global = np.arctan2(np.diff(y_global), np.diff(x_global))
        # yaw_global = np.append(yaw_global, yaw_global[-1])
        # yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
        v_max_global = np.ones(len(x_global)) * self.hyperparameters['max_v']
        waypoints = np.stack((x_global, y_global, v_max_global), axis=1)

        return waypoints, ky

    def get_discretelanes(self, road_info):

        # 将观察值中的discrete_lanes转换为字典，方便后续处理
        discrete_lanes_dict = road_info._discretelanes
        discrete_lanes = {}
        for i, key in enumerate(discrete_lanes_dict):
            discrete_lane = discrete_lanes_dict[key]
            discrete_lane_dict = vars(discrete_lane)
            discrete_lanes[i] = discrete_lane_dict

        # 创建一个空的集合，用于存储所有点
        all_points = set()

        # 遍历discrete_lanes列表中的每个字典元素
        for lane_dict in discrete_lanes.values():
            # 获取左侧和右侧顶点的列表
            left_vertices = lane_dict['_left_vertices']
            right_vertices = lane_dict['_right_vertices']
            # 将左侧和右侧顶点的列表合并成一个列表
            vertices = set([tuple(point) for point in left_vertices] + [tuple(point) for point in right_vertices])
            # 遍历所有顶点，并将它们添加到集合中
            for point in vertices:
                if tuple(point) not in all_points:
                    all_points.add(tuple(point))
                else:
                    all_points.remove(tuple(point))

        # 将集合中的所有点转换为Numpy数组
        all_points_np = np.array(list(all_points))

        return all_points_np