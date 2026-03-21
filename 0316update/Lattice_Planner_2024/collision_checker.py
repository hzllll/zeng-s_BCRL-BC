import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt
import shapely.geometry as sg
from rtree import index

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight

    # '''
    def collision_check(self, paths, obstacles_1, obstacles_2, obstacles_3, self_Car):

        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            collision_free_1 = True
            collision_free_2 = True
            collision_free_3 = True
            collision_free_4 = True
            collision_free_5 = True
            collision_free_6 = True
            path           = paths[i]

            # Iterate over the points in the path.
            # lead car
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_3 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_4 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets * sin(path[2][j])
                circle_locations_2[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 0.5
                circle_locations_2[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 0.5
                circle_locations_3[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.0
                circle_locations_3[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.0
                circle_locations_4[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.5
                circle_locations_4[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.5

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations_3)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations_4)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                if not collision_free_1:
                    break


            # follow
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_3 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_4 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets * sin(path[2][j])
                circle_locations_2[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 0.8
                circle_locations_2[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 0.8
                circle_locations_3[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.0
                circle_locations_3[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.0
                circle_locations_4[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.5
                circle_locations_4[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.5

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations_3)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations_4)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                if not collision_free_2:
                    break

            # side lead
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_3 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_4 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets * sin(path[2][j])
                circle_locations_2[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 0.3
                circle_locations_2[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 0.3
                circle_locations_3[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.0
                circle_locations_3[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.0
                circle_locations_4[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.5
                circle_locations_4[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.5

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations_3)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations_4)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                if not collision_free_3:
                    break


            if collision_free_1 == collision_free_2 == collision_free_3:
                collision_free = collision_free_1
            else:
                collision_free = False

            collision_check_array[i] = collision_free

        return collision_check_array
    # '''

    '''
    def collision_check(self, paths, obstacles_1, obstacles_2, obstacles_3, self_Car):

        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            collision_free_1 = True
            collision_free_2 = True
            collision_free_3 = True
            collision_free_4 = True
            collision_free_5 = True
            collision_free_6 = True
            path           = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break
                if not collision_free_1:
                    break



            for m in range(len(path[0])):

                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations_2[:, 0] = path[0][m] + circle_offsets*cos(path[2][m]) + self_Car[0] * np.cos(self_Car[1]) * 1.5
                circle_locations_2[:, 1] = path[1][m] + circle_offsets*sin(path[2][m]) + self_Car[0] * np.sin(self_Car[1]) * 1.5

                # n=len(obstacles)/2
                for n in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[n],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break
                if not collision_free_2:
                    break

            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break
                if not collision_free_3:
                    break



            for m in range(len(path[0])):

                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations_2[:, 0] = path[0][m] + circle_offsets*cos(path[2][m]) + self_Car[0] * np.cos(self_Car[1]) * 1.0
                circle_locations_2[:, 1] = path[1][m] + circle_offsets*sin(path[2][m]) + self_Car[0] * np.sin(self_Car[1]) * 1.0

                # n=len(obstacles)/2
                for n in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[n],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_4 = collision_free_4 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_4:
                        break
                if not collision_free_4:
                    break

            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_5 = collision_free_5 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_5:
                        break
                if not collision_free_5:
                    break



            for m in range(len(path[0])):

                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations_2[:, 0] = path[0][m] + circle_offsets*cos(path[2][m]) + self_Car[0] * np.cos(self_Car[1]) * 0.5
                circle_locations_2[:, 1] = path[1][m] + circle_offsets*sin(path[2][m]) + self_Car[0] * np.sin(self_Car[1]) * 0.5


                for n in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[n],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_6 = collision_free_6 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_6:
                        break
                if not collision_free_6:
                    break


            if collision_free_1 == collision_free_2 == collision_free_3 == collision_free_4 == collision_free_5 == collision_free_6:
                collision_free = collision_free_1
            else:
                collision_free = False

            collision_check_array[i] = collision_free

        return collision_check_array
    '''


    def select_best_path_index(self, paths, collision_check_array):

        best_index = int(len(paths) / 2)
        best_score = float('Inf')
        score = best_score
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                mid = int(len(paths) / 2)
                diff = i - mid

                score = np.sqrt(diff ** 2)
                # if i == 1:
                #     score = score - 3
                #
                # if i == 9:
                #     score = score - 3

            if score < best_score:
                best_score = score
                best_index = i

        if best_score == float('Inf'):
            best_index = int(len(paths) / 2)

        return best_index

    def select_best_path_index_city(self, paths, collision_check_array, global_path, close_index):

        best_index = int(len(paths) / 2)
        best_score = float('Inf')
        score = best_score
        dis_gap = float('Inf')
        gap_path = ((paths[0][0][1] - paths[0][0][0]) ** 2 + (paths[0][1][1] - paths[0][1][0]) ** 2) ** 0.5
        if close_index > 1:
            gap_global = ((global_path[close_index][0] - global_path[close_index - 1][0]) ** 2 + (global_path[close_index][1] - global_path[close_index - 1][1]) ** 2) ** 0.5
        else:
            gap_global = ((global_path[min((close_index + 1), len(global_path))][0] - global_path[close_index][0]) ** 2 + (global_path[min((close_index + 1), len(global_path))][1] - global_path[close_index][1]) ** 2) ** 0.5
        # dis_ju = gap_path / gap_global
        dis_ju = gap_global / gap_path
        if dis_ju > 10 or dis_ju < 0:
            dis_ju = 1

        # print('dis_ju:', dis_ju)
        # dis_ju = 0.5
        for i in range(len(paths)):
            if collision_check_array[i]:
                dis_path = 0
                m = 1
                for j in range(min(len(paths[i][0]), len(paths[i][1]))):
                    if j % 5 == 0 and j > 0:
                        dis = ((paths[i][0][j] - global_path[min(round(close_index + 5 * m * dis_ju), len(global_path) - 1)][0]) ** 2 + (paths[i][1][j] - global_path[min(round(close_index + 5 * m * dis_ju), len(global_path) - 1)][1]) ** 2) ** 0.5
                        # print('dis:', dis)
                        dis_path += dis
                        m += 1
            # print('dis_path:', dis_path)

                if dis_path < dis_gap:
                    dis_gap = dis_path
                    best_index = i

        if dis_gap == float('Inf'):
            best_index = int(len(paths) / 2)

        return best_index

    def select_best_path_index_ped(self, paths, collision_check_array, global_path, close_index):

        best_index = int(len(paths) / 2)
        best_score = float('Inf')
        score = best_score
        dis_gap = float('Inf')
        gap_path = ((paths[0][0][1] - paths[0][0][0]) ** 2 + (paths[0][1][1] - paths[0][1][0]) ** 2) ** 0.5
        if close_index > 1:
            gap_global = ((global_path[close_index][0] - global_path[close_index - 1][0]) ** 2 + (global_path[close_index][1] - global_path[close_index - 1][1]) ** 2) ** 0.5
        else:
            gap_global = ((global_path[min((close_index + 1), len(global_path))][0] - global_path[close_index][0]) ** 2 + (global_path[min((close_index + 1), len(global_path))][1] - global_path[close_index][1]) ** 2) ** 0.5
        dis_ju = gap_path / gap_global
        if dis_ju > 10 or dis_ju < 0:
            dis_ju = 1

        # print('dis_ju:', dis_ju)
        # dis_ju = 0.5
        for i in range(len(paths)):
            dis_path = 0
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                m = 1
                for j in range(min(len(paths[i][0]), len(paths[i][1]))):
                    if j % 5 == 0 and j > 0:
                        dis = ((paths[i][0][j] - global_path[min(round(close_index + 5 * m * dis_ju), len(global_path))][0]) ** 2 + (paths[i][1][j] - global_path[min(round(close_index + 5 * m * dis_ju), len(global_path))][1]) ** 2) ** 0.5
                        # print('dis:', dis)
                        dis_path += dis
                        m += 1
                # print('dis_path:', dis_path)

                if dis_path < dis_gap:
                    dis_gap = dis_path
                    best_index = i

        if dis_gap == float('Inf'):
            best_index = int(len(paths) / 2)

        return best_index

    # def preprocess_map(self, map_data):
    #     """预处理地图数据，生成R树索引和四边形列表"""
    #     idx = index.Index()
    #     shapes = []
    #     for road in map_data:
    #         left = road['left']
    #         right = road['right']
    #         n = len(left)
    #         for i in range(n - 1):
    #             # 获取四边形顶点
    #             ll = left[i]
    #             lr = left[i+1]
    #             rr = right[i+1]
    #             rl = right[i]
    #             # 创建多边形
    #             try:
    #                 polygon = sg.Polygon([ll, lr, rr, rl])
    #             except Exception:
    #                 continue
    #             if not polygon.is_valid:
    #                 polygon = polygon.buffer(0)
    #                 if not polygon.is_valid or polygon.is_empty:
    #                     continue
    #             # 插入到索引
    #             bounds = polygon.bounds
    #             shapes.append(polygon)
    #             idx.insert(len(shapes) - 1, bounds)
    #     return idx, shapes
    def preprocess_map(self, map_data):
        """预处理地图数据，生成R树索引和四边形列表（修复自相交问题）"""
        idx = index.Index()
        shapes = []
        for road in map_data:
            left = road['left']
            right = road['right']
            n = len(left)
            if len(right) != n:
                # logging.warning(f"Road {road['id']} left/right point count mismatch")
                continue  # 跳过点数不匹配路段

            for i in range(n - 1):
                # 调整顶点顺序：左当前点 -> 右当前点 -> 右下一节点 -> 左下一节点
                ll = left[i]
                rl = right[i]
                rr = right[i + 1]
                lr = left[i + 1]

                # 创建多边形并验证几何有效性
                try:
                    polygon = sg.Polygon([ll, rl, rr, lr])
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                except Exception as e:
                    # logging.debug(f"几何创建失败于 {road['id']}段{i}: {str(e)}")
                    continue

                # 跳过无效或空几何
                if polygon.is_empty or not polygon.is_valid:
                    continue

                # 插入到索引
                bounds = polygon.bounds
                shapes.append(polygon)
                idx.insert(len(shapes) - 1, bounds)
        return idx, shapes

    def check_paths(self, path, map_data):
        """判断每条局部路径是否在道路范围内"""
        idx, shapes = self.preprocess_map(map_data)
        path_road = []
        for path_idx in range(len(path)):
            current_path = path[path_idx]
            xs = current_path[0]
            ys = current_path[1]
            all_valid = True
            for x, y in zip(xs, ys):
                point = sg.Point(x, y)
                hits = list(idx.intersection((x, y, x, y)))
                found = False
                for hit in hits:
                    if shapes[hit].contains(point):
                        found = True
                        break
                if not found:
                    all_valid = False
                    break
            path_road.append(all_valid)
        return path_road

# 现有一系列局部路径，格式为path[i]=[[x1,x2...],[y1,y2...]],i为0-8，即共有9条局部路径，同时给出全局地图,格式为map=[{'id': '0.0.-1.-1', 'mid':[[x1,y1],[x2,y2]...],'left': [[x1,y1],[x2,y2]...], 'right':[[x1,y1],[x2,y2]...],'next_id': ['1.0.-1.-1'], 'pre_id': []},{...}...]，其中坐标按照道路方向顺序排列,要求判断每个局部路径是否在道路范围内，有时map的体量会非常大，mid等类型中的点个数极多，要求在保证计算准确性的情况下减少运算量，保证时效性，要求生成一个python函数，输入为path，map，输出为每条局部路径是否在地图范围内，例如若第0条路径不在道路范围内，则输出path_road[0]=False,若在道路范围内，则输出True
# [{'id': '0.0.-1.-1', 'mid': [[2.772508439493203, -1.061170278635529], [2.5087340562435942, -0.6355923140986743], [2.2449596729939865, -0.21001434956181964], [1.9811852897443778, 0.2155636149750353], [1.7174109064947696, 0.6411415795118898], [1.4536365232451613, 1.0667195440487447], [1.1898621399955527, 1.4922975085855996], [0.9260877567459445, 1.917875473122454], [0.6623133734963362, 2.3434534376593086], [0.398538990246728, 2.7690314021961635], [0.13476460699711978, 3.1946093667330184], [-0.12900977625248888, 3.620187331269873], [-0.39278415950209755, 4.045765295806728], [-0.6565585427517058, 4.471343260343582], [-0.920332926001314, 4.896921224880437], [-1.1841073092509222, 5.322499189417291], [-1.4478816925005304, 5.748077153954146], [-1.7116560757501391, 6.173655118491002], [-1.975430458999747, 6.5992330830278565], [-2.2392048422493556, 7.024811047564709], [-2.5029792254989633, 7.4503890121015655], [-2.766753608748572, 7.875966976638421], [-3.0305279919981807, 8.301544941175274], [-3.2943023752477885, 8.727122905712129], [-3.558076758497398, 9.152700870248985], [-3.821851141747006, 9.57827883478584], [-4.0856255249966145, 10.003856799322694], [-4.349399908246222, 10.429434763859549], [-4.613174291495831, 10.855012728396403], [-4.87694867474544, 11.280590692933258], [-5.140723057995047, 11.706168657470112], [-5.404497441244656, 12.131746622006967], [-5.668271824494264, 12.557324586543821], [-5.932046207743873, 12.982902551080677], [-6.195820590993481, 13.408480515617532], [-6.459594974243091, 13.834058480154386], [-6.723369357492697, 14.25963644469124], [-6.987143740742306, 14.685214409228097], [-7.250918123991914, 15.110792373764948], [-7.514692507241524, 15.536370338301804], [-7.77846689049113, 15.96194830283866], [-8.04224127374074, 16.387526267375513], [-8.306015656990347, 16.81310423191237], [-8.569790040239956, 17.238682196449226], [-8.833564423489564, 17.664260160986082], [-9.097338806739172, 18.089838125522938], [-9.36111318998878, 18.515416090059787], [-9.62488757323839, 18.940994054596644], [-9.888661956487999, 19.3665720191335], [-10.152436339737605, 19.792149983670356], [-10.416210722987215, 20.217727948207212], [-10.679985106236822, 20.64330591274406], [-10.943759489486432, 21.068883877280918], [-11.20753387273604, 21.494461841817774], [-11.471308255985647, 21.92003980635463], [-11.735082639235255, 22.34561777089148], [-11.998857022484865, 22.771195735428336], [-12.262631405734473, 23.196773699965192], [-12.526405788984082, 23.622351664502048], [-12.790180172233688, 24.047929629038897], [-13.053954555483298, 24.473507593575754], [-13.317728938732904, 24.89908555811261], [-13.581503321982515, 25.324663522649466], [-13.845277705232123, 25.750241487186315], [-14.109052088481729, 26.17581945172317], [-14.37282647173134, 26.601397416260028], [-14.636600854980951, 27.026975380796884], [-14.900375238230556, 27.45255334533374], [-15.164149621480167, 27.878131309870597], [-15.427924004729771, 28.303709274407446], [-15.691698387979383, 28.729287238944302], [-15.955472771228987, 29.154865203481158], [-16.219247154478598, 29.580443168018014], [-16.48302153772821, 30.00602113255487], [-16.746795920977814, 30.431599097091727], [-17.010570304227418, 30.857177061628576], [-17.27434468747703, 31.282755026165425], [-17.53811907072664, 31.70833299070228], [-17.801893453976252, 32.13391095523914], [-18.065667837225856, 32.559488919775994], [-18.32944222047546, 32.98506688431285], [-18.59321660372507, 33.4106448488497], [-18.856990986974683, 33.836222813386556], [-19.120765370224294, 34.26180077792341], [-19.3845397534739, 34.68737874246027], [-19.648314136723503, 35.112956706997124], [-19.912088519973114, 35.53853467153398], [-20.175862903222725, 35.96411263607084], [-20.43963728647233, 36.389690600607686], [-20.70341166972194, 36.81526856514454], [-20.967186052971545, 37.2408465296814], [-21.230960436221157, 37.666424494218255], [-21.49473481947076, 38.092002458755104], [-21.758509202720372, 38.51758042329196], [-22.022283585969983, 38.94315838782882], [-22.286057969219588, 39.36873635236567], [-22.5498323524692, 39.79431431690253], [-22.813606735718803, 40.21989228143938], [-23.077381118968415, 40.645470245976234], [-23.341155502218026, 41.07104821051309], [-23.60492988546763, 41.49662617504995], [-23.86870426871724, 41.922204139586796], [-24.132478651966846, 42.34778210412365], [-24.396253035216457, 42.77336006866051], [-24.66002741846607, 43.198938033197365], [-24.923801801715673, 43.624515997734214], [-25.18797663656443, 44.050545599628805], [-25.45185227584275, 44.47606078872727], [-25.71572791512107, 44.90157597782573], [-25.97960355439939, 45.32709116692419], [-26.24347919367771, 45.75260635602265], [-26.507354832956032, 46.178121545121115], [-26.771230472234354, 46.60363673421959], [-27.035106111512675, 47.029151923318054], [-27.298981750790993, 47.454667112416516], [-27.562857390069315, 47.88018230151498], [-27.826733029347636, 48.30569749061344], [-28.090608668625958, 48.7312126797119], [-28.354484307904276, 49.156727868810364], [-28.618359947182594, 49.582243057908826], [-28.882235586460915, 50.0077582470073], [-29.146111225739237, 50.433273436105765], [-29.409986865017554, 50.85878862520423], [-29.673862504295876, 51.28430381430269], [-29.937738143574197, 51.70981900340115], [-30.201613782852522, 52.13533419249961], [-30.465489422130833, 52.560849381598075], [-30.72936506140916, 52.98636457069654], [-30.993240700687483, 53.411879759795], [-31.257116339965794, 53.83739494889346], [-31.52099197924412, 54.26291013799194], [-31.784867618522437, 54.6884253270904], [-32.048743257800766, 55.11394051618886], [-32.312618897079076, 55.539455705287324], [-32.57587600442548, 55.96406426728903], [-32.8380344585942, 56.38769829020557], [-33.09976856016992, 56.81159462250501], [-33.361078046692484, 57.23575283911515], [-33.621962656127515, 57.66017251470114], [-33.88242212686671, 58.08485322366593], [-34.142456197728066, 58.50979454015075], [-34.40249096531468, 58.93564830026819], [-34.66332311788865, 59.36303584962085], [-34.92415527046262, 59.790423398973516], [-35.18498742303659, 60.217810948326175], [-35.44581957561056, 60.64519849767885], [-35.70665172818451, 61.07258604703151], [-35.96748388075848, 61.499973596384166], [-36.22831603333245, 61.92736114573684], [-36.489148185906416, 62.3547486950895], [-36.74998033848037, 62.78213624444216], [-37.01081249105434, 63.20952379379483], [-37.271644643628306, 63.63691134314749], [-37.532476796202275, 64.06429889250016], [-37.79330894877624, 64.49168644185282], [-38.0541411013502, 64.91907399120548], [-38.31497325392418, 65.34646154055814], [-38.575805406498134, 65.7738490899108], [-38.8366375590721, 66.20123663926346], [-39.09746971164607, 66.62862418861613], [-39.35830186422004, 67.05601173796879], [-39.61913401679401, 67.48339928732146], [-39.87996616936796, 67.91078683667412], [-40.14079832194193, 68.33817438602678], [-40.4016304745159, 68.76556193537945], [-40.662462627089866, 69.19294948473211], [-40.923294779663834, 69.62033703408477], [-41.18412693223779, 70.04772458343743], [-41.44495908481176, 70.47511213279009], [-41.705791237385725, 70.90249968214276], [-41.96662338995969, 71.32988723149542], [-42.22745554253365, 71.75727478084808], [-42.48828769510763, 72.18466233020075], [-42.749119847681584, 72.61204987955341], [-43.00995200025555, 73.03943742890608], [-43.27078415282952, 73.46682497825874], [-43.53201351309231, 73.89482252002551], [-43.79448980353301, 74.32413931166691], [-44.057395874166296, 74.75319304799012], [-44.32073146135681, 75.18198329875125], [-44.58449630103844, 75.61050963397055], [-44.848690128714665, 76.03877162393293], [-45.11331267945879, 76.4667688391884], [-45.37836368791415, 76.89450085055248], [-45.643842888294515, 77.32196722910656], [-45.909750014384244, 77.74916754619852], [-46.17608479953854, 78.17610137344295], [-46.442846976683825, 78.60276828272168], [-46.710036278317894, 79.02916784618424], [-46.97765243651026, 79.45529963624818], [-47.2456951829024, 79.88116322559966], [-47.514164248708, 80.3067581871937], [-47.78305936471325, 80.73208409425472], [-48.05238026127711, 81.15714052027695], [-48.32212666833158, 81.58192703902483], [-48.59075095164587, 82.00447443119407], [-48.859368931241264, 82.42701189722061], [-49.127986910836654, 82.84954936324715], [-49.39660489043206, 83.27208682927369], [-49.66522287002745, 83.69462429530023], [-49.933840849622854, 84.1171617613268], [-50.202458829218244, 84.53969922735334], [-50.471076808813635, 84.96223669337985], [-50.73969478840904, 85.38477415940642], [-51.00831276800443, 85.80731162543296], [-51.27693074759982, 86.2298490914595], [-51.545548727195225, 86.65238655748604], [-51.814166706790616, 87.07492402351258], [-52.082784686386006, 87.49746148953912], [-52.3514026659814, 87.91999895556566], [-52.6200206455768, 88.3425364215922], [-52.88863862517219, 88.76507388761874], [-53.15725660476758, 89.18761135364528], [-53.42587458436299, 89.61014881967185], [-53.69449256395838, 90.03268628569839], [-53.96311054355377, 90.45522375172493], [-54.23172852314917, 90.87776121775147], [-54.50034650274456, 91.30029868377801], [-54.76896448233995, 91.72283614980455], [-55.03758246193536, 92.14537361583109], [-55.30620044153075, 92.56791108185763], [-55.57481842112615, 92.99044854788417], [-55.843436400721544, 93.41298601391071], [-56.112054380316934, 93.83552347993725], [-56.38067235991234, 94.25806094596379], [-56.64929033950773, 94.68059841199033], [-56.916980358728324, 95.10186009306369], [-57.183779785279654, 95.52258673407475], [-57.45015776832241, 95.94358033333089], [-57.716114040739654, 96.3648404686707], [-57.98164833583735, 96.78636671766542], [-58.24676038734458, 97.20815865761946], [-58.51144992941388, 97.63021586557086], [-58.777168915768655, 98.05443284803604]], 'left': [[0.6475639973760021, -2.3782166399741183], [0.3837896141263938, -1.9526386754372635], [0.12001523087678545, -1.5270607109004088], [-0.143759152372823, -1.101482746363554], [-0.4075335356224312, -0.6759047818266994], [-0.6713079188720394, -0.2503268172898445], [-0.9350823021216481, 0.1752511472470104], [-1.1988566853712563, 0.6008291117838649], [-1.4626310686208646, 1.0264070763207194], [-1.7264054518704728, 1.4519850408575743], [-1.990179835120081, 1.8775630053944292], [-2.2539542183696897, 2.3031409699312837], [-2.5177286016192983, 2.728718934468139], [-2.7815029848689066, 3.1542968990049935], [-3.045277368118515, 3.579874863541848], [-3.309051751368123, 4.0054528280787025], [-3.5728261346177312, 4.431030792615557], [-3.83660051786734, 4.856608757152412], [-4.100374901116948, 5.282186721689267], [-4.364149284366556, 5.70776468622612], [-4.627923667616164, 6.133342650762977], [-4.891698050865773, 6.558920615299831], [-5.1554724341153815, 6.984498579836686], [-5.419246817364989, 7.41007654437354], [-5.683021200614599, 7.835654508910396], [-5.946795583864207, 8.26123247344725], [-6.210569967113815, 8.686810437984105], [-6.474344350363423, 9.11238840252096], [-6.738118733613032, 9.537966367057814], [-7.00189311686264, 9.963544331594669], [-7.265667500112248, 10.389122296131523], [-7.529441883361857, 10.814700260668378], [-7.793216266611465, 11.240278225205232], [-8.056990649861074, 11.665856189742088], [-8.320765033110682, 12.091434154278943], [-8.584539416360291, 12.517012118815797], [-8.848313799609898, 12.942590083352652], [-9.112088182859507, 13.368168047889508], [-9.375862566109115, 13.79374601242636], [-9.639636949358724, 14.219323976963215], [-9.90341133260833, 14.644901941500072], [-10.16718571585794, 15.070479906036924], [-10.430960099107548, 15.49605787057378], [-10.694734482357157, 15.921635835110637], [-10.958508865606765, 16.34721379964749], [-11.222283248856373, 16.772791764184348], [-11.48605763210598, 17.198369728721197], [-11.74983201535559, 17.623947693258053], [-12.0136063986052, 18.04952565779491], [-12.277380781854806, 18.475103622331766], [-12.541155165104415, 18.90068158686862], [-12.804929548354023, 19.32625955140547], [-13.068703931603633, 19.751837515942327], [-13.33247831485324, 20.177415480479183], [-13.596252698102848, 20.60299344501604], [-13.860027081352456, 21.02857140955289], [-14.123801464602066, 21.454149374089745], [-14.387575847851673, 21.8797273386266], [-14.651350231101283, 22.305305303163458], [-14.915124614350889, 22.730883267700307], [-15.178898997600498, 23.156461232237163], [-15.442673380850104, 23.58203919677402], [-15.706447764099716, 24.007617161310876], [-15.970222147349324, 24.433195125847725], [-16.23399653059893, 24.85877309038458], [-16.49777091384854, 25.284351054921437], [-16.761545297098152, 25.709929019458293], [-17.025319680347756, 26.13550698399515], [-17.289094063597368, 26.561084948532006], [-17.552868446846972, 26.986662913068855], [-17.816642830096583, 27.41224087760571], [-18.080417213346188, 27.837818842142568], [-18.3441915965958, 28.263396806679424], [-18.60796597984541, 28.68897477121628], [-18.871740363095014, 29.114552735753136], [-19.13551474634462, 29.540130700289986], [-19.39928912959423, 29.965708664826835], [-19.66306351284384, 30.39128662936369], [-19.926837896093453, 30.816864593900547], [-20.190612279343057, 31.242442558437403], [-20.45438666259266, 31.66802052297426], [-20.718161045842272, 32.09359848751111], [-20.981935429091884, 32.519176452047965], [-21.245709812341495, 32.94475441658482], [-21.5094841955911, 33.37033238112168], [-21.773258578840704, 33.795910345658534], [-22.037032962090315, 34.22148831019539], [-22.300807345339926, 34.647066274732246], [-22.56458172858953, 35.072644239269096], [-22.828356111839142, 35.49822220380595], [-23.092130495088746, 35.92380016834281], [-23.355904878338357, 36.349378132879664], [-23.61967926158796, 36.77495609741651], [-23.883453644837573, 37.20053406195337], [-24.147228028087184, 37.626112026490226], [-24.41100241133679, 38.05168999102708], [-24.6747767945864, 38.47726795556394], [-24.938551177836004, 38.90284592010079], [-25.202325561085615, 39.328423884637644], [-25.466099944335227, 39.7540018491745], [-25.72987432758483, 40.17957981371136], [-25.993648710834442, 40.605157778248206], [-26.257423094084047, 41.03073574278506], [-26.521197477333658, 41.45631370732192], [-26.78497186058327, 41.881891671858774], [-27.048746243832873, 42.30746963639562], [-27.312607635951917, 42.73299365892903], [-27.57648327523024, 43.1585088480275], [-27.84035891450856, 43.584024037125964], [-28.104234553786878, 44.009539226224426], [-28.3681101930652, 44.43505441532289], [-28.63198583234352, 44.86056960442135], [-28.895861471621842, 45.28608479351982], [-29.159737110900164, 45.71159998261828], [-29.42361275017848, 46.137115171716744], [-29.687488389456803, 46.562630360815206], [-29.951364028735124, 46.988145549913675], [-30.215239668013446, 47.41366073901214], [-30.479115307291764, 47.8391759281106], [-30.74299094657008, 48.26469111720906], [-31.006866585848403, 48.69020630630753], [-31.270742225126725, 49.11572149540599], [-31.534617864405043, 49.541236684504455], [-31.798493503683364, 49.96675187360292], [-32.062369142961685, 50.39226706270138], [-32.32624478224001, 50.81778225179985], [-32.59012042151832, 51.2432974408983], [-32.853996060796646, 51.66881262999677], [-33.11787170007497, 52.094327819095234], [-33.38174733935328, 52.519843008193696], [-33.64562297863161, 52.945358197292165], [-33.909498617909925, 53.37087338639063], [-34.17337425718825, 53.7963885754891], [-34.43724989646657, 54.22190376458755], [-34.70108466265452, 54.64744429212979], [-34.96456049598992, 55.07320712923184], [-35.227609844307224, 55.4992335938544], [-35.49023244382739, 55.925523258789205], [-35.75242803119928, 56.35207569656407], [-36.014196343499975, 56.778890479443255], [-36.275537118235015, 57.205967179428], [-36.536470807563745, 57.63329272943714], [-36.797302960137706, 58.060680278789796], [-37.05813511271168, 58.48806782814246], [-37.31896726528564, 58.91545537749512], [-37.57979941785961, 59.342842926847794], [-37.84063157043357, 59.77023047620045], [-38.10146372300754, 60.19761802555311], [-38.36229587558151, 60.625005574905785], [-38.62312802815547, 61.052393124258444], [-38.88396018072943, 61.4797806736111], [-39.1447923333034, 61.90716822296377], [-39.40562448587737, 62.33455577231643], [-39.666456638451336, 62.7619433216691], [-39.9272887910253, 63.18933087102176], [-40.18812094359926, 63.61671842037442], [-40.448953096173234, 64.04410596972708], [-40.709785248747195, 64.47149351907974], [-40.97061740132116, 64.8988810684324], [-41.23144955389513, 65.32626861778508], [-41.49228170646909, 65.75365616713773], [-41.75311385904306, 66.18104371649041], [-42.01394601161702, 66.60843126584307], [-42.27477816419099, 67.03581881519573], [-42.53561031676496, 67.4632063645484], [-42.79644246933892, 67.89059391390106], [-43.05727462191289, 68.31798146325372], [-43.31810677448685, 68.74536901260637], [-43.57893892706082, 69.17275656195903], [-43.83977107963479, 69.6001441113117], [-44.10060323220875, 70.02753166066437], [-44.36143538478271, 70.45491921001702], [-44.622267537356684, 70.8823067593697], [-44.883099689930646, 71.30969430872236], [-45.143931842504614, 71.73708185807503], [-45.40476399507858, 72.16446940742769], [-45.6656142227488, 72.5918459232467], [-45.92678466099825, 73.01902681045206], [-46.18838274123038, 73.44594595107218], [-46.450408201121434, 73.87260291700365], [-46.712860777919076, 74.29899728040593], [-46.97574020844268, 74.72512861370183], [-47.2390462290836, 75.15099648957792], [-47.50277857580536, 75.57660048098495], [-47.76693698414403, 76.00194016113828], [-48.03152118920843, 76.42701510351834], [-48.29653092568037, 76.851824881871], [-48.56196592781498, 77.27636907020805], [-48.82782592944092, 77.7006472428076], [-49.09411066396069, 78.12465897421451], [-49.36081986435088, 78.54840383924086], [-49.62795326316242, 78.97188141296627], [-49.89551059252088, 79.39509127073843], [-50.16349158412671, 79.81803298817348], [-50.431895969255535, 80.24070614115647], [-50.70051394418359, 80.6632436101501], [-50.96913192377898, 81.08578107617664], [-51.23774990337438, 81.50831854220318], [-51.506367882969776, 81.93085600822972], [-51.77498586256517, 82.35339347425626], [-52.04360384216057, 82.77593094028282], [-52.31222182175596, 83.19846840630936], [-52.58083980135135, 83.62100587233589], [-52.84945778094676, 84.04354333836244], [-53.11807576054215, 84.46608080438898], [-53.38669374013754, 84.88861827041552], [-53.65531171973294, 85.31115573644206], [-53.92392969932833, 85.7336932024686], [-54.19254767892373, 86.15623066849516], [-54.46116565851912, 86.5787681345217], [-54.72978363811452, 87.00130560054824], [-54.998401617709916, 87.42384306657478], [-55.26701959730531, 87.84638053260132], [-55.53563757690071, 88.26891799862787], [-55.8042555564961, 88.69145546465441], [-56.07287353609149, 89.11399293068095], [-56.3414915156869, 89.53653039670749], [-56.61010949528229, 89.95906786273403], [-56.87872747487768, 90.38160532876057], [-57.14734545447308, 90.80414279478711], [-57.41596343406847, 91.22668026081365], [-57.68458141366387, 91.6492177268402], [-57.95319939325927, 92.07175519286675], [-58.22181737285466, 92.49429265889329], [-58.490435352450056, 92.91683012491983], [-58.75905333204545, 93.33936759094637], [-59.027587620917934, 93.76195823666059], [-59.29572774810521, 94.18479908189784], [-59.56344431397732, 94.60790822687889], [-59.83073705007506, 95.03128524732091], [-60.09760568836419, 95.45492971867237], [-60.36404996123578, 95.87884121611363], [-60.630069601506435, 96.30301931455725], [-60.89585586955154, 96.7273437056382]], 'right': [[4.897452881610404, 0.25587608270306017], [4.633678498360795, 0.6814540472399149], [4.369904115111187, 1.1070320117767696], [4.106129731861579, 1.5326099763136245], [3.8423553486119704, 1.958187940850479], [3.578580965362362, 2.383765905387334], [3.3148065821127535, 2.809343869924189], [3.0510321988631453, 3.2349218344610433], [2.787257815613537, 3.660499798997898], [2.523483432363929, 4.086077763534753], [2.2597090491143206, 4.511655728071608], [1.995934665864712, 4.937233692608462], [1.7321602826151032, 5.3628116571453175], [1.468385899365495, 5.788389621682172], [1.2046115161158868, 6.2139675862190265], [0.9408371328662786, 6.639545550755881], [0.6770627496166703, 7.065123515292735], [0.4132883663670617, 7.490701479829591], [0.1495139831174539, 7.916279444366445], [-0.11426040013215477, 8.341857408903298], [-0.37803478338176255, 8.767435373440154], [-0.6418091666313712, 9.19301333797701], [-0.9055835498809799, 9.618591302513863], [-1.1693579331305877, 10.04416926705072], [-1.4331323163801972, 10.469747231587576], [-1.696906699629805, 10.895325196124428], [-1.9606810828794137, 11.320903160661285], [-2.2244554661290215, 11.746481125198137], [-2.48822984937863, 12.172059089734994], [-2.752004232628239, 12.597637054271846], [-3.0157786158778466, 13.023215018808703], [-3.2795529991274552, 13.448792983345555], [-3.543327382377063, 13.874370947882412], [-3.8071017656266726, 14.299948912419268], [-4.07087614887628, 14.72552687695612], [-4.33465053212589, 15.151104841492977], [-4.598424915375496, 15.57668280602983], [-4.8621992986251055, 16.002260770566686], [-5.125973681874713, 16.42783873510354], [-5.389748065124323, 16.853416699640395], [-5.653522448373929, 17.27899466417725], [-5.917296831623538, 17.704572628714104], [-6.181071214873146, 18.13015059325096], [-6.444845598122756, 18.555728557787816], [-6.7086199813723635, 18.98130652232467], [-6.972394364621971, 19.406884486861525], [-7.236168747871579, 19.832462451398374], [-7.499943131121189, 20.25804041593523], [-7.763717514370798, 20.683618380472087], [-8.027491897620404, 21.109196345008943], [-8.291266280870014, 21.5347743095458], [-8.555040664119621, 21.96035227408265], [-8.818815047369231, 22.385930238619505], [-9.082589430618839, 22.81150820315636], [-9.346363813868447, 23.237086167693217], [-9.610138197118054, 23.662664132230066], [-9.873912580367664, 24.088242096766923], [-10.137686963617272, 24.51382006130378], [-10.401461346866881, 24.939398025840635], [-10.665235730116487, 25.364975990377484], [-10.929010113366097, 25.79055395491434], [-11.192784496615703, 26.216131919451197], [-11.456558879865314, 26.641709883988053], [-11.720333263114922, 27.067287848524902], [-11.984107646364528, 27.49286581306176], [-12.24788202961414, 27.918443777598615], [-12.51165641286375, 28.34402174213547], [-12.775430796113355, 28.769599706672327], [-13.039205179362966, 29.195177671209184], [-13.30297956261257, 29.620755635746033], [-13.566753945862182, 30.04633360028289], [-13.830528329111786, 30.471911564819745], [-14.094302712361397, 30.8974895293566], [-14.358077095611009, 31.323067493893458], [-14.621851478860613, 31.748645458430314], [-14.885625862110217, 32.17422342296717], [-15.149400245359828, 32.599801387504016], [-15.41317462860944, 33.02537935204087], [-15.676949011859051, 33.45095731657773], [-15.940723395108655, 33.876535281114585], [-16.20449777835826, 34.30211324565144], [-16.46827216160787, 34.72769121018829], [-16.732046544857482, 35.153269174725146], [-16.995820928107094, 35.578847139262], [-17.259595311356698, 36.00442510379886], [-17.523369694606302, 36.430003068335715], [-17.787144077855913, 36.85558103287257], [-18.050918461105525, 37.28115899740943], [-18.31469284435513, 37.70673696194628], [-18.57846722760474, 38.13231492648313], [-18.842241610854344, 38.55789289101999], [-19.106015994103956, 38.983470855556845], [-19.36979037735356, 39.409048820093695], [-19.63356476060317, 39.83462678463055], [-19.897339143852783, 40.26020474916741], [-20.161113527102387, 40.68578271370426], [-20.424887910352, 41.11136067824112], [-20.688662293601602, 41.53693864277797], [-20.952436676851214, 41.962516607314825], [-21.216211060100825, 42.38809457185168], [-21.47998544335043, 42.81367253638854], [-21.74375982660004, 43.23925050092539], [-22.007534209849645, 43.66482846546224], [-22.271308593099256, 44.0904064299991], [-22.535082976348868, 44.515984394535955], [-22.798857359598472, 44.941562359072805], [-23.063345637176944, 45.36809754032857], [-23.327221276455262, 45.79361272942704], [-23.591096915733587, 46.2191279185255], [-23.854972555011905, 46.64464310762396], [-24.118848194290223, 47.070158296722425], [-24.382723833568548, 47.49567348582089], [-24.646599472846866, 47.92118867491936], [-24.91047511212519, 48.34670386401782], [-25.17435075140351, 48.77221905311628], [-25.438226390681827, 49.19773424221474], [-25.70210202996015, 49.62324943131321], [-25.96597766923847, 50.048764620411674], [-26.229853308516788, 50.474279809510136], [-26.493728947795105, 50.8997949986086], [-26.75760458707343, 51.32531018770707], [-27.02148022635175, 51.75082537680553], [-27.285355865630066, 52.17634056590399], [-27.54923150490839, 52.601855755002454], [-27.81310714418671, 53.027370944100916], [-28.076982783465034, 53.452886133199385], [-28.340858422743345, 53.87840132229784], [-28.60473406202167, 54.30391651139631], [-28.868609701299995, 54.72943170049477], [-29.132485340578306, 55.15494688959323], [-29.39636097985663, 55.5804620786917], [-29.66023661913495, 56.005977267790165], [-29.924112258413274, 56.431492456888634], [-30.187987897691592, 56.85700764598709], [-30.45066734619644, 57.28068424244827], [-30.711508421198484, 57.7021894511793], [-30.971927276032613, 58.12395565115563], [-31.231923649557576, 58.54598241944109], [-31.49149728105575, 58.968269332838204], [-31.750647910233436, 59.3908159678886], [-32.009375277221125, 59.8136219008735], [-32.26851112306563, 60.238003871099245], [-32.52934327563959, 60.665391420451904], [-32.790175428213566, 61.09277896980457], [-33.05100758078753, 61.52016651915723], [-33.311839733361495, 61.9475540685099], [-33.572671885935456, 62.37494161786256], [-33.833504038509425, 62.80232916721522], [-34.09433619108339, 63.22971671656789], [-34.355168343657354, 63.65710426592055], [-34.616000496231315, 64.08449181527321], [-34.876832648805284, 64.51187936462588], [-35.13766480137925, 64.93926691397854], [-35.39849695395322, 65.36665446333122], [-35.65932910652718, 65.79404201268387], [-35.92016125910114, 66.22142956203653], [-36.18099341167512, 66.64881711138919], [-36.44182556424908, 67.07620466074185], [-36.70265771682304, 67.50359221009451], [-36.963489869397016, 67.93097975944718], [-37.22432202197098, 68.35836730879984], [-37.485154174544945, 68.78575485815252], [-37.745986327118906, 69.21314240750517], [-38.006818479692875, 69.64052995685783], [-38.26765063226684, 70.0679175062105], [-38.528482784840804, 70.49530505556316], [-38.78931493741477, 70.92269260491582], [-39.050147089988734, 71.35008015426848], [-39.3109792425627, 71.77746770362114], [-39.57181139513667, 72.20485525297381], [-39.83264354771063, 72.63224280232647], [-40.09347570028459, 73.05963035167913], [-40.35430785285857, 73.4870179010318], [-40.61514000543253, 73.91440545038446], [-40.8759721580065, 74.34179299973714], [-41.136804310580466, 74.7691805490898], [-41.398412803435825, 75.19779911680435], [-41.66219494606777, 75.62925181288176], [-41.926409007102215, 76.06044014490809], [-42.19105472159218, 76.49136368049886], [-42.4561318241578, 76.92202198753517], [-42.72164004898664, 77.35241463416403], [-42.98757912983397, 77.78254118879889], [-43.25394880002295, 78.21240122011999], [-43.520748792445005, 78.64199429707485], [-43.787978839560054, 79.07131998887871], [-44.055638673396714, 79.50037786501488], [-44.32372802555267, 79.9291674952353], [-44.59224662719487, 80.35768844956087], [-44.86119420905983, 80.78594029828184], [-45.130570501453924, 81.21392261195845], [-45.40037523425357, 81.64163496142112], [-45.67060813690562, 82.069076917771], [-45.94126893842751, 82.49624805238041], [-46.21235736740763, 82.92314793689322], [-46.48098795910815, 83.34570525223805], [-46.74960593870354, 83.76824271826459], [-47.018223918298936, 84.19078018429113], [-47.286841897894334, 84.61331765031767], [-47.555459877489724, 85.03585511634421], [-47.82407785708513, 85.45839258237076], [-48.09269583668052, 85.8809300483973], [-48.36131381627591, 86.30346751442383], [-48.629931795871315, 86.72600498045038], [-48.898549775466705, 87.14854244647692], [-49.167167755062096, 87.57107991250346], [-49.4357857346575, 87.99361737853], [-49.70440371425289, 88.41615484455654], [-49.97302169384829, 88.8386923105831], [-50.24163967344368, 89.26122977660964], [-50.510257653039076, 89.68376724263618], [-50.778875632634474, 90.10630470866272], [-51.047493612229864, 90.52884217468926], [-51.31611159182527, 90.95137964071581], [-51.58472957142066, 91.37391710674235], [-51.85334755101605, 91.7964545727689], [-52.121965530611455, 92.21899203879543], [-52.390583510206845, 92.64152950482197], [-52.659201489802236, 93.06406697084851], [-52.92781946939764, 93.48660443687506], [-53.19643744899303, 93.9091419029016], [-53.46505542858843, 94.33167936892815], [-53.733673408183826, 94.75421683495469], [-54.002291387779216, 95.17675430098123], [-54.270909367374614, 95.59929176700777], [-54.539527346970004, 96.02182923303431], [-54.80637309653871, 96.44176194946678], [-55.07183182245411, 96.86037438625166], [-55.3368712226675, 97.2792524397829], [-55.60149103140426, 97.6983956900205], [-55.8656909833105, 98.11780371665846], [-56.12947081345337, 98.53747609912529], [-56.39283025732132, 98.95741241658448], [-56.65848196198577, 99.38152199043387]], 'next_id': ['1.0.-1.-1'], 'pre_id': []}, {'id': '0.0.1.-1', 'mid': [[-63.01454282333442, 95.40025456324037], [-62.74868927359899, 94.97582276354363], [-62.481339535126985, 94.54952377460779], [-62.21356304089103, 94.12349271967933], [-61.94536005941045, 93.6977300259711], [-61.67673085963223, 93.2722361204269], [-61.40767571093076, 92.84701142972092], [-61.138194883107545, 92.42205638025749], [-60.868816324583165, 91.9981367699024], [-60.600198344987774, 91.57559930387586], [-60.331580365392384, 91.15306183784932], [-60.06296238579699, 90.73052437182278], [-59.79434440620159, 90.30798690579621], [-59.5257264266062, 89.88544943976967], [-59.25710844701081, 89.46291197374313], [-58.9884904674154, 89.04037450771659], [-58.71987248782001, 88.61783704169005], [-58.45125450822461, 88.19529957566351], [-58.18263652862922, 87.77276210963697], [-57.91401854903383, 87.35022464361043], [-57.64540056943842, 86.92768717758389], [-57.37678258984303, 86.50514971155735], [-57.10816461024764, 86.08261224553081], [-56.83954663065224, 85.66007477950427], [-56.570928651056846, 85.23753731347773], [-56.30231067146144, 84.81499984745116], [-56.03369269186605, 84.39246238142462], [-55.76507471227066, 83.96992491539808], [-55.496456732675256, 83.54738744937154], [-55.227838753079865, 83.124849983345], [-54.959220773484475, 82.70231251731846], [-54.69060279388907, 82.27977505129192], [-54.42198481429368, 81.85723758526538], [-54.15336683469829, 81.43470011923884], [-53.884748855102885, 81.0121626532123], [-53.616130875507494, 80.58962518718576], [-53.347512895912104, 80.16708772115922], [-53.0788949163167, 79.74455025513268], [-52.81027693672131, 79.32201278910614], [-52.54166527017949, 78.8994852432881], [-52.27460290697631, 78.47892545607002], [-52.007961820328504, 78.05809844722214], [-51.74174227761684, 77.63700463873884], [-51.47594454579935, 77.21564445288206], [-51.21056889141112, 76.79401831218084], [-50.945615580563945, 76.37212663943097], [-50.68108487894613, 75.9499698576944], [-50.4169770518222, 75.52754839029905], [-50.15329236403262, 75.10486266083817], [-49.89003107999355, 74.68191309317], [-49.627193463696564, 74.25870011141743], [-49.36477977870841, 73.83522413996744], [-49.102790288170695, 73.41148560347072], [-48.84122525479971, 72.9874849268413], [-48.58008494088605, 72.56322253525605], [-48.31936960829446, 72.13869885415423], [-48.059079518463484, 71.7139143092372], [-47.79921493240529, 71.28886932646787], [-47.538743837327644, 70.86211383659663], [-47.27791168475366, 70.43472628724396], [-47.01707953217971, 70.0073387378913], [-46.75624737960574, 69.57995118853864], [-46.49541522703177, 69.15256363918597], [-46.2345830744578, 68.72517608983331], [-45.97375092188385, 68.29778854048065], [-45.71291876930988, 67.87040099112798], [-45.45208661673591, 67.44301344177532], [-45.19125446416194, 67.01562589242266], [-44.930422311587975, 66.58823834307], [-44.66959015901401, 66.16085079371733], [-44.40875800644005, 65.73346324436467], [-44.147925853866084, 65.30607569501201], [-43.887093701292116, 64.87868814565934], [-43.62626154871815, 64.45130059630668], [-43.365429396144194, 64.02391304695402], [-43.10459724357021, 63.59652549760135], [-42.84376509099626, 63.16913794824869], [-42.58293293842229, 62.74175039889603], [-42.32210078584832, 62.314362849543365], [-42.06126863327435, 61.886975300190706], [-41.8004364807004, 61.45958775083805], [-41.53960432812643, 61.032200201485374], [-41.27877217555245, 60.60481265213271], [-41.01794002297849, 60.17742510278005], [-40.757107870404525, 59.75003755342739], [-40.49627571783056, 59.32265000407472], [-40.2354435652566, 58.89526245472206], [-39.974611412682634, 58.4678749053694], [-39.713779260108666, 58.040487356016726], [-39.4529471075347, 57.61309980666407], [-39.19211495496074, 57.18571225731141], [-38.93128280238676, 56.75832470795874], [-38.67045064981281, 56.33093715860608], [-38.40861803874196, 55.90213981870525], [-38.14597056013324, 55.47292773522058], [-37.88289340627104, 55.04397887842699], [-37.61938684096229, 54.61529367846326], [-37.35545112844453, 54.186872565203785], [-37.091086533385635, 53.758715968258095], [-36.826293320883565, 53.33082431697055], [-36.56188089585406, 52.90435182388778], [-36.298005256575735, 52.47883663478932], [-36.03412961729741, 52.053321445690855], [-35.7702539780191, 51.62780625659239], [-35.506378338740774, 51.20229106749393], [-35.24250269946246, 50.77677587839547], [-34.97862706018414, 50.35126068929701], [-34.71475142090581, 49.92574550019853], [-34.45087578162749, 49.50023031110007], [-34.18700014234918, 49.07471512200161], [-33.92312450307085, 48.649199932903144], [-33.65924886379253, 48.22368474380468], [-33.39537322451421, 47.79816955470622], [-33.13149758523589, 47.37265436560776], [-32.867621945957566, 46.947139176509296], [-32.603746306679255, 46.521623987410834], [-32.33987066740093, 46.09610879831236], [-32.075995028122605, 45.670593609213896], [-31.81211938884429, 45.24507842011543], [-31.54824374956597, 44.81956323101697], [-31.284368110287648, 44.39404804191851], [-31.020492471009327, 43.96853285282005], [-30.75661683173101, 43.543017663721585], [-30.492741192452687, 43.11750247462312], [-30.228865553174366, 42.69198728552466], [-29.964989913896044, 42.266472096426185], [-29.701114274617726, 41.84095690732774], [-29.437238635339405, 41.41544171822926], [-29.173690685950074, 40.99042327505703], [-28.90991630270047, 40.564845310520184], [-28.64614191945085, 40.13926734598332], [-28.382367536201247, 39.71368938144647], [-28.118593152951636, 39.28811141690961], [-27.85481876970203, 38.86253345237276], [-27.591044386452428, 38.43695548783591], [-27.32727000320281, 38.011377523299046], [-27.063495619953205, 37.5857995587622], [-26.7997212367036, 37.16022159422535], [-26.53594685345399, 36.734643629688485], [-26.272172470204385, 36.309065665151635], [-26.008398086954767, 35.88348770061477], [-25.744623703705162, 35.45790973607792], [-25.480849320455558, 35.032331771541074], [-25.217074937205947, 34.60675380700421], [-24.953300553956343, 34.18117584246736], [-24.689526170706724, 33.7555978779305], [-24.42575178745712, 33.33001991339365], [-24.161977404207516, 32.9044419488568], [-23.898203020957904, 32.478863984319936], [-23.6344286377083, 32.05328601978309], [-23.370654254458696, 31.62770805524623], [-23.106879871209077, 31.202130090709375], [-22.843105487959473, 30.77655212617252], [-22.579331104709862, 30.350974161635662], [-22.315556721460258, 29.925396197098813], [-22.051782338210653, 29.499818232561957], [-21.788007954961035, 29.0742402680251], [-21.52423357171143, 28.648662303488244], [-21.260459188461827, 28.223084338951402], [-20.996684805212215, 27.79750637441454], [-20.73291042196261, 27.37192840987769], [-20.469136038712993, 26.946350445340826], [-20.20536165546339, 26.520772480803977], [-19.941587272213784, 26.09519451626712], [-19.677812888964173, 25.669616551730265], [-19.41403850571457, 25.244038587193415], [-19.15026412246495, 24.818460622656552], [-18.886489739215346, 24.392882658119703], [-18.62271535596574, 23.967304693582847], [-18.35894097271613, 23.54172672904599], [-18.095166589466526, 23.116148764509134], [-17.831392206216915, 22.690570799972285], [-17.567617822967303, 22.26499283543542], [-17.3038434397177, 21.839414870898572], [-17.040069056468088, 21.413836906361716], [-16.77629467321848, 20.98825894182486], [-16.512520289968876, 20.56268097728801], [-16.24874590671926, 20.137103012751147], [-15.984971523469657, 19.7115250482143], [-15.721197140220045, 19.285947083677435], [-15.457422756970441, 18.860369119140586], [-15.193648373720833, 18.434791154603737], [-14.929873990471219, 18.009213190066873], [-14.666099607221614, 17.583635225530024], [-14.40232522397201, 17.158057260993175], [-14.138550840722399, 16.73247929645631], [-13.874776457472791, 16.306901331919462], [-13.611002074223178, 15.8813233673826], [-13.347227690973572, 15.455745402845752], [-13.083453307723968, 15.030167438308903], [-12.819678924474355, 14.604589473772041], [-12.555904541224749, 14.179011509235192], [-12.292130157975135, 13.753433544698328], [-12.02835577472553, 13.32785558016148], [-11.764581391475925, 12.902277615624627], [-11.500807008226312, 12.476699651087767], [-11.237032624976706, 12.051121686550916], [-10.9732582417271, 11.625543722014065], [-10.709483858477487, 11.199965757477203], [-10.445709475227883, 10.774387792940354], [-10.18193509197827, 10.34880982840349], [-9.918160708728664, 9.923231863866642], [-9.654386325479058, 9.49765389932979], [-9.390611942229445, 9.07207593479293], [-9.12683755897984, 8.646497970256078], [-8.863063175730234, 8.220920005719229], [-8.599288792480621, 7.795342041182366], [-8.335514409231017, 7.369764076645517], [-8.071740025981402, 6.944186112108655], [-7.807965642731798, 6.518608147571804], [-7.544191259482192, 6.093030183034953], [-7.280416876232579, 5.6674522184980916], [-7.0166424929829745, 5.241874253961242], [-6.75286810973336, 4.816296289424379], [-6.489093726483755, 4.39071832488753], [-6.225319343234149, 3.9651403603506785], [-5.961544959984536, 3.539562395813817], [-5.697770576734931, 3.113984431276967], [-5.433996193485326, 2.688406466740117], [-5.170221810235713, 2.2628285022032544], [-4.906447426986107, 1.8372505376664043], [-4.642673043736494, 1.4116725731295419], [-4.378898660486889, 0.9860946085926918], [-4.115124277237284, 0.5605166440558418], [-3.8513498939876705, 0.13493867951897975], [-3.5875755107380645, -0.2906392850178703], [-3.3238011274884522, -0.7162172495547328], [-3.0600267442388462, -1.1417952140915828], [-2.796252360989241, -1.5673731786284328], [-2.532477977739628, -1.9929511431652949], [-2.268703594490023, -2.418529107702145], [-2.004929211240417, -2.844107072238995], [-1.7411548279908042, -3.2696850367758574], [-1.4773804447411987, -3.6952630013127075]], 'left': [[-60.89585586955154, 96.7273437056382], [-60.630069601506435, 96.30301931455725], [-60.36404996123578, 95.87884121611363], [-60.09760568836419, 95.45492971867237], [-59.83073705007505, 95.0312852473209], [-59.56344431397732, 94.60790822687889], [-59.29572774810521, 94.18479908189784], [-59.02758762091793, 93.76195823666058], [-58.75905333204545, 93.33936759094637], [-58.490435352450056, 92.91683012491983], [-58.22181737285466, 92.49429265889329], [-57.95319939325927, 92.07175519286675], [-57.684581413663864, 91.64921772684019], [-57.41596343406847, 91.22668026081365], [-57.14734545447308, 90.80414279478711], [-56.87872747487768, 90.38160532876057], [-56.61010949528229, 89.95906786273403], [-56.34149151568688, 89.53653039670748], [-56.07287353609149, 89.11399293068095], [-55.8042555564961, 88.69145546465441], [-55.535637576900704, 88.26891799862786], [-55.26701959730531, 87.84638053260132], [-54.998401617709916, 87.42384306657478], [-54.72978363811452, 87.00130560054824], [-54.46116565851912, 86.5787681345217], [-54.192547678923724, 86.15623066849514], [-53.92392969932833, 85.7336932024686], [-53.65531171973294, 85.31115573644206], [-53.38669374013754, 84.88861827041552], [-53.11807576054215, 84.46608080438898], [-52.84945778094675, 84.04354333836243], [-52.58083980135135, 83.62100587233589], [-52.31222182175596, 83.19846840630936], [-52.043603842160564, 82.7759309402828], [-51.77498586256517, 82.35339347425626], [-51.506367882969776, 81.93085600822972], [-51.23774990337438, 81.50831854220318], [-50.96913192377898, 81.08578107617664], [-50.70051394418359, 80.6632436101501], [-50.431895969255535, 80.24070614115647], [-50.16349158412671, 79.81803298817348], [-49.89551059252087, 79.39509127073842], [-49.62795326316242, 78.97188141296627], [-49.36081986435087, 78.54840383924085], [-49.09411066396069, 78.12465897421451], [-48.82782592944092, 77.7006472428076], [-48.561965927814974, 77.27636907020803], [-48.29653092568037, 76.851824881871], [-48.03152118920843, 76.42701510351834], [-47.76693698414403, 76.00194016113828], [-47.50277857580536, 75.57660048098495], [-47.2390462290836, 75.15099648957792], [-46.97574020844268, 74.72512861370183], [-46.712860777919076, 74.29899728040593], [-46.45040820112143, 73.87260291700365], [-46.18838274123038, 73.44594595107218], [-45.926784660998244, 73.01902681045205], [-45.6656142227488, 72.5918459232467], [-45.40476399507858, 72.16446940742769], [-45.14393184250461, 71.73708185807502], [-44.883099689930646, 71.30969430872236], [-44.622267537356684, 70.8823067593697], [-44.36143538478271, 70.45491921001702], [-44.10060323220875, 70.02753166066437], [-43.83977107963479, 69.6001441113117], [-43.57893892706082, 69.17275656195903], [-43.31810677448685, 68.74536901260637], [-43.05727462191289, 68.31798146325372], [-42.79644246933892, 67.89059391390106], [-42.53561031676495, 67.46320636454838], [-42.27477816419099, 67.03581881519573], [-42.01394601161702, 66.60843126584307], [-41.753113859043054, 66.1810437164904], [-41.49228170646909, 65.75365616713773], [-41.23144955389513, 65.32626861778508], [-40.97061740132116, 64.8988810684324], [-40.709785248747195, 64.47149351907974], [-40.448953096173234, 64.04410596972708], [-40.18812094359926, 63.61671842037442], [-39.9272887910253, 63.18933087102176], [-39.666456638451336, 62.7619433216691], [-39.40562448587737, 62.33455577231643], [-39.14479233330339, 61.90716822296376], [-38.88396018072943, 61.4797806736111], [-38.62312802815547, 61.052393124258444], [-38.3622958755815, 60.62500557490577], [-38.10146372300754, 60.19761802555311], [-37.84063157043357, 59.77023047620045], [-37.579799417859604, 59.34284292684778], [-37.31896726528564, 58.91545537749512], [-37.05813511271168, 58.48806782814246], [-36.797302960137706, 58.060680278789796], [-36.536470807563745, 57.63329272943714], [-36.275537118235015, 57.205967179428], [-36.014196343499975, 56.778890479443255], [-35.75242803119927, 56.35207569656406], [-35.49023244382739, 55.925523258789205], [-35.227609844307224, 55.4992335938544], [-34.964560495989915, 55.073207129231825], [-34.70108466265452, 54.64744429212979], [-34.43724989646657, 54.22190376458755], [-34.17337425718824, 53.79638857548908], [-33.909498617909925, 53.37087338639063], [-33.64562297863161, 52.945358197292165], [-33.38174733935328, 52.519843008193696], [-33.11787170007497, 52.094327819095234], [-32.853996060796646, 51.66881262999677], [-32.59012042151832, 51.2432974408983], [-32.32624478224, 50.81778225179984], [-32.062369142961685, 50.39226706270138], [-31.798493503683364, 49.96675187360292], [-31.534617864405043, 49.541236684504455], [-31.27074222512672, 49.115721495405985], [-31.0068665858484, 48.69020630630752], [-30.74299094657008, 48.26469111720906], [-30.479115307291764, 47.8391759281106], [-30.21523966801344, 47.41366073901213], [-29.95136402873512, 46.98814554991367], [-29.687488389456803, 46.562630360815206], [-29.42361275017848, 46.137115171716744], [-29.15973711090016, 45.711599982618274], [-28.89586147162184, 45.28608479351981], [-28.63198583234352, 44.86056960442135], [-28.3681101930652, 44.43505441532289], [-28.104234553786878, 44.009539226224426], [-27.840358914508556, 43.58402403712596], [-27.57648327523024, 43.1585088480275], [-27.312607635951917, 42.73299365892903], [-27.048746243832873, 42.30746963639562], [-26.78497186058327, 41.881891671858774], [-26.52119747733365, 41.45631370732191], [-26.257423094084047, 41.03073574278506], [-25.993648710834435, 40.6051577782482], [-25.72987432758483, 40.17957981371135], [-25.466099944335227, 39.7540018491745], [-25.20232556108561, 39.32842388463764], [-24.938551177836004, 38.90284592010079], [-24.6747767945864, 38.47726795556394], [-24.41100241133679, 38.051689991027075], [-24.147228028087184, 37.626112026490226], [-23.883453644837566, 37.20053406195336], [-23.61967926158796, 36.77495609741651], [-23.355904878338357, 36.349378132879664], [-23.092130495088746, 35.9238001683428], [-22.828356111839142, 35.49822220380595], [-22.564581728589523, 35.07264423926909], [-22.30080734533992, 34.64706627473224], [-22.037032962090315, 34.22148831019539], [-21.773258578840704, 33.79591034565853], [-21.5094841955911, 33.37033238112168], [-21.245709812341495, 32.94475441658482], [-20.981935429091877, 32.519176452047965], [-20.718161045842272, 32.09359848751111], [-20.45438666259266, 31.668020522974253], [-20.190612279343057, 31.242442558437403], [-19.926837896093453, 30.816864593900547], [-19.663063512843834, 30.39128662936369], [-19.39928912959423, 29.965708664826835], [-19.135514746344626, 29.540130700289993], [-18.871740363095014, 29.11455273575313], [-18.60796597984541, 28.68897477121628], [-18.344191596595792, 28.263396806679417], [-18.080417213346188, 27.837818842142568], [-17.816642830096583, 27.41224087760571], [-17.552868446846972, 26.986662913068855], [-17.289094063597368, 26.561084948532006], [-17.02531968034775, 26.135506983995143], [-16.761545297098145, 25.709929019458293], [-16.49777091384854, 25.284351054921437], [-16.23399653059893, 24.85877309038458], [-15.970222147349324, 24.433195125847725], [-15.706447764099716, 24.007617161310876], [-15.442673380850104, 23.582039196774012], [-15.178898997600498, 23.156461232237163], [-14.915124614350885, 22.730883267700307], [-14.65135023110128, 22.30530530316345], [-14.387575847851675, 21.8797273386266], [-14.123801464602062, 21.454149374089738], [-13.860027081352456, 21.02857140955289], [-13.596252698102843, 20.602993445016025], [-13.332478314853239, 20.177415480479176], [-13.068703931603633, 19.751837515942327], [-12.80492954835402, 19.326259551405464], [-12.541155165104414, 18.900681586868615], [-12.27738078185481, 18.475103622331766], [-12.013606398605196, 18.049525657794902], [-11.74983201535559, 17.623947693258053], [-11.486057632105977, 17.19836972872119], [-11.222283248856371, 16.77279176418434], [-10.958508865606767, 16.34721379964749], [-10.694734482357154, 15.92163583511063], [-10.430960099107548, 15.49605787057378], [-10.167185715857935, 15.070479906036917], [-9.903411332608329, 14.644901941500068], [-9.639636949358724, 14.219323976963215], [-9.375862566109111, 13.793746012426356], [-9.112088182859505, 13.368168047889505], [-8.8483137996099, 12.942590083352654], [-8.584539416360286, 12.517012118815792], [-8.320765033110682, 12.091434154278943], [-8.056990649861069, 11.66585618974208], [-7.793216266611463, 11.24027822520523], [-7.529441883361857, 10.81470026066838], [-7.265667500112245, 10.389122296131518], [-7.001893116862639, 9.963544331594667], [-6.7381187336130335, 9.537966367057818], [-6.47434435036342, 9.112388402520955], [-6.210569967113815, 8.686810437984105], [-5.946795583864202, 8.261232473447244], [-5.683021200614597, 7.835654508910393], [-5.419246817364991, 7.410076544373542], [-5.155472434115378, 6.98449857983668], [-4.891698050865773, 6.558920615299831], [-4.62792366761616, 6.133342650762968], [-4.364149284366555, 5.707764686226119], [-4.100374901116949, 5.282186721689268], [-3.8366005178673355, 4.856608757152406], [-3.5728261346177304, 4.431030792615556], [-3.309051751368125, 4.005452828078706], [-3.045277368118512, 3.5798748635418436], [-2.7815029848689066, 3.1542968990049935], [-2.5177286016192935, 2.728718934468131], [-2.253954218369688, 2.303140969931281], [-1.9901798351200828, 1.877563005394431], [-1.7264054518704697, 1.451985040857569], [-1.4626310686208641, 1.026407076320719], [-1.1988566853712512, 0.6008291117838565], [-0.9350823021216457, 0.1752511472470064], [-0.6713079188720403, -0.25032681728984363], [-0.4075335356224272, -0.6759047818267059], [-0.14375915237282177, -1.101482746363556], [0.12001523087678367, -1.527060710900406], [0.38378961412639667, -1.9526386754372682], [0.6475639973760021, -2.3782166399741183]], 'right': [[-65.13322977711731, 94.07316542084254], [-64.86730894569155, 93.64862621253002], [-64.59862910901819, 93.22020633310197], [-64.32952039341788, 92.79205572068629], [-64.05998306874585, 92.3641748046213], [-63.790017405287145, 91.93656401397489], [-63.51962367375631, 91.50922377754401], [-63.248802145297155, 91.08215452385438], [-62.97857931712089, 90.65690594885842], [-62.7099613375255, 90.23436848283188], [-62.4413433579301, 89.81183101680534], [-62.17272537833471, 89.3892935507788], [-61.904107398739306, 88.96675608475225], [-61.635489419143916, 88.54421861872571], [-61.366871439548525, 88.12168115269917], [-61.09825345995312, 87.69914368667263], [-60.82963548035773, 87.27660622064609], [-60.561017500762325, 86.85406875461953], [-60.292399521166935, 86.431531288593], [-60.023781541571545, 86.00899382256647], [-59.75516356197615, 85.58645635653991], [-59.48654558238075, 85.16391889051337], [-59.21792760278536, 84.74138142448683], [-58.94930962318996, 84.31884395846029], [-58.680691643594564, 83.89630649243375], [-58.412073663999166, 83.4737690264072], [-58.143455684403776, 83.05123156038066], [-57.874837704808385, 82.62869409435412], [-57.60621972521298, 82.20615662832758], [-57.33760174561759, 81.78361916230104], [-57.06898376602219, 81.36108169627448], [-56.800365786426795, 80.93854423024794], [-56.531747806831405, 80.51600676422142], [-56.26312982723601, 80.09346929819486], [-55.99451184764061, 79.67093183216832], [-55.72589386804522, 79.24839436614178], [-55.45727588844982, 78.82585690011524], [-55.188657908854424, 78.4033194340887], [-54.92003992925903, 77.98078196806216], [-54.651434571103444, 77.55826434541972], [-54.38571422982591, 77.13981792396656], [-54.12041304813613, 76.72110562370585], [-53.85553129207126, 76.30212786451142], [-53.59106922724783, 75.88288506652326], [-53.32702711886155, 75.46337765014718], [-53.06340523168697, 75.04360603605434], [-52.80020383007729, 74.62357064518076], [-52.53742317796403, 74.20327189872711], [-52.27506353885681, 73.78271021815797], [-52.01312517584306, 73.36188602520171], [-51.75160835158777, 72.94079974184991], [-51.49051332833323, 72.51945179035695], [-51.22984036789872, 72.09784259323962], [-50.96958973168035, 71.67597257327668], [-50.70976168065068, 71.25384215350843], [-50.45035647535854, 70.83145175723627], [-50.191374375928724, 70.40880180802235], [-49.932815642061776, 69.98589272968904], [-49.6727236795767, 69.55975826576558], [-49.41189152700272, 69.13237071641291], [-49.15105937442876, 68.70498316706025], [-48.8902272218548, 68.27759561770759], [-48.629395069280825, 67.85020806835492], [-48.368562916706864, 67.42282051900226], [-48.1077307641329, 66.9954329696496], [-47.846898611558935, 66.56804542029693], [-47.586066458984966, 66.14065787094427], [-47.325234306411005, 65.71327032159161], [-47.06440215383704, 65.28588277223895], [-46.80357000126307, 64.85849522288628], [-46.54273784868911, 64.43110767353362], [-46.28190569611514, 64.00372012418096], [-46.02107354354117, 63.576332574828285], [-45.76024139096721, 63.148945025475626], [-45.49940923839325, 62.72155747612297], [-45.23857708581927, 62.294169926770294], [-44.97774493324531, 61.866782377417636], [-44.71691278067135, 61.43939482806498], [-44.456080628097375, 61.01200727871231], [-44.195248475523414, 60.58461972935965], [-43.93441632294945, 60.15723218000699], [-43.673584170375484, 59.72984463065432], [-43.41275201780151, 59.302457081301654], [-43.15191986522755, 58.875069531948995], [-42.89108771265359, 58.447681982596336], [-42.63025556007962, 58.02029443324366], [-42.36942340750566, 57.592906883891004], [-42.10859125493169, 57.165519334538345], [-41.84775910235772, 56.73813178518567], [-41.58692694978376, 56.31074423583301], [-41.3260947972098, 55.883356686480354], [-41.06526264463582, 55.45596913712769], [-40.80443049206186, 55.02858158777503], [-40.541698959248905, 54.5983124579825], [-40.27774477676651, 54.16696499099791], [-40.0133587813428, 53.735882060289924], [-39.748541238097204, 53.30506409813732], [-39.48329241258183, 52.874511536553165], [-39.217612570781355, 52.444224807284364], [-38.95150197911261, 52.014204341811315], [-38.686511895241544, 51.586799883188014], [-38.42263625596322, 51.161284694089545], [-38.1587606166849, 50.73576950499109], [-37.89488497740658, 50.31025431589263], [-37.63100933812826, 49.88473912679416], [-37.36713369884995, 49.4592239376957], [-37.10325805957162, 49.033708748597235], [-36.8393824202933, 48.608193559498766], [-36.57550678101498, 48.1826783704003], [-36.31163114173666, 47.75716318130184], [-36.04775550245834, 47.33164799220338], [-35.78387986318002, 46.90613280310492], [-35.520004223901694, 46.48061761400645], [-35.256128584623376, 46.055102424907986], [-34.99225294534506, 45.629587235809524], [-34.72837730606674, 45.20407204671106], [-34.464501666788415, 44.77855685761259], [-34.2006260275101, 44.35304166851413], [-33.93675038823178, 43.92752647941567], [-33.672874748953454, 43.502011290317206], [-33.408999109675136, 43.07649610121874], [-33.14512347039681, 42.650980912120275], [-32.88124783111849, 42.22546572302181], [-32.617372191840175, 41.79995053392335], [-32.35349655256185, 41.37443534482489], [-32.08962091328353, 40.94892015572642], [-31.825745274005214, 40.523404966627965], [-31.56186963472689, 40.097889777529495], [-31.298635128067275, 39.67337691371844], [-31.03486074481767, 39.24779894918159], [-30.771086361568052, 38.82222098464473], [-30.507311978318448, 38.39664302010788], [-30.243537595068837, 37.97106505557102], [-29.979763211819233, 37.54548709103417], [-29.71598882856963, 37.11990912649732], [-29.45221444532001, 36.694331161960456], [-29.188440062070406, 36.26875319742361], [-28.9246656788208, 35.84317523288676], [-28.66089129557119, 35.417597268349894], [-28.397116912321586, 34.992019303813045], [-28.133342529071967, 34.56644133927618], [-27.869568145822363, 34.14086337473933], [-27.60579376257276, 33.71528541020248], [-27.342019379323148, 33.28970744566562], [-27.078244996073543, 32.86412948112877], [-26.814470612823925, 32.43855151659191], [-26.55069622957432, 32.01297355205506], [-26.286921846324717, 31.587395587518213], [-26.023147463075105, 31.16181762298135], [-25.7593730798255, 30.7362396584445], [-25.495598696575897, 30.310661693907644], [-25.23182431332628, 29.885083729370788], [-24.968049930076674, 29.45950576483393], [-24.704275546827063, 29.033927800297075], [-24.44050116357746, 28.608349835760226], [-24.176726780327854, 28.18277187122337], [-23.912952397078236, 27.757193906686513], [-23.64917801382863, 27.331615942149657], [-23.385403630579027, 26.906037977612815], [-23.121629247329416, 26.48046001307595], [-22.857854864079812, 26.054882048539103], [-22.594080480830193, 25.62930408400224], [-22.33030609758059, 25.20372611946539], [-22.066531714330985, 24.778148154928534], [-21.802757331081374, 24.352570190391678], [-21.53898294783177, 23.92699222585483], [-21.27520856458215, 23.501414261317965], [-21.011434181332547, 23.075836296781116], [-20.747659798082942, 22.65025833224426], [-20.48388541483333, 22.224680367707403], [-20.220111031583727, 21.799102403170547], [-19.956336648334116, 21.373524438633698], [-19.692562265084504, 20.947946474096835], [-19.4287878818349, 20.522368509559985], [-19.16501349858529, 20.09679054502313], [-18.90123911533568, 19.671212580486273], [-18.637464732086077, 19.245634615949424], [-18.373690348836462, 18.82005665141256], [-18.109915965586858, 18.39447868687571], [-17.846141582337246, 17.968900722338848], [-17.582367199087642, 17.543322757802], [-17.318592815838034, 17.11774479326515], [-17.05481843258842, 16.692166828728286], [-16.791044049338815, 16.266588864191437], [-16.52726966608921, 15.841010899654588], [-16.2634952828396, 15.415432935117725], [-15.999720899589992, 14.989854970580875], [-15.735946516340379, 14.564277006044012], [-15.472172133090773, 14.138699041507163], [-15.208397749841168, 13.713121076970314], [-14.944623366591555, 13.28754311243345], [-14.68084898334195, 12.861965147896601], [-14.417074600092336, 12.436387183359738], [-14.15330021684273, 12.010809218822889], [-13.889525833593126, 11.585231254286036], [-13.625751450343513, 11.159653289749176], [-13.361977067093907, 10.734075325212327], [-13.0982026838443, 10.308497360675474], [-12.834428300594688, 9.882919396138615], [-12.570653917345084, 9.457341431601765], [-12.30687953409547, 9.031763467064902], [-12.043105150845864, 8.606185502528053], [-11.779330767596258, 8.1806075379912], [-11.515556384346645, 7.7550295734543395], [-11.251782001097041, 7.329451608917489], [-10.988007617847435, 6.903873644380639], [-10.724233234597822, 6.478295679843776], [-10.460458851348218, 6.052717715306927], [-10.196684468098603, 5.627139750770065], [-9.932910084848999, 5.201561786233214], [-9.669135701599393, 4.7759838216963635], [-9.40536131834978, 4.350405857159502], [-9.141586935100175, 3.9248278926226527], [-8.87781255185056, 3.4992499280857894], [-8.614038168600956, 3.0736719635489402], [-8.35026378535135, 2.6480939990120893], [-8.086489402101737, 2.2225160344752277], [-7.822715018852132, 1.7969380699383777], [-7.558940635602527, 1.3713601054015276], [-7.295166252352914, 0.9457821408646652], [-7.031391869103308, 0.5202041763278151], [-6.767617485853695, 0.09462621179095265], [-6.5038431026040895, -0.3309517527458974], [-6.240068719354484, -0.7565297172827474], [-5.976294336104871, -1.1821076818196095], [-5.712519952855265, -1.6076856463564595], [-5.448745569605653, -2.033263610893322], [-5.184971186356047, -2.458841575430172], [-4.921196803106442, -2.884419539967022], [-4.657422419856829, -3.309997504503884], [-4.393648036607224, -3.735575469040734], [-4.129873653357618, -4.161153433577584], [-3.866099270108005, -4.586731398114447], [-3.6023248868583995, -5.012309362651297]], 'next_id': [], 'pre_id': ['1.0.1.-1']}, {'id': '1.0.-1.-1', 'mid': [[-58.77717181086555, 98.05443747003454], [-59.04324951655741, 98.47922834935953], [-59.309327222249266, 98.90401922868452], [-59.57540492794112, 99.3288101080095], [-59.841482633632964, 99.7536009873345], [-60.10756033932482, 100.17839186665948], [-60.37797942616198, 100.60902232747802], [-60.65103455702126, 101.04037833758073], [-60.92568791484345, 101.47071848790057], [-61.201935720323036, 101.9000368568391], [-61.479774172214455, 102.3283275368578], [-61.75919944738445, 102.75558463455951], [-62.0402077008646, 103.1818022707694], [-62.32279506590429, 103.60697458061593], [-62.60695765402388, 104.03109571361155], [-62.89269155506827, 104.45415983373314], [-63.179992837260606, 104.87616111950246], [-63.46885754725648, 105.29709376406606], [-63.75928171019828, 105.71695197527536], [-64.0512613297699, 106.13572997576622], [-64.34479238825168, 106.55342200303858], [-64.6398708465758, 106.97002230953554], [-64.9281029434027, 107.37574537513248], [-65.21468830792438, 107.78084667802452], [-65.5004598092803, 108.1865225163186], [-65.78541629566737, 108.5927712549367], [-66.0695566185673, 108.99959125649164], [-66.35287963275141, 109.40698088129383], [-66.63538419628517, 109.81493848735781], [-66.91706917053288, 110.22346243040879], [-67.19793342016209, 110.63255106388944], [-67.47797581314836, 111.04220273896642], [-67.75719522077969, 111.45241580453703], [-68.0355905176612, 111.86318860723593], [-68.31316058171956, 112.27451949144168], [-68.5899042942075, 112.68640679928356], [-68.86582053970842, 113.09884887064817], [-69.14093685131287, 113.5118870452203]], 'left': [[-60.89585876464844, 96.72734832763672], [-61.161936470340294, 97.1521392069617], [-61.42801417603215, 97.5769300862867], [-61.694091881724006, 98.00172096561168], [-61.960169587415855, 98.42651184493667], [-62.22624729310771, 98.85130272426166], [-62.492806774819826, 99.27579118271397], [-62.76090176380751, 99.69931146015514], [-63.030565947406814, 100.1218343312614], [-63.30179561496463, 100.54335398200217], [-63.57458703428637, 100.9638646121515], [-63.848936451687294, 101.38336043536779], [-64.12484009204421, 101.80183567927351], [-64.40229415884735, 102.21928458553455], [-64.6812948342527, 102.63570140993951], [-64.96183827913445, 103.05108042247869], [-65.24392063313789, 103.465415907423], [-65.52753801473247, 103.87870216340254], [-65.81268652126525, 104.2909335034851], [-66.09936222901462, 104.70210425525443], [-66.3875611932442, 105.11220876088824], [-66.67727944825722, 105.52124137723607], [-66.96757221716636, 105.92986672138109], [-67.25705623279187, 106.33906539798498], [-67.5457181534797, 106.74884442109605], [-67.83355681577686, 107.15920213909831], [-68.12057105954847, 107.5701368980434], [-68.4067597279825, 107.98164704165714], [-68.6921216675944, 108.39373091134628], [-68.97665572823178, 108.80638684620517], [-69.26036076307894, 109.21961318302243], [-69.54323562866165, 109.63340825628771], [-69.82527918485164, 110.04777039819831], [-70.10649029487125, 110.46269793866603], [-70.38686782529801, 110.87818920532374], [-70.66641064606918, 111.29424252353229], [-70.94511763048632, 111.71085621638714], [-71.22298769962953, 112.12802857520833]], 'right': [[-56.65848485708267, 99.38152661243237], [-56.924562562774526, 99.80631749175735], [-57.19064026846638, 100.23110837108234], [-57.45671797415824, 100.65589925040733], [-57.72279567985008, 101.08069012973232], [-57.988873385541936, 101.5054810090573], [-58.26315207750413, 101.94225347224204], [-58.541167350235014, 102.38144521500631], [-58.82080988228009, 102.81960264453976], [-59.10207582568143, 103.25671973167601], [-59.38496131014254, 103.6927904615641], [-59.66946244308159, 104.12780883375122], [-59.95557530968499, 104.56176886226528], [-60.243295972961214, 104.9946645756973], [-60.53262047379506, 105.42649001728358], [-60.823544831002074, 105.85723924498761], [-61.11606504138332, 106.28690633158192], [-61.410177079780496, 106.7154853647296], [-61.705876899131304, 107.14297044706562], [-62.00316043052518, 107.56935569627802], [-62.30202358325917, 107.9946352451889], [-62.60246224489437, 108.41880324183501], [-62.888633669639034, 108.82162402888386], [-63.17232038305687, 109.22262795806404], [-63.45520146508092, 109.62420061154117], [-63.73727577555788, 110.02634037077507], [-64.01854217758613, 110.4290456149399], [-64.29899953752032, 110.83231472093054], [-64.57864672497595, 111.23614606336932], [-64.85748261283399, 111.6405380146124], [-65.13550607724524, 112.04548894475644], [-65.41271599763506, 112.45099722164514], [-65.68911125670775, 112.85706121087576], [-65.96469074045116, 113.26367927580583], [-66.23945333814109, 113.67084977755962], [-66.51339794234582, 114.07857107503484], [-66.78652344893052, 114.4868415249092], [-67.05888600299622, 114.89574551523229]], 'next_id': [], 'pre_id': ['0.0.-1.-1']}, {'id': '1.0.1.-1', 'mid': [[-73.30503854794618, 110.74417010519636], [-73.02441472126422, 110.32286356212612], [-72.74291699793086, 109.90207824778102], [-72.46057506887647, 109.48185891920579], [-72.1773900720813, 109.06220727009612], [-71.89336314892358, 108.6431249918596], [-71.60849544417493, 108.224613773609], [-71.32278810599578, 107.80667530215543], [-71.03624228593067, 107.38931126200156], [-70.74885913890364, 106.97252333533476], [-70.4606398232136, 106.55631320202045], [-70.17158550052963, 106.14068253959516], [-69.88169733588634, 105.72563302325993], [-69.5909764976791, 105.3111663258735], [-69.29942415765936, 104.89728411794545], [-69.00704149093002, 104.4839880676297], [-68.71468804993864, 104.0724604449366], [-68.43032999823672, 103.67099551873791], [-68.14746312825935, 103.26847853474263], [-67.86609133233222, 102.86491503169485], [-67.58621848220847, 102.46031056273901], [-67.30784842901517, 102.05467069534353], [-67.03098500320064, 101.64800101122424], [-66.7556320144815, 101.24030710626748], [-66.48179325179044, 100.83159459045318], [-66.20947248322382, 100.42186908777762], [-65.93867345599014, 100.01113623617607], [-65.66939989635827, 99.59940168744518], [-65.40165550960623, 99.18667110716524], [-65.13544397997018, 98.77295017462222], [-64.87076897059376, 98.35824458272955], [-64.60763412347768, 97.94256003794993], [-64.3449342468906, 97.52421358186383], [-64.07885654119875, 97.09942270253885], [-63.81277883550689, 96.67463182321386], [-63.546701129815034, 96.24984094388887], [-63.28062342412318, 95.82505006456388], [-63.01454571843132, 95.4002591852389]], 'left': [[-71.22298769962953, 112.12802857520833], [-70.94511763048632, 111.71085621638714], [-70.66641064606918, 111.29424252353229], [-70.38686782529801, 110.87818920532374], [-70.10649029487125, 110.46269793866603], [-69.82527918485164, 110.04777039819831], [-69.54323562866165, 109.63340825628771], [-69.26036076307894, 109.21961318302243], [-68.97665572823178, 108.80638684620517], [-68.6921216675944, 108.39373091134628], [-68.4067597279825, 107.98164704165714], [-68.12057105954847, 107.5701368980434], [-67.83355681577686, 107.15920213909831], [-67.5457181534797, 106.74884442109605], [-67.25705623279187, 106.33906539798498], [-66.96757221716636, 105.92986672138109], [-66.67727944825722, 105.52124137723607], [-66.3875611932442, 105.11220876088824], [-66.09936222901462, 104.70210425525443], [-65.81268652126525, 104.2909335034851], [-65.52753801473247, 103.87870216340254], [-65.24392063313789, 103.465415907423], [-64.96183827913445, 103.05108042247869], [-64.6812948342527, 102.63570140993951], [-64.40229415884735, 102.21928458553455], [-64.12484009204421, 101.80183567927351], [-63.848936451687294, 101.38336043536779], [-63.57458703428637, 100.96386461215148], [-63.30179561496463, 100.54335398200216], [-63.030565947406814, 100.1218343312614], [-62.76090176380751, 99.69931146015514], [-62.492806774819826, 99.27579118271397], [-62.22624729310771, 98.85130272426166], [-61.960169587415855, 98.42651184493667], [-61.694091881724006, 98.00172096561168], [-61.42801417603215, 97.5769300862867], [-61.161936470340294, 97.1521392069617], [-60.89585876464844, 96.72734832763672]], 'right': [[-75.38708939626284, 109.36031163518437], [-75.10371181204212, 108.93487090786509], [-74.81942334979254, 108.50991397202974], [-74.53428231245493, 108.08552863308785], [-74.24828984929134, 107.66171660152622], [-73.96144711299553, 107.23847958552086], [-73.67375525968824, 106.81581929093028], [-73.38521544891263, 106.39373742128842], [-73.09582884362956, 105.97223567779794], [-72.80559661021286, 105.55131575932324], [-72.51451991844469, 105.13097936238374], [-72.22259994151081, 104.71122818114691], [-71.92983785599584, 104.29206390742155], [-71.63623484187848, 103.87348823065093], [-71.34179208252687, 103.45550283790593], [-71.04651076469369, 103.03810941387832], [-70.75209665162008, 102.62367951263712], [-70.47309880322922, 102.22978227658758], [-70.19556402750406, 101.83485281423083], [-69.91949614339919, 101.43889655990459], [-69.64489894968445, 101.04191896207547], [-69.37177622489246, 100.64392548326407], [-69.10013172726684, 100.24492159996977], [-68.82996919471033, 99.84491280259545], [-68.5612923447335, 99.4439045953718], [-68.29410487440343, 99.04190249628174], [-68.028410460293, 98.63891203698437], [-67.76421275843019, 98.23493876273888], [-67.50151540424784, 97.82998823232832], [-67.24032201253354, 97.42406601798304], [-66.98063617738, 97.01717770530398], [-66.72246147213552, 96.6093288931859], [-66.46362120067349, 96.19712443946601], [-66.19754349498163, 95.77233356014102], [-65.93146578928977, 95.34754268081603], [-65.66538808359792, 94.92275180149105], [-65.39931037790606, 94.49796092216606], [-65.1332326722142, 94.07317004284107]], 'next_id': ['0.0.1.-1'], 'pre_id': []}]


