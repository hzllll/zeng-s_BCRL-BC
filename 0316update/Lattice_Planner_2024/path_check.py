import shapely.geometry as sg
from rtree import index
import numpy as np
import logging
import math

# 配置优化参数
COORD_PRECISION = 3  # 坐标精度（小数位数）
SIMPLIFY_TOLERANCE = 0.1  # 几何简化容差
BUFFER_DISTANCE = 0.0  # 缓冲距离


def optimized_preprocess(observation, path, map_data):
    """高性能单线程预处理（兼容各Shapely版本）"""
    idx = index.Index(interleaved=True)
    shapes = []
    road_count = 0
    ego_x = observation.ego_info.x
    ego_y = observation.ego_info.y
    ego_yaw = observation.ego_info.yaw
    len_path = ((path[0][0][0] - path[0][0][-1]) ** 2 + (path[0][1][0] - path[0][1][-1]) ** 2) ** 0.5
    if observation.test_info['dt'] == 0.04:
        gap = 5  # 4
    else:
        gap = 2  # 2

    for road in map_data:
        # 数据校验
        if 'left' not in road or 'right' not in road:
            continue

        # 转换为numpy数组提升处理速度
        try:
            left = np.array(road['left'][::gap], dtype=np.float32)
            right = np.array(road['right'][::gap], dtype=np.float32)
        except ValueError:
            # logging.warning(f"Invalid coordinates in road {road.get('id', 'unknown')}")
            continue

        # 快速长度校验
        min_len = min(len(left), len(right))
        if min_len < 2:
            continue
        # dis_path = ((left[-1][0] - left[0][0]) ** 2 + (left[-1][1] - left[0][1]) ** 2) ** 0.5
        # dis_ll = ((left[0][0] - ego_x) ** 2 + (left[0][1] - ego_y) ** 2) ** 0.5
        # # dis_rl = ((right[i][0] - ego_x) ** 2 + (right[i][1] - ego_y) ** 2) ** 0.5
        # dis_rr = ((left[-1][0] - ego_x) ** 2 + (left[-1][1] - ego_y) ** 2) ** 0.5
        # # dis_lr = ((right[i + 1][0] - ego_x) ** 2 + (right[i + 1][1] - ego_y) ** 2) ** 0.5
        # if max(dis_ll, dis_rr) > 1.1 * dis_path:
        #     continue

        # 批量处理坐标
        left = left[:min_len]
        right = right[:min_len]

        # 生成四边形批处理
        for i in range(min_len - 1):
            # 优化顶点顺序和精度控制
            if i % 1 == 0:
                dx = left[i][0] - ego_x
                dy = left[i][1] - ego_y

                dis_ll = (dx ** 2 + dy ** 2) ** 0.5
                if dis_ll > 1.1 * len_path:
                    continue
                if dis_ll > 5:
                    forward_x = math.cos(ego_yaw)
                    forward_y = math.sin(ego_yaw)
                    dot_product = dx * forward_x + dy * forward_y
                    if dot_product < 0:
                        # print('pass road!!!!!!!!!!!')
                        continue
                # dis_ll = ((left[i][0] - ego_x) ** 2 + (left[i][1] - ego_y) ** 2) ** 0.5
                # if dis_ll > 1.2 * len_path:
                #     continue
                # gap_yaw = np.arctan2(left[i][1] - ego_y, left[i][0] - ego_x)
                # # print(((gap_yaw - ego_yaw) + np.pi) / np.pi)
                # if abs(((gap_yaw - ego_yaw) + np.pi) / np.pi) > np.pi / 2:
                #     continue


                ll = tuple(np.round(left[i], COORD_PRECISION))
                rl = tuple(np.round(right[i], COORD_PRECISION))
                rr = tuple(np.round(right[i + 1], COORD_PRECISION))
                lr = tuple(np.round(left[i + 1], COORD_PRECISION))


                # 创建多边形并自动修复几何
                try:
                    poly = sg.Polygon([ll, rl, rr, lr])

                    # 几何有效性处理
                    if not poly.is_valid:
                        poly = poly.buffer(BUFFER_DISTANCE).simplify(SIMPLIFY_TOLERANCE)
                        if not poly.is_valid or poly.is_empty:
                            continue
                except Exception as e:
                    # logging.debug(f"几何创建失败: {str(e)}")
                    continue

                # 插入索引
                bbox = poly.bounds
                idx.insert(road_count, bbox)
                shapes.append(poly)
                road_count += 1

    return idx, shapes


def batch_check_path(path, idx, shapes):
    """批量路径检测优化"""
    path_status = np.zeros(len(path), dtype=bool)

    for path_id in range(len(path)):
        if path_id >= len(path):
            path_status[path_id] = False
            continue

        xs, ys = path[path_id][0][::5], path[path_id][1][::5]  # 5
        if not xs or not ys or len(xs) != len(ys):
            path_status[path_id] = False
            continue

        # 批量处理坐标点
        points = np.column_stack((
            np.round(xs, COORD_PRECISION),
            np.round(ys, COORD_PRECISION)
        ))

        # 快速包围盒过滤
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        candidates = list(idx.intersection((x_min, y_min, x_max, y_max)))

        # 提前终止判断
        if not candidates:
            path_status[path_id] = False
            continue

        # 逐点检查
        valid = True
        for pt in points:
            p = sg.Point(pt)
            found = False

            # 精确查询优化
            for c in idx.intersection((pt[0], pt[1], pt[0], pt[1])):
                if shapes[c].contains(p):
                    found = True
                    break

            if not found:
                valid = False
                break

        path_status[path_id] = valid

    return path_status


def check_paths(observation, path, map_data):
    """主检查函数"""
    idx, shapes = optimized_preprocess(observation, path, map_data)
    return batch_check_path(path, idx, shapes)

# # 上述程序中对道路以及创建的多边形进行与自车的距离检测，距离小于20m是才对多边形进行遍历，否则跳过改多边形或道路，要求生成完整程序
# import shapely.geometry as sg
# from rtree import index
# import numpy as np
# import logging
# from math import floor, ceil
#
# # 配置参数
# COORD_PRECISION = 3  # 坐标处理精度
# DISTANCE_THRESHOLD = 20.0  # 距离阈值（米）
#
#
# class RoadCache:
#     """带距离检测的道路缓存系统"""
#
#     def __init__(self, map_data):
#         self.idx = index.Index(interleaved=True)
#         self.polygons = []
#         self._build_index(map_data)
#
#     def _build_index(self, map_data):
#         """预处理所有道路并构建索引"""
#         for road in map_data:
#             if 'left' not in road or 'right' not in road:
#                 continue
#
#             try:
#                 left = np.array(road['left'], dtype=np.float32)
#                 right = np.array(road['right'], dtype=np.float32)
#             except ValueError:
#                 continue
#
#             min_len = min(len(left), len(right))
#             if min_len < 2:
#                 continue
#
#             for i in range(min_len - 1):
#                 # 创建多边形
#                 ll = tuple(np.round(left[i], COORD_PRECISION))
#                 rl = tuple(np.round(right[i], COORD_PRECISION))
#                 rr = tuple(np.round(right[i + 1], COORD_PRECISION))
#                 lr = tuple(np.round(left[i + 1], COORD_PRECISION))
#
#                 try:
#                     poly = sg.Polygon([ll, rl, rr, lr])
#                     if not poly.is_valid:
#                         poly = poly.buffer(0)
#                 except:
#                     continue
#
#                 if poly.is_valid and not poly.is_empty:
#                     self.idx.insert(len(self.polygons), poly.bounds)
#                     self.polygons.append(poly)
#
#     def get_nearby_polygons(self, path_geometry):
#         """获取路径附近的多边形（两步过滤）"""
#         # 第一步：空间索引快速过滤
#         expanded_bbox = self._expand_bbox(path_geometry.bounds)
#         candidates = list(self.idx.intersection(expanded_bbox))
#
#         # 第二步：精确距离计算
#         return [self.polygons[i] for i in candidates
#                 if path_geometry.distance(self.polygons[i]) <= DISTANCE_THRESHOLD]
#
#     def _expand_bbox(self, bbox):
#         """扩展包围盒范围"""
#         return (
#             bbox[0] - DISTANCE_THRESHOLD,
#             bbox[1] - DISTANCE_THRESHOLD,
#             bbox[2] + DISTANCE_THRESHOLD,
#             bbox[3] + DISTANCE_THRESHOLD
#         )
#
#
# def check_paths(paths, map_data):
#     """主检查函数"""
#     road_cache = RoadCache(map_data)
#     results = np.zeros(len(paths), dtype=bool)
#     # path_status = np.zeros(len(path), dtype=bool)
#
#     for path_id in range(9):
#         # 输入校验
#         if path_id >= len(paths) or len(paths[path_id]) != 2:
#             results[path_id] = False
#             continue
#
#         xs, ys = paths[path_id][0], paths[path_id][1]
#         if not xs or not ys or len(xs) != len(ys):
#             results[path_id] = False
#             continue
#
#         # 创建路径几何体
#         try:
#             points = np.column_stack((xs, ys))
#             path_line = sg.LineString(points) if len(points) > 1 else sg.Point(points[0])
#         except:
#             results[path_id] = False
#             continue
#
#         # 获取附近多边形
#         nearby_polys = road_cache.get_nearby_polygons(path_line)
#         if not nearby_polys:
#             results[path_id] = False
#             continue
#
#         # 检查每个点是否在有效多边形内
#         valid = True
#         for pt in points:
#             p = sg.Point(pt)
#             if not any(poly.contains(p) for poly in nearby_polys):
#                 valid = False
#                 break
#
#         results[path_id] = valid
#
#     return results


# # 示例用法
# if __name__ == "__main__":
#     # 测试数据
#     test_map = [
#         {'left': [[0, 0], [10, 0], [20, 0]], 'right': [[0, 5], [10, 5], [20, 5]]},  # 道路宽5米
#         {'left': [[50, 50], [60, 50]], 'right': [[50, 55], [60, 55]]}  # 远处道路
#     ]
#
#     test_paths = [
#         [[5, 2], [5, 2]],  # 在道路内
#         [[15, 6], [15, 6]],  # 在道路外但距离<20米
#         [[25, 2], [25, 2]]  # 距离超过20米
#     ]
#
#     print(check_paths(test_paths, test_map))
#     # 期望输出：{'path_road[0}': True, 'path_road[1}': False, 'path_road[2}': False}

# # 修改为仅遍历附近道路，而不是遍历全部道路，以提高性能

# import shapely.geometry as sg
# from rtree import index
# import numpy as np
# import logging
#
# # 全局配置参数
# COORD_PRECISION = 3  # 坐标精度（小数位）
# SEARCH_BUFFER = 50.0  # 道路搜索缓冲距离
# SIMPLIFY_TOLERANCE = 0.1  # 几何简化阈值
# GRID_SIZE = 100.0  # 空间网格尺寸
#
#
# class RoadSpatialCache:
#     """高性能道路缓存系统"""
#
#     def __init__(self, map_data):
#         self.road_index = index.Index(interleaved=True)
#         self.grid_cache = {}
#         self._preprocess_roads(map_data)
#
#     def _preprocess_roads(self, map_data):
#         """预生成所有道路段多边形并构建空间索引"""
#         self.all_polygons = []
#         self.poly_bounds = []
#
#         for road in map_data:
#             if 'left' not in road or 'right' not in road:
#                 continue
#
#             left = np.asarray(road['left'], dtype=np.float32)
#             right = np.asarray(road['right'], dtype=np.float32)
#             min_len = min(len(left), len(right))
#
#             for i in range(min_len - 1):
#                 # 快速生成四边形
#                 ll = tuple(np.round(left[i], COORD_PRECISION))
#                 rl = tuple(np.round(right[i], COORD_PRECISION))
#                 rr = tuple(np.round(right[i + 1], COORD_PRECISION))
#                 lr = tuple(np.round(left[i + 1], COORD_PRECISION))
#
#                 # 创建并缓存多边形
#                 try:
#                     poly = sg.Polygon([ll, rl, rr, lr])
#                     if not poly.is_valid:
#                         poly = poly.buffer(0)
#                 except:
#                     continue
#
#                 if poly.is_valid and not poly.is_empty:
#                     # 计算网格归属
#                     bounds = poly.bounds
#                     grid_x = int(bounds[0] // GRID_SIZE)
#                     grid_y = int(bounds[1] // GRID_SIZE)
#                     grid_key = (grid_x, grid_y)
#
#                     # 更新索引
#                     poly_id = len(self.all_polygons)
#                     self.all_polygons.append(poly)
#                     self.road_index.insert(poly_id, bounds)
#
#                     # 更新网格缓存
#                     if grid_key not in self.grid_cache:
#                         self.grid_cache[grid_key] = []
#                     self.grid_cache[grid_key].append(poly_id)
#
#     def query_polygons(self, bbox):
#         """快速查询候选多边形"""
#         # 网格过滤
#         min_x, min_y, max_x, max_y = bbox
#         grid_start_x = int(min_x // GRID_SIZE) - 1
#         grid_start_y = int(min_y // GRID_SIZE) - 1
#         grid_end_x = int(max_x // GRID_SIZE) + 1
#         grid_end_y = int(max_y // GRID_SIZE) + 1
#
#         candidates = set()
#         for x in range(grid_start_x, grid_end_x + 1):
#             for y in range(grid_start_y, grid_end_y + 1):
#                 if (x, y) in self.grid_cache:
#                     candidates.update(self.grid_cache[(x, y)])
#
#         # R树精确过滤
#         final_candidates = [
#             pid for pid in candidates
#             if self.road_index.intersection(bbox, objects=False)
#         ]
#         return [self.all_polygons[pid] for pid in final_candidates]
#
#
# def check_paths(paths, map_data):
#     """主检查函数（速度优化版）"""
#     # 初始化全局缓存
#     road_cache = RoadSpatialCache(map_data)
#     results = np.zeros(len(paths), dtype=bool)
#
#     for path_id in range(9):
#         # 获取路径数据
#         if path_id >= len(paths) or len(paths[path_id]) != 2:
#             results[path_id] = False
#             continue
#
#         xs, ys = paths[path_id][0], paths[path_id][1]
#         if not xs or not ys or len(xs) != len(ys):
#             results[path_id] = False
#             continue
#
#         # 转换为numpy数组
#         try:
#             points = np.column_stack((xs, ys))
#         except:
#             results[path_id] = False
#             continue
#
#         # 计算路径包围盒
#         min_x, min_y = np.floor(np.min(points, axis=0))
#         max_x, max_y = np.ceil(np.max(points, axis=0))
#         path_bbox = (min_x - SEARCH_BUFFER,
#                      min_y - SEARCH_BUFFER,
#                      max_x + SEARCH_BUFFER,
#                      max_y + SEARCH_BUFFER)
#
#         # 快速获取候选多边形
#         candidates = road_cache.query_polygons(path_bbox)
#         if not candidates:
#             results[path_id] = False
#             continue
#
#         # 批量检查所有点
#         valid = True
#         for pt in points:
#             x, y = round(pt[0], COORD_PRECISION), round(pt[1], COORD_PRECISION)
#             p = sg.Point(x, y)
#
#             # 快速包围盒过滤
#             if not any(poly.bounds[0] <= x <= poly.bounds[2] and
#                        poly.bounds[1] <= y <= poly.bounds[3]
#                        for poly in candidates):
#                 valid = False
#                 break
#
#             # 精确检查
#             if not any(poly.contains(p) for poly in candidates):
#                 valid = False
#                 break
#
#         results[path_id] = valid
#
#     return results


# # 示例用法
# if __name__ == "__main__":
#     # 测试数据
#     test_map = [
#         {'left': [[0, 0], [1, 1], [2, 2]], 'right': [[0, 1], [1, 2], [2, 3]]},
#         {'left': [[5, 5], [6, 6]], 'right': [[5, 6], [6, 7]]}
#     ]
#
#     test_paths = [
#         [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]],  # 有效
#         [[5.5, 6.5], [5.5, 6.5]],  # 有效
#         [[10, 10], [10, 10]]  # 无效
#     ]
#
#     print(check_paths(test_paths, test_map))