from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np


class global_planner:
    def __init__(self, map):
        self.map = map
        self.map_mid_x = []
        self.map_mid_y = []
        self.next_id = []
        self.pre_id = []
        self.left_id = []
        self.right_id = []
        self.parallel_id = []
        self.bad_end_point = []

    # 处理初始的地图数据，将每一个lane的前后左右id找到
    def map_data_process(self):
        # map_data = [[], [], [], [], [], [], [], []]
        map = self.map
        map_mid_x = [[]]
        map_mid_y = [[]]
        next_data = [[]]
        pre_data = [[]]
        map_id_data = [[], []]
        map_id = 0
        # map里的每一个点进行搜索，找到他们的id点和与前后id
        for i in map:
            for xx in range(len(i['mid'])):
                id = i['id'].split('.')[0]
                id1 = i['id'].split('.')[2]
                map_mid_x[map_id].append(i['mid'][xx][0])
                map_mid_y[map_id].append(i['mid'][xx][1])
            map_id_data[0].append(int(id))
            map_id_data[1].append(abs(int(id1)))
            map_mid_x.append([])
            map_mid_y.append([])

            if len(i['next_id'])!=0:
                for next_d in i['next_id']:
                    next_id = next_d.split('.')[0]
                    next_id1 = next_d.split('.')[2]
                    if next_id1 == 'None':
                        next_id1 =1
                    next_data[map_id].append(int(next_id))
                    next_data[map_id].append(abs(int(next_id1)))
            else:
                next_data[map_id].append(0)
            if len(i['pre_id'])!=0:
                for pre_d in i['pre_id']:
                    pre_id = pre_d.split('.')[0]
                    pre_id1 = pre_d.split('.')[2]
                    if pre_id1 == 'None':
                        pre_id1 =1
                    pre_data[map_id].append(int(pre_id))
                    pre_data[map_id].append(abs(int(pre_id1)))
            else:
                pre_data[map_id].append(0)
            map_id = map_id+1
            next_data.append([])
            pre_data.append([])

        # 将前后id转换为能看的index
        map_id = 0
        next_data_final = [[]]
        pre_data_final = [[]]
        for next2id in next_data:
            for id3 in range(len(next2id)//2):
                id_main = next2id[id3*2]
                id_sub = next2id[id3*2+1]
                id4 = []
                for index in range(len(map_id_data[0])):
                    if id_main == map_id_data[0][index]:
                        id4.append(index)
                for id5 in id4:
                    if id_sub == map_id_data[1][id5]:
                        next_data_final[map_id].append(id5)
                        break
            map_id = map_id+1
            next_data_final.append([])
        map_id = 0
        for pre2id in pre_data:
            for id3 in range(len(pre2id)//2):
                id_main = pre2id[id3*2]
                id_sub = pre2id[id3*2+1]
                id4 = []
                for index in range(len(map_id_data[0])):
                    if id_main == map_id_data[0][index]:
                        id4.append(index)
                for id5 in id4:
                    if id_sub == map_id_data[1][id5]:
                        pre_data_final[map_id].append(id5)
                        break
            map_id = map_id+1
            pre_data_final.append([])
        next_id = next_data_final
        pre_id = pre_data_final

        # 找到左右id
        left_id = [[]]
        right_id = [[]]
        for ori_point in range(len(map_id_data[0])):
            par_id_all = []
            for id6 in range(len(map_id_data[0])):
                if map_id_data[0][id6] == map_id_data[0][ori_point]:
                    par_id_all.append(id6)
            for par_id in par_id_all:
                gap_of_id = map_id_data[1][par_id]-map_id_data[1][ori_point]
                if gap_of_id == 1:
                    right_id[ori_point].append(par_id)
                if gap_of_id == -1:
                    left_id[ori_point].append(par_id)
            right_id.append([])
            left_id.append([])

        map_mid_x.pop()
        map_mid_y.pop()
        next_id.pop()
        next_id.pop()
        pre_id.pop()
        pre_id.pop()
        left_id.pop()
        right_id.pop()

        self.map_mid_x = map_mid_x
        self.map_mid_y = map_mid_y
        self.next_id = next_id
        self.pre_id = pre_id
        self.left_id = left_id
        self.right_id = right_id

        # 对next_id做校验
        err_flag = []
        for i in range(len(next_id)):
            last_pos = [map_mid_x[i][-1], map_mid_y[i][-1]]
            if next_id[i] != []:
                for i1 in next_id[i]:
                    next_pos = [map_mid_x[i1][0], map_mid_y[i1][0]]
                    distance = np.linalg.norm(np.array(last_pos) - np.array(next_pos))
                    if distance>2:
                        err_flag.append(i)
                        next_id[i] = []
                        break
            else:
                err_flag.append(i)
        if err_flag != []:
            next_pos_list = []
            for i in range(len(map_mid_x)):
                next_pos_list.append([map_mid_x[i][0], map_mid_y[i][0]])
        for i in range(len(err_flag)):
            path_id = err_flag[i]
            last_pos = [map_mid_x[path_id][-1], map_mid_y[path_id][-1]]
            for i1 in range(len(next_pos_list)):
                next_pos = next_pos_list[i1]
                distance = np.linalg.norm(np.array(last_pos) - np.array(next_pos))
                if distance<1:
                    next_id[path_id].append(i1)
                    break
        self.next_id = next_id

        parallel_id, same_direct_id = self.judge_parallel()
        self.parallel_id = parallel_id
        self.same_direct_id = same_direct_id

    def judge_parallel(self):
        parallel_data = []
        for i in range(len(self.map_mid_x)):
            if i == 7:
                a=  1
            try:
                last_pos1 = [self.map_mid_x[i][0], self.map_mid_y[i][0]]
                last_pos11 = [self.map_mid_x[i][5], self.map_mid_y[i][5]]
                last_pos2 = [self.map_mid_x[i][-1], self.map_mid_y[i][-1]]
                last_pos22 = [self.map_mid_x[i][-5], self.map_mid_y[i][-5]]
            except:
                last_pos1 = [self.map_mid_x[i][0], self.map_mid_y[i][0]]
                last_pos11 = [self.map_mid_x[i][1], self.map_mid_y[i][1]]
                last_pos2 = [self.map_mid_x[i][-1], self.map_mid_y[i][-1]]
                last_pos22 = [self.map_mid_x[i][-2], self.map_mid_y[i][-2]]
            last_direct1 = self.find_direction([last_pos1[0] - last_pos11[0]], [last_pos1[1] - last_pos11[1]])
            last_direct2 = self.find_direction([last_pos2[0] - last_pos22[0]], [last_pos2[1] - last_pos22[1]])
            parallel_data.append([last_pos1, last_pos2, last_direct1, last_direct2])
        parallel_id = []
        same_direct_id = []
        for i in range(len(parallel_data)):
            parallel_id_ = []
            same_direct_id_ = []
            for j in range(len(self.map_mid_x)):
                if i == 6 and j == 7:
                    a = 1
                direct1 = parallel_data[i][2]-parallel_data[j][2]
                direct2 = parallel_data[i][3]-parallel_data[j][3]
                direct1 = self.normalize_radians(direct1)
                direct2 = self.normalize_radians(direct2)
                direct11 = self.find_direction([parallel_data[i][0][0]-parallel_data[i][1][0]],[parallel_data[i][0][1]-parallel_data[i][1][1]])
                direct22 = self.find_direction([parallel_data[j][0][0]-parallel_data[j][1][0]],[parallel_data[j][0][1]-parallel_data[j][1][1]])
                direct3 = direct11-direct22
                direct3 = self.normalize_radians(direct3)
                distance1 = np.linalg.norm(np.array(parallel_data[i][0]) - np.array(parallel_data[j][0]))
                distance2 = np.linalg.norm(np.array(parallel_data[i][1]) - np.array(parallel_data[j][1]))
                # 1/2.起点与终点距离不超过一条车道且方向一致
                # 3.两条线之间起点与终点距离大于0.5米，防止是道路合并或者分叉
                # 4.两条线之间起始的角度差小于30度
                # 5.两个端点的距离怎么样都不能大于7米
                judge1 = direct1<np.pi/8 and distance1<4.5 
                judge2 = direct2<np.pi/8 and distance2<4.5 
                judge3 = distance1 > 0.5 and distance2 > 0.5
                judge4 = direct3 < np.pi/8
                judge5 = distance1 < 7 and distance2 < 7
                j1 = judge1 or judge2
                if j1 and judge3 and judge4 and judge5:
                    if j!=i:
                        parallel_id_.append(j)
                judge5 =  j in self.next_id[i]
                if judge5 and judge4:
                    same_direct_id_.append(j) 
            parallel_id.append(parallel_id_)
            same_direct_id.append(same_direct_id_)
        return parallel_id, same_direct_id

    def find_direction(self, dx, dy):
        # k = np.polyfit(dx,dy, 1)[0]
        if dx[0] == 0:
            if dy[0] > 0:
                direction = np.pi/2
            else:
                direction = -np.pi/2
            return direction
        k = dy[0]/dx[0]
        direction = np.arctan(k)
        # 第三象限/第二象限的角度转换，将角度转换为[-pi,pi]
        if k>=0 and dx[0]<0:
            direction = direction - np.pi
        elif k<=0 and dx[0]<0:
            direction = direction + np.pi
        # direction = direction+np.pi/2
        direction = self.normalize_radians(direction)
        return direction

    def normalize_radians(self, theta):
        # 将所有角度整理到[0,2pi]
        if theta > np.pi*2:
            theta -= np.pi*2
        elif theta < -np.pi*2:
            theta += np.pi*2

        if theta > np.pi:
            theta -= np.pi*2
        elif theta < -np.pi:
            theta += np.pi*2
        
        # if theta < 0:
        #     theta += np.pi*2

        return theta

    def search_map_id(self, ego_pos, expand_flag):
        map_mid_x = self.map_mid_x
        map_mid_y = self.map_mid_y
        now_id = []
        now_path_point = []

        if len(ego_pos) == 1:
            limitation_x = [ego_pos[0][0]+2,ego_pos[0][0]-2]
            limitation_y = [ego_pos[0][1]+2,ego_pos[0][1]-2]
            else_x = 0
        else:
            if ego_pos[0][0]>ego_pos[1][0]:
                limitation_x = [ego_pos[0][0],ego_pos[1][0]]
            else:
                limitation_x = [ego_pos[1][0],ego_pos[0][0]]
            if ego_pos[0][1]>ego_pos[1][1]:
                limitation_y = [ego_pos[0][1],ego_pos[1][1]]
            else:
                limitation_y = [ego_pos[1][1],ego_pos[0][1]]
            else_x = 1
        distance = 114514
        if expand_flag == True:
            limitation_x[0] = limitation_x[0]+2
            limitation_x[1] = limitation_x[1]-2
            limitation_y[0] = limitation_y[0]+2
            limitation_y[1] = limitation_y[1]-2
            if else_x == 1:
                self.bad_end_point = [(limitation_x[0]+limitation_x[1])/2,(limitation_y[0]+limitation_y[1])/2]
        distance_min = 114514
        distance_current = 114514
        for i in range(len(map_mid_x)):
            for i1 in range(len(map_mid_x[i])):
                if expand_flag == False:
                    search_point = len(map_mid_x[i])-i1-1
                else:
                    search_point = i1
                ifx = (map_mid_x[i][search_point]<limitation_x[0]) and (map_mid_x[i][search_point]>limitation_x[1])
                ify = (map_mid_y[i][search_point]<limitation_y[0]) and (map_mid_y[i][search_point]>limitation_y[1])
                if ifx and ify:
                    if else_x == 0:
                        try:
                            direction = self.find_direction([map_mid_x[i][search_point]-map_mid_x[i][search_point-1]],[map_mid_y[i][search_point]-map_mid_y[i][search_point-1]])
                        except:
                            direction = self.find_direction([map_mid_x[i][search_point+1]-map_mid_x[i][search_point]],[map_mid_y[i][search_point+1]-map_mid_y[i][search_point]])
                        ego_direction = self.normalize_radians(self.ego_state[3])
                        direction = self.normalize_radians(direction-ego_direction)
                        if abs(direction)>np.pi/2:
                            continue
                        a = np.array([map_mid_x[i][search_point],map_mid_y[i][search_point]])
                        b = np.array(ego_pos)
                        distance = np.linalg.norm(a - b)
                        if distance<distance_min:
                            distance_min = distance
                            now_id_ = [i]
                            now_path_point_ = [search_point]
                    elif else_x == 1:
                        now_id.append(i)
                        now_path_point.append(search_point)
                        break  
            if distance_min!=114514 and else_x == 0:
                if distance_current > distance_min:
                    now_id = now_id_
                    now_path_point = now_path_point_
                    distance_current = distance_min
                distance_min = 114514 
        return now_id, now_path_point
    
    def detect_now_lane(self, i_id, ego_pos, passing_id):
        map_mid_x = self.map_mid_x
        map_mid_y = self.map_mid_y
        now_id = passing_id[i_id]
        tar_pos = [map_mid_x[now_id][-1], map_mid_y[now_id][-1]]
        distance = np.linalg.norm(np.array(tar_pos) - np.array(ego_pos))
        if distance<4:
            i_id = i_id+1
        return i_id
    
    def find_next_lane(self, temp_id_sequence,lane_has_passed):
        temp_id_sequence_ = []
        path_need_to_be_delete = []
        temp_id_sequence_have_p = []
        passing_id_= []
        for i in range(len(temp_id_sequence)):
            id_now = temp_id_sequence[i][-1]
            id_next = self.next_id[id_now]
            # 0.如果下一段为空，那这个到头了就能删了
            # 1.如果下一段为只有一个，说明无分叉
            # 2.如果分叉了，那么就不管他的左右变道
            if id_next == []:
                path_need_to_be_delete.append(i)
                if self.parallel_id[id_now] != []:
                    temp_id_sequence_have_p = temp_id_sequence[i].copy()
                continue
            if len(id_next) == 1:
                id_next = id_next[0]
                new_id_sequence = temp_id_sequence[i].copy()
                new_id_sequence.append(id_next)
                temp_id_sequence.append(new_id_sequence)
                if id_next in self.tar_id:
                    passing_id_.append(new_id_sequence)
            else:
                for id_next_ in id_next:
                    temp_id_sequence_1 = temp_id_sequence[i].copy()
                    temp_id_sequence_1.append(id_next_)
                    temp_id_sequence_.append(temp_id_sequence_1)
                    if id_next_ in self.tar_id:
                        passing_id_.append(temp_id_sequence_[-1])
                # path_need_to_be_delete.append(i)
        # for i in path_need_to_be_delete:
        #     temp_id_sequence.pop(path_need_to_be_delete[i])
        temp_id_sequence = [temp_id_sequence[i] for i in range(len(temp_id_sequence)) if i not in path_need_to_be_delete]
        for i in temp_id_sequence_:
            temp_id_sequence.append(i)
        # 如果一个断头路还能变道，那就让他变，把它放在第一个
        if temp_id_sequence_have_p != []:
            temp_id_sequence_have_p = [temp_id_sequence_have_p]
            for i in temp_id_sequence:
                temp_id_sequence_have_p.append(i)
                # if len(temp_id_sequence_have_p) == 1:
                #     temp_id_sequence = [temp_id_sequence_have_p]
                # else:
            temp_id_sequence = temp_id_sequence_have_p
        if temp_id_sequence == []:
            return temp_id_sequence, passing_id_
        # 把当前的终点全部抠出来，用于后续左右变道的对比
        gone_id = []
        for i in range(len(temp_id_sequence)):
            gone_id.append(temp_id_sequence[i][-1]) 
        # 左右变道
        path_need_to_be_delete = []
        temp_id_sequence_ = []
        for i in range(len(temp_id_sequence)):
            id_next = temp_id_sequence[i][-1]
            p_id = self.parallel_id[id_next]
            if len(p_id) != 0:
                intersection_p = list(set(p_id) & set(gone_id))
                p_id = [item for item in p_id if item not in intersection_p]
                for p_id_ in p_id:
                    temp_id_sequence_1 = temp_id_sequence[i].copy()
                    temp_id_sequence_1.append(p_id_)
                    temp_id_sequence_.append(temp_id_sequence_1)
                    if p_id_ in self.tar_id:
                        passing_id_.append(temp_id_sequence_[-1])
        for i in temp_id_sequence_:
            if i not in lane_has_passed:
                lane_has_passed.append(i)
                temp_id_sequence.append(i)
        # if temp_id_sequence_have_p != []:
        #     temp_id_sequence_have_p = []
        #     for i in range(len(temp_id_sequence)-1):
        #         temp_id_sequence_have_p.append(temp_id_sequence[i+1])
        #     temp_id_sequence = temp_id_sequence_have_p
        return temp_id_sequence, passing_id_, lane_has_passed
            
    # 寻找
    def find_dis(self, ref_point, path_id):
        map_mid_x = self.map_mid_x
        map_mid_y = self.map_mid_y
        if isinstance(path_id, list):
            path_id = path_id[0]
        ref_point = np.array(ref_point)
        search_path = [[],[]]
        search_path[0] = map_mid_x[path_id]
        search_path[1] = map_mid_y[path_id]
        distance = 114514
        distance_min = 114514
        distance_min_index = 1010100
        # print('now_id:',path_id)
        for i in range(len(search_path[0])):
            path_point = np.array([search_path[0][i],search_path[1][i]])
            distance = np.linalg.norm(path_point-ref_point)
            if distance<distance_min:
                distance_min = distance
                distance_min_index = i
        return distance_min, distance_min_index

    def lane_changer(self, max_gap_distance, id, global_path, open2access):
        # 读取下一个id的坐标
        next_lane = [self.map_mid_x[self.passing_id[id]],self.map_mid_y[self.passing_id[id]]]
        # 定义当前的变道位置,一般是需要变道的车道的初始点
        now_lane_change = [global_path[-1,0],global_path[-1,1]]
        # 确定需要变道的时候的方向,以免变错
        if global_path.shape[0] == 1:
            last_lane_direction = self.ego_state[3]
            last_lane_direction = self.normalize_radians(last_lane_direction)
            # last_lane_direction = np.tan(last_lane_direction)
            # last_lane_direction = np.tan(last_lane_direction)
        else:
            last_lane_direction = self.find_direction([global_path[-1,0]-global_path[-2,0]], [global_path[-1,1]-global_path[-2,1]])
            # last_lane_direction = np.tan(last_lane_direction)
        next_flag = -1

        # 在需要变过去的车道找有没有一个适合变的点
        # 先是搜路径搜出来的原始车道
        for change_point in range(len(next_lane[0])):
            continue_flag = False
            change_point_ = len(next_lane[0])-change_point-1
            next_lane_change = [next_lane[0][change_point_],next_lane[1][change_point_]]
            gap_distance = np.linalg.norm(np.array(now_lane_change) - np.array(next_lane_change))
            if abs(gap_distance-max_gap_distance) < 1.5:
                change_point = change_point_
                next_lane_point = [next_lane[0][change_point],next_lane[1][change_point]]
                direction_judge = self.find_direction([next_lane_change[0]-now_lane_change[0]], [next_lane_change[1]-now_lane_change[1]])
                direction_judge = abs(self.normalize_radians(direction_judge-last_lane_direction))
                if abs(direction_judge) < np.pi/2:
                    next_flag = 0
                    nxt_lane = next_lane
                else:
                    gap_distance = 114514
                break
        if open2access:
        # 如果没有找到合适的点，就在下一个车道里找
            if abs(gap_distance-max_gap_distance) > 1.5:
                next2_lane = [self.map_mid_x[self.passing_id[id+1]],self.map_mid_y[self.passing_id[id+1]]]
                for change_point in range(len(next2_lane[0])):
                    change_point_ = len(next2_lane[0])-change_point-1
                    next_lane_change = [next2_lane[0][change_point_],next2_lane[1][change_point_]]
                    gap_distance = np.linalg.norm(np.array(now_lane_change) - np.array(next_lane_change))
                    if abs(gap_distance-max_gap_distance) < 1.5:
                        next_flag = 1
                        change_point = change_point_
                        nxt_lane = next2_lane
                        next_lane_point = [next2_lane[0][change_point],next2_lane[1][change_point]]
                        break
        if next_flag == -1:
            raise Exception("找不到能变道的地方")
        last_lane_point = global_path[-1]
        try:
            next_lane_direction = self.find_direction([nxt_lane[0][change_point]-nxt_lane[0][change_point-1]], [nxt_lane[1][change_point]-nxt_lane[1][change_point-1]])
        except:
            next_lane_direction = self.find_direction([nxt_lane[0][change_point+1]-nxt_lane[0][change_point]], [nxt_lane[1][change_point+1]-nxt_lane[1][change_point]])
        change_lane = self.change_lane_spline(last_lane_point, next_lane_point, last_lane_direction, next_lane_direction)

        change_lane = np.array(change_lane)
        change_lane = np.transpose(change_lane)
        change_lane = change_lane[1:-2,:]
        global_path = np.append(global_path,change_lane,axis=0)
        # global_path = global_path[1:-1,:]
        
        # 判断变道后是否还需要变道
        if next_flag == 1:
            if id+2 < len(self.passing_id):
                if self.passing_id[id+2] in self.parallel_id[self.passing_id[id+1]]:
                    continue_flag = True
                else:
                    now_lane = nxt_lane
                    now_lane = np.array(now_lane)
                    now_lane = np.transpose(now_lane)
                    now_lane = now_lane[change_point:-1,:]
                    global_path = np.append(global_path,now_lane,axis=0)
                    continue_flag = True
            else:
                now_lane = nxt_lane
                now_lane = np.array(now_lane)
                now_lane = np.transpose(now_lane)
                now_lane = now_lane[change_point:-1,:]
                global_path = np.append(global_path,now_lane,axis=0)
                continue_flag = True
        else:
            now_lane = nxt_lane
            now_lane = np.array(now_lane)
            now_lane = np.transpose(now_lane)
            now_lane = now_lane[change_point:-1,:]
            global_path = np.append(global_path,now_lane,axis=0)
        return global_path, continue_flag

    def change_lane_spline(self,last_lane_point, next_lane_point, last_lane_direction, next_lane_direction):
        next_lane_point, next_lane_direction = self.relative_coordinates(last_lane_point, next_lane_point, last_lane_direction, next_lane_direction)
        llp = last_lane_point
        last_lane_point = [0,0]
        lld = last_lane_direction
        last_lane_direction = 0
        next_lane_direction = np.tan(next_lane_direction)

        if last_lane_point[0]<next_lane_point[0]:
            cs = CubicSpline([last_lane_point[0],next_lane_point[0]], [last_lane_point[1],next_lane_point[1]], bc_type=((1, last_lane_direction), (1, next_lane_direction)))
        else:
            cs = CubicSpline([next_lane_point[0],last_lane_point[0]], [next_lane_point[1],last_lane_point[1]], bc_type=((1, next_lane_direction), (1, last_lane_direction)))
        coeffs = cs.c
        gap_distance = np.linalg.norm(np.array(next_lane_point) - np.array(last_lane_point))
        gap_distance = int(gap_distance/0.4)
        change_lane = [[],[]]
        change_lane[0] = np.linspace(last_lane_point[0],next_lane_point[0],gap_distance)
        change_lane[1] = cs(change_lane[0])
        for pi in range(len(change_lane[0])):
            old_pos = [change_lane[0][pi],change_lane[1][pi]]
            new_pos,_ = self.relative_coordinates([0,0], old_pos, -lld, 0)
            new_pos = new_pos+llp
            change_lane[0][pi] = new_pos[0]
            change_lane[1][pi] = new_pos[1]
        return change_lane

    def relative_coordinates(self, p1, p2, t1, t2):
        # 计算相对坐标
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        
        # 旋转坐标
        x_prime = delta_x * np.cos(t1) + delta_y * np.sin(t1)
        y_prime = -delta_x * np.sin(t1) + delta_y * np.cos(t1)

        t2_ = t2 - t1
        
        return [x_prime, y_prime], np.tan(t2_)

    def lane_changer_param(self, id, id_plus,global_path):

        # 定义当前的道路与下一条道路
        id_now_lane = self.passing_id[id]
        if id_plus != -1:
            id_next_lane = self.passing_id[id+1]
            next_lane = [self.map_mid_x[id_next_lane],self.map_mid_y[id_next_lane]]
        else:
            id_next_lane = id_now_lane
        # 尝试一下有没有下下条道路，并尝试下下条道路是不是与下一条道路相连
        try:
            id_next2_lane = self.passing_id[id+2+id_plus]
            if id_next2_lane in self.same_direct_id[id_next_lane]:
                next_lane = [self.map_mid_x[id_next2_lane],self.map_mid_y[id_next2_lane]]
                open2access = True
            else:
                next_lane = [self.map_mid_x[id_next_lane],self.map_mid_y[id_next_lane]]
                open2access = False
        except:
            next_lane = [self.map_mid_x[id_next_lane],self.map_mid_y[id_next_lane]]
            open2access = False
        # 计算应有的最大变道距离
        max_gap_distance = self.ego_state[2]*4
        max_gap_distance_limitation = np.linalg.norm(np.array([global_path[-1][0],global_path[-1][1]])
                                                        -np.array([next_lane[0][-1],next_lane[1][-1]]))
        if max_gap_distance_limitation/2<15:
            max_gap_distance_ = max(10,min(abs(max_gap_distance_limitation*4/5), max_gap_distance))
        else:
            max_gap_distance_ = max(10,min(abs(max_gap_distance_limitation/2), max_gap_distance))
        # 计算当前当前位置往下个位置的变道路径
        try:
            global_path, continue_flag = self.lane_changer(max_gap_distance_, id+1+id_plus, global_path, open2access)
        except:
            max_gap_distance = max(0,min(abs(max_gap_distance_limitation/2), max_gap_distance))
            global_path, continue_flag = self.lane_changer(max_gap_distance, id+1+id_plus, global_path, open2access)

        return global_path, continue_flag

    def find_global_path(self, ref_point, ego_state):
        map_mid_x = self.map_mid_x.copy()
        map_mid_y = self.map_mid_y.copy()
        next_id = self.next_id
        # pre_id = self.pre_id
        # left_id = self.left_id
        # right_id = self.right_id
        parallel_id = self.parallel_id
        self.ego_state = ego_state
        ego_pos = [[ego_state[0], ego_state[1]]]

        # 1.找到当前初始点与目标点,并将初始和目标线改短
        self.id_start, self.id_start_point = self.search_map_id(ego_pos, False)
        self.tar_id, self.tar_id_point = self.search_map_id(ref_point, False)
        # 如果目标的为之内没有点,那就把它放大一点再搜一次
        if self.tar_id == []:
            self.tar_id, self.tar_id_point = self.search_map_id(ref_point, True)
            # distance_tar = 114514
            # for i in range(len(self.tar_id)):
            #     distance_tar_ = np.linalg.norm(np.array([map_mid_x[self.tar_id[i]][self.tar_id_point[i]],map_mid_y[self.tar_id[i]][self.tar_id_point[i]]])- 
            #                                    np.array(self.bad_end_point))
            #     if distance_tar_<distance_tar:
            #         distance_tar = distance_tar_
            #         distance_tar_id = i
            # self.tar_id = [self.tar_id[distance_tar_id]]
            # self.tar_id_point = [self.tar_id_point[distance_tar_id]]
        if self.id_start == []:
            self.id_start, self.id_start_point = self.search_map_id(ego_pos, True)
        # 如果终点位置不是整条路上的最后的点,那就把终点的路径缩短
        for i in range(len(self.tar_id)):
            if self.tar_id_point[i] != len(self.map_mid_x[self.tar_id[i]]):
                self.map_mid_x[self.tar_id[i]][self.tar_id_point[i]+1:] = [] 
                self.map_mid_y[self.tar_id[i]][self.tar_id_point[i]+1:] = []
        # 如果起始位置不是整条路上的第0点,那就把起始的路径缩短
        if self.id_start_point[0] != 0:
            self.map_mid_x[self.id_start[0]][:self.id_start_point[0]-1] = []
            self.map_mid_y[self.id_start[0]][:self.id_start_point[0]-1] = []
        
        map_mid_x = self.map_mid_x
        map_mid_y = self.map_mid_y

        # 2.开启循环
        id_now = self.id_start[0]
        self.passing_id = [id_now]
    
        while True:
            # 2.1 找一下下一个参考点是不是在当前的路径上
            if id_now in self.tar_id:
                break
            # 看看是不是平行两边的变道
            temp_id_sequence = [[id_now]]
            p_id = parallel_id[id_now]
            for p_id_ in p_id:
                if p_id_ in self.tar_id:
                    id_now = p_id_
                    self.passing_id.append(id_now)
                    break
                temp_id_sequence.append([id_now,p_id_])
            if self.passing_id[-1] in self.tar_id:
                break

            errflag = 0
            passing_id_ = []
            # 为了尽量晚变道，所以对于第一次变道，需要延后一次输入进临时点集
            data_append = []
            if len(temp_id_sequence) != 1:
                for i in range(len(temp_id_sequence)-1):
                    data_append.append(temp_id_sequence[i+1])
                temp_id_sequence = [temp_id_sequence[0]]
            lane_has_passed = []
            while passing_id_ == []:
                temp_id_sequence,passing_id_,lane_has_passed  = self.find_next_lane(temp_id_sequence,lane_has_passed)
                if data_append != []:
                    for i in data_append:
                        temp_id_sequence.append(i)
                    data_append = []
                errflag+=1
                if errflag>10:
                    raise Exception("Not find global path")
            if len(passing_id_) == 1:
                self.passing_id = passing_id_[0]
            else:
                # 删掉长的
                passing_id_lend = []
                for pid in passing_id_:
                    passing_id_lend.append(len(pid))
                min_len = min(passing_id_lend)
                passing_id__ = []
                for pid in passing_id_:
                    if len(pid) == min_len:
                        passing_id__.append(pid)
                passing_id_ = passing_id__
                if len(passing_id_) == 1:
                    self.passing_id = passing_id_[0]
                    break
                # 找谁变道晚一些
                change_lane_time = -1
                for pid in passing_id_:
                    for pid_ in range(len(pid)-1):
                        if pid[pid_+1] not in self.next_id[pid[pid_]]:
                            if pid_ > change_lane_time:
                                change_lane_time = pid_
                                self.passing_id = pid
                                break
                if self.passing_id[0] == self.id_start[0]:
                    self.passing_id = passing_id_[0]
            break
            

        # 找路径
        passing_id = self.passing_id
        # 如果路径的第一条就要变道，那直接把第一条删掉
        if len(passing_id)>1:
            if passing_id[1] in parallel_id[passing_id[0]]:
                passing_id.pop(0)
        # 路过的路径总数,指定循环的次数
        loop_time = len(self.passing_id)
        # 确定globaal_path,然后初始值车辆的当前position
        global_path = np.zeros([1,2])
        global_path[0,:] = np.array([ego_pos[0][0],ego_pos[0][1]])
        continue_flag = False
        # 变道需要的长度(假设路上都是匀速,以起点速度为准)
        max_gap_distance = ego_state[2]*4
        # max_gap_distance = min(abs(max_gap_distance),20)
        # 如果直接能到终点，那就拿出来直接用
        if loop_time == 1:
            id_now_lane = passing_id[0]
            now_lane = [map_mid_x[id_now_lane],map_mid_y[id_now_lane]]
            max_gap_distance_limitation = np.linalg.norm(np.array([now_lane[0][0],now_lane[1][0]])
                                                         -np.array([now_lane[0][-1],now_lane[1][-1]]))
            max_gap_distance = max(0,min(abs(max_gap_distance_limitation/2), max_gap_distance))
            global_path, _ = self.lane_changer(max_gap_distance, 0, global_path, False)
        # 开始找路循环
        for id in range(loop_time-1):
            max_gap_distance = ego_state[2]*4
            ending_location = global_path[-1,:]
            ending_loaction_passing = [map_mid_x[passing_id[id]][-1],map_mid_y[passing_id[id]][-1]]
            distance_ending = np.linalg.norm(np.array(ending_location) - np.array(ending_loaction_passing))
            if distance_ending < 2 and loop_time !=1 and continue_flag == False:
                continue
            if continue_flag:
                if id ==1 and passing_id[2] in parallel_id[passing_id[1]]:
                    continue_flag = False
                else:
                    continue_flag = False
                    continue
            ending_loaction_passing = [map_mid_x[passing_id[-1]][1],map_mid_y[passing_id[-1]][1]]
            distance_ending = np.linalg.norm(np.array(ending_location) - np.array(ending_loaction_passing))
            if distance_ending < 2 and loop_time !=1 :
                break
            # 读取当前的id与当前道路的坐标
            id_now_lane = passing_id[id]
            now_lane = [map_mid_x[id_now_lane],map_mid_y[id_now_lane]]
            # 检查是否需要变道
            id_next_lane = passing_id[id+1]
            id_next_map = next_id[passing_id[id]]
            if id_next_lane not in id_next_map:
                global_path, continue_flag = self.lane_changer_param(id, 0, global_path)
            else:
                if id == 0:
                    global_path, continue_flag = self.lane_changer_param(id, -1, global_path)
                else:
                    now_lane = [map_mid_x[passing_id[id]],map_mid_y[passing_id[id]]]
                    now_lane = np.array(now_lane)
                    now_lane = np.transpose(now_lane)
                    global_path = np.append(global_path,now_lane,axis=0)
                    if id == loop_time-1:
                        now_lane = [map_mid_x[passing_id[id+1]],map_mid_y[passing_id[id+1]]]
                        now_lane = np.array(now_lane)
                        now_lane = np.transpose(now_lane)
                        global_path = np.append(global_path,now_lane,axis=0)
            # self.draw_map(global_path, [passing_id[id]])
        ending_location = global_path[-1,:]
        ending_loaction_passing = [map_mid_x[passing_id[-1]][-1],map_mid_y[passing_id[-1]][-1]]
        distance_ending = np.linalg.norm(np.array(ending_location) - np.array(ending_loaction_passing))
        if distance_ending > 2 and loop_time !=1 :
            now_lane = [map_mid_x[passing_id[-1]],map_mid_y[passing_id[-1]]]
            now_lane = np.array(now_lane)
            now_lane = np.transpose(now_lane)
            global_path = np.append(global_path,now_lane,axis=0)
        if self.bad_end_point != []:
            direction_ = self.find_direction([global_path[-1,0]-global_path[-2,0]], [global_path[-1,1]-global_path[-2,1]])
            change_lane = self.change_lane_spline(global_path[-1,:], self.bad_end_point, direction_,direction_)
            change_lane = np.array(change_lane)
            change_lane = np.transpose(change_lane)
            global_path = np.append(global_path,change_lane,axis=0)
        
        return global_path 
    
    def draw_map(self, global_path, pointswant2see):
        map = self.map
        ego_pos = self.ego_state
        goal_id = [self.tar_id,self.tar_id_point]
        ego_pos_id = [self.id_start,self.id_start_point]
        map_mid_x = self.map_mid_x
        map_mid_y = self.map_mid_y

        map_mid = [[], []]
        map_left = [[], []]
        map_right = [[], []]
        path_x = []
        path_y = []

        # 原地图
        for i in map:
            for xx in range(len(i['mid'])):
                map_mid[0].append(i['mid'][xx][0])
                map_mid[1].append(i['mid'][xx][1])
            for xx in range(len(i['left'])):
                map_left[0].append(i['left'][xx][0])
                map_left[1].append(i['left'][xx][1])
            for xx in range(len(i['right'])):
                map_right[0].append(i['right'][xx][0])
                map_right[1].append(i['right'][xx][1])
        plt.scatter(map_mid[0], map_mid[1], color="C1", s=1)
        plt.scatter(map_left[0], map_left[1], color="C7", s=1)
        plt.scatter(map_right[0], map_right[1], color="C7", s=1)

        # 起始ID
        ego_pos_theta = ego_pos[3]
        lenth = ego_pos[2]
        dx = lenth * np.cos(ego_pos_theta)
        dy = lenth * np.sin(ego_pos_theta)
        plt.quiver(ego_pos[0], ego_pos[1], dx, dy, angles='xy', scale_units='xy', scale=1, color='r')
        plt.scatter(ego_pos[0], ego_pos[1], color="C0", s=1)
        plt.scatter(map_mid_x[ego_pos_id[0][0]], map_mid_y[ego_pos_id[0][0]], color="C4", s=1)
        plt.plot(map_mid_x[ego_pos_id[0][0]][0], map_mid_y[ego_pos_id[0][0]][0], "o", color="C4")
        
        
        # 目标ID
        for i in range(len(goal_id[0])):
            plt.scatter(map_mid_x[goal_id[0][i]], map_mid_y[goal_id[0][i]], color="C2", s=1)
            plt.plot(map_mid_x[goal_id[0][i]][-1], map_mid_y[goal_id[0][i]][-1], "o", color="C4")
        
        # 看最后的路径点
        if global_path != []:
            for i in global_path:
                path_x.append(i[0])
                path_y.append(i[1])
            plt.plot(path_x, path_y, color="C6")

        if pointswant2see != []:
            for i in pointswant2see:
                color_ = 'C'+str(i+5)
                plt.scatter(map_mid_x[i], map_mid_y[i], color=color_, s=10)
        
        plt.show(block=True)
        plt.show()
        aa = 1
