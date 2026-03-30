import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Planner import Lattice_Planner
class Vehicle(object):
    # 创建一个直道上的车辆
    def __init__(self, x, y, heading, speed):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = speed
        self.position = np.array([x, y], dtype=np.float64)
        self.action = {'steering': 0, 'acceleration': 0}
        self.LENGTH = 5
        self.MAX_VELOCITY = 40
        self.MIN_VELOCITY = 20
        self.dt = 0.025
        self.WIDTH = 2.5
        self.lane = self.get_lane_index()

    def step(self):
        if self.velocity > self.MAX_VELOCITY:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_VELOCITY - self.velocity))
        elif self.velocity < self.MIN_VELOCITY:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_VELOCITY - self.velocity))
        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * self.dt
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * self.dt
        self.velocity += self.action['acceleration'] * self.dt
        if self.velocity > self.MAX_VELOCITY:
            self.velocity = self.MAX_VELOCITY
        elif self.velocity < self.MIN_VELOCITY:
            self.velocity = self.MIN_VELOCITY
        # print(self.heading)

    def get_lane_index(self):
        return (self.y+2) // 4

class EgoVehicle(Vehicle):
    def __init__(self, x, y, heading, speed, vehicles):
        super(EgoVehicle, self).__init__(x, y, heading, speed)
        self.planner = Lattice_Planner(0.025)
        self.obs = vehicles

    def _get_veh_obs(self):
        return self.obs
    def get_action(self):
        action = self.planner.exec_waypoint_nav_demo_em_stitch_sn(self)
        self.action['steering'] = action[1]
        self.action['acceleration'] = action[0]

    def obs_update(self, road):
        self.obs = road.vehicles[1:]

    def step(self):
        self.get_action()
        super().step()


class Road(object):
    def __init__(self, num_lane, vehicles=None, render=False, road_length=300):
        self.vehicles = vehicles or []
        self.road_length = road_length
        self.num_lane = num_lane - 1
        self.render = render
        self.count = 0
        if self.render:
            self.fig, self.ax = plt.subplots()
            plt.ion()

    def plot_road(self):
        # self.ax.clear()
        if self.count % 8 == 0:
            for vehicle in self.vehicles:
                pos = (vehicle.position[0] - vehicle.LENGTH / 2, vehicle.position[1] - vehicle.WIDTH / 2)  # 矩形右下角坐标位置
                rect = patches.Rectangle(pos, vehicle.LENGTH, vehicle.WIDTH, linewidth=1, edgecolor='black', facecolor='none', angle=np.rad2deg(vehicle.heading))
                self.ax.add_patch(rect)
            self.ax.set_aspect('equal')
            x = np.linspace(0, self.road_length, int(self.road_length/0.5+1))
            plt.plot(x, np.ones_like(x) * 0 - 2, linewidth=0.75, color='black')
            plt.plot(x, np.ones_like(x) * self.num_lane * 4 + 2, linewidth=0.75, color='black')    # , linestyle='--'
            for lane in range(self.num_lane):
                plt.plot(x, np.ones_like(x) * lane * 4 + 2, linewidth=0.75, color='black', linestyle='--')
        # for i, path in enumerate(self.vehicles[0].planner.paths):
        #     if i == self.vehicles[0].planner.best_index:
        #         self.ax.plot(path[0, :], path[1, :], color='red', linewidth=1.5, alpha=0.5)
        #     else:
        #         self.ax.plot(path[0, :], path[1, :], color='green', linewidth=1.5, alpha=0.5)

        plt.draw()
        plt.pause(0.01)
        self.count += 1

    def step(self):
        for vehicle in self.vehicles:
            # if vehicle == self.vehicles[0]:   # 更新自车的视野
            #     vehicle.obs_update(self)
            vehicle.step()
            if self.render is True:
                self.plot_road()




# class Lane(object):
#     def __init__(self, num_lane, vehicles=None):