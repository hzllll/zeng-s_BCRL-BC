from Vehicle import Vehicle, Road, EgoVehicle
import numpy as np
import EM_Planner
from vehicle_control import PID
import matplotlib.pyplot as plt
import osqp
from scipy import sparse
from scipy import io

# v2 = Vehicle(x=50, y=0, heading=0, speed=20)
v3 = Vehicle(x=40, y=4, heading=0, speed=23)
v4 = Vehicle(x=20, y=8, heading=0, speed=25)
v5 = Vehicle(x=21, y=0, heading=0, speed=21)
ego = EgoVehicle(x=20, y=4, heading=0, speed=25, vehicles=[v3, v4, v5])
road = Road(3, vehicles=[ego, v3, v4, v5], render=True, road_length=200)
time = int(6 / 0.025)
x = np.linspace(0, 300, int(300 / 0.1) + 1)
y = np.ones_like(x) * 4
heading = np.zeros_like(x)
waypoints = np.vstack((x, y, heading)).T
waypoints_kappa = np.zeros_like(x)
ego.planner.waypoints = waypoints
ego.planner.waypoint_kappa = waypoints_kappa

x_array = []
y_array = []
heading_array = []
velocity_array = []
t_array = []
acc_array = []
steer_array = []


for i in range(time):
    t = 0.025 * i
    t_array.append(t)
    x = []
    y = []
    heading = []
    velocity = []
    acc = []
    steer = []
    for v in road.vehicles:
        x.append(v.position[0])
        y.append(v.position[1])
        heading.append(v.heading)
        velocity.append(v.velocity)
        acc.append(v.action['acceleration'])
        steer.append(v.action['steering'])
    x_array.append(x)
    y_array.append(y)
    heading_array.append(heading)
    velocity_array.append(velocity)
    road.step()
    acc_array.append(acc)
    steer_array.append(steer)
    print(t)
    if 0.4 == round(t, 2):
        io.savemat("path_4.mat",
                   {"path": road.vehicles[0].planner.paths,
                    "best_index": road.vehicles[0].planner.best_index,
                    "x_array": np.array(x_array),
                    "y_array": np.array(y_array),
                    "heading_array": np.array(heading_array),
                    "velocity_array": np.array(velocity_array),
                    "t_array": np.array(t_array)})
    if 1.0 == round(t, 2):
        io.savemat("path_10.mat",
                   {"path": road.vehicles[0].planner.paths,
                    "best_index": road.vehicles[0].planner.best_index,
                    "x_array": np.array(x_array),
                    "y_array": np.array(y_array),
                    "heading_array": np.array(heading_array),
                    "velocity_array": np.array(velocity_array),
                    "t_array": np.array(t_array)})

    if 4.0 == round(t, 2):
        io.savemat("path_50.mat",
                   {"path": road.vehicles[0].planner.paths,
                    "best_index": road.vehicles[0].planner.best_index,
                    "x_array": np.array(x_array),
                    "y_array": np.array(y_array),
                    "heading_array": np.array(heading_array),
                    "velocity_array": np.array(velocity_array),
                    "t_array": np.array(t_array)})

io.savemat('vehicle.mat',{
    "x_array": np.array(x_array),
    "y_array": np.array(y_array),
    "heading_array": np.array(heading_array),
    "velocity_array": np.array(velocity_array),
    "t_array": np.array(t_array),
    "acc_array":np.array(acc_array),
    'steer_array':np.array(steer_array)})
plt.close()
plt.close()
plt.close()
plt.ioff()
# plt.plot(np.array(t_array), np.array(road.vehicles[0].planner.target_acc))
plt.figure()
plt.plot(np.array(t_array), np.array(velocity_array)[:, 0])
plt.show()
print('-')


