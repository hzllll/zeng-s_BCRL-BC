from Vehicle import Vehicle, Road, EgoVehicle
import numpy as np
import EM_Planner
from vehicle_control import PID
import matplotlib.pyplot as plt
import osqp
from scipy import sparse
from scipy import io

ego = Vehicle(x=50, y=2, heading=0, speed=25)
v2 = Vehicle(x=50, y=0, heading=0, speed=20)
v3 = Vehicle(x=100, y=4, heading=0, speed=30)
v4 = Vehicle(x=50, y=8, heading=0, speed=25)
vehicle = [ego]
road = Road(3, vehicles=vehicle, render=False, road_length=200)
time = int(12 / 0.025)
x = np.linspace(0, 300, int(300 / 0.1) + 1)
y = np.ones_like(x) * 0
heading = np.zeros_like(x)
waypoints = np.vstack((x, y, heading))


# startPoint = [0, 50]
# endPoint = [4, 160]
# coef = qiuntic_coef_(startPoint, endPoint, 26, 25)[:, None]
t = np.linspace(0, 12, int(12 / 0.025) + 1)
# s = np.sum(coef * np.vstack((t**5, t**4, t**3, t**2, t, np.ones_like(t))), axis=0)
# v = np.sum(coef * np.vstack((5 * t**4, 4 * t**3, 3 * t**2, t*2, np.ones_like(t), np.zeros_like(t))), axis=0)
# a = np.sum(coef * np.vstack((20 * t**3, 12 * t**2, 6 * t, 2 * np.ones_like(t), np.zeros_like(t), np.zeros_like(t))), axis=0)

def pp_controller(state):
    l = 5
    pre_x = 35.0
    alpha = np.arctan2(0 - state[1], pre_x) - state[2]
    ld = np.sqrt(pre_x ** 2 + (0 - state[1]) ** 2)
    u = np.arctan(2 * l * np.sin(alpha) / ld)
    return np.clip(u, -np.pi / 4, np.pi / 4)

s_pid = PID(KP=1.0, KI=0.001, KD=0.5)
v_pid = PID(KP=1.0, KI=0.001, KD=0.5)
target_s_list = []
target_v_list = []
target_a_list = []
s_car = []
v_car = []
a_car = []

for i in range(time+1):
    # vehicle[0].action['steering'] = pp_controller([vehicle[0].position[0], vehicle[0].position[1], vehicle[0].heading])
    # print(np.rad2deg(vehicle[0].action['steering']) * 17)
    ts = i * 0.025
    target_s = 0.1 * ts ** 3 / 3 + 51 + 24 * ts
    target_s_list.append(target_s)
    err_s = target_s - vehicle[0].position[0]
    ds = s_pid.output_cal(err_s)
    target_v = 0.1 * ts ** 2 + 24
    target_v_list.append(target_v)
    err_v = target_v - vehicle[0].velocity + ds
    dv = v_pid.output_cal(err_v)
    target_a = 0.2 * ts
    target_a_list.append(target_a)
    vehicle[0].action['acceleration'] = dv + 0.2 * ts
    s_car.append(vehicle[0].position[0])
    v_car.append(vehicle[0].velocity)
    a_car.append(vehicle[0].action['acceleration'])
    road.step()

io.savemat('pid_data.mat', {'target_v': np.array(target_v_list),
                            'v_car': np.array(v_car),
                            'target_s': np.array(target_s_list),
                            's_car': np.array(s_car)})
plt.figure()
plt.plot(t, np.array(target_v_list))
plt.plot(t, np.array(v_car))

plt.figure()
plt.plot(t, np.array(target_s_list))
plt.plot(t, np.array(s_car))

plt.figure()
plt.plot(t, np.array(target_a_list))
plt.plot(t, np.array(a_car))
plt.show()
#
#
#
print('------')


