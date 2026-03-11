# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from onsite import scenarioOrganizer, env
import sys
sys.path.append('.')
import pandas as pd
import numpy as np

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BipedalWalker-v3"
    # env_id: str = "Pendulum-v1"
    # env_id: str = "Hopper-v4"
    # env_id: str = "HumanoidStandup-v4"
    """the id of the environment"""
    total_timesteps: int = 3000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 500
    # num_steps: int = 2000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 5
    # num_minibatches: int = 30
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    # ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    # target_kl: float = None
    target_kl: float = 0.01
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    obs_space: int = 27
    action_space: int = 2
def observation_to_state1(observation, frame_n, action, goal, map_info):
    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()

    # 加入主车的状态v yaw
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])

    # # 将上一时刻的acc、steer_angle加入状态
    # if frame_n == 0:
    #     states.append(0)
    #     states.append(0)
    # else:
    #     states.append(action[0])
    #     states.append(action[1])

    # 加入目标区域中点的坐标
    states.extend([goal[0] - state[0, 0], goal[1] - state[0, 1]])

    # 加入车道线纵坐标(4个)
    states.extend([map_info[0]['left_bound'] - state[0, 1], map_info[0]['right_bound'] - state[0, 1]])
    states.append(map_info[1]['right_bound'] - state[0, 1])
    states.append(map_info[2]['right_bound'] - state[0, 1])

    # 加入偏移量
    offset = 0
    if state[0, 1] > map_info[0]['right_bound']:
        offset = map_info[0]['center'] - state[0, 1]
    elif map_info[1]['right_bound'] < state[0, 1] < map_info[1]['left_bound']:
        offset = map_info[1]['center'] - state[0, 1]
    elif state[0, 1] < map_info[2]['left_bound']:
        offset = map_info[2]['center'] - state[0, 1]
    states.append(offset)

    # 他车的状态，每个车道最多考虑2辆车，分别是主车前后的第一辆车辆，纵向距离200m以内，不足的补(200, dy, 0)
    # 初始化六个车道的车辆信息
    car_init = [[0, 0, 0] for _ in range(6)]
    car_init[0] = [-200, map_info[0]['center'] - state[0, 1], 0]
    car_init[1] = [200, map_info[0]['center'] - state[0, 1], 0]
    car_init[2] = [-200, map_info[1]['center'] - state[0, 1], 0]
    car_init[3] = [200, map_info[1]['center'] - state[0, 1], 0]
    car_init[4] = [-200, map_info[2]['center'] - state[0, 1], 0]
    car_init[5] = [200, map_info[2]['center'] - state[0, 1], 0]
    # 依次判断每个车属于哪个车道，如果距离小于200m，就更新车辆信息
    for i in range(1, len(state)):
        if map_info[0]['right_bound'] < state[i, 1] < map_info[0]['left_bound'] and 0 < state[i, 0] - state[0, 0] < car_init[1][0]:
            car_init[1] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info[0]['right_bound'] < state[i, 1] < map_info[0]['left_bound'] and 0 > state[i, 0] - state[0, 0] > car_init[0][0]:
            car_init[0] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info[1]['right_bound'] < state[i, 1] < map_info[1]['left_bound'] and 0 < state[i, 0] - state[0, 0] < car_init[3][0]:
            car_init[3] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info[1]['right_bound'] < state[i, 1] < map_info[1]['left_bound'] and 0 > state[i, 0] - state[0, 0] > car_init[2][0]:
            car_init[2] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info[2]['right_bound'] < state[i, 1] < map_info[2]['left_bound'] and 0 < state[i, 0] - state[0, 0] < car_init[5][0]:
            car_init[5] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info[2]['right_bound'] < state[i, 1] < map_info[2]['left_bound'] and 0 > state[i, 0] - state[0, 0] > car_init[4][0]:
            car_init[4] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
    # 添加车辆状态信息
    for i in range(6):
        states.extend(car_init[i])

    states = torch.Tensor(states).to(device).unsqueeze(0)

    return states
def observation_to_state2(observation, frame_n, action, goal, map_info):
    goal_neg = [-goal[0], -goal[1]]
    map_info_neg = {}
    for j in range(3):
        map_info_neg[j] = {
            'left_bound': -map_info[j]['left_bound'],
            'center': -map_info[j]['center'],
            'right_bound': -map_info[j]['right_bound']
        }

    frame = pd.DataFrame()
    states = []
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'a', 'yaw', 'length'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()
    state[:, 0] = -state[:, 0]
    state[:, 1] = -state[:, 1]
    # 0802 修正yaw: -state[:, 4] -> state[:, 4]
    state[:, 4] = state[:, 4] - np.pi

    # 加入主车的状态v yaw
    states.extend([state[0, 2], ((state[0, 4] + np.pi) % (2 * np.pi) - np.pi)])

    # # 将上一时刻的acc、steer_angle加入状态
    # if frame_n == 0:
    #     states.append(0)
    #     states.append(0)
    # else:
    #     states.append(action[0])
    #     states.append(action[1])

    # 加入目标区域中点的坐标
    states.extend([goal_neg[0] - state[0, 0], goal_neg[1] - state[0, 1]])

    # 加入车道线纵坐标(4个)
    states.extend([map_info_neg[0]['left_bound'] - state[0, 1], map_info_neg[0]['right_bound'] - state[0, 1]])
    states.append(map_info_neg[1]['right_bound'] - state[0, 1])
    states.append(map_info_neg[2]['right_bound'] - state[0, 1])

    # 加入偏移量
    offset = 0
    if state[0, 1] > map_info_neg[0]['right_bound']:
        offset = map_info_neg[0]['center'] - state[0, 1]
    elif map_info_neg[1]['right_bound'] < state[0, 1] < map_info_neg[1]['left_bound']:
        offset = map_info_neg[1]['center'] - state[0, 1]
    elif state[0, 1] < map_info_neg[2]['left_bound']:
        offset = map_info_neg[2]['center'] - state[0, 1]
    states.append(offset)

    # 他车的状态，每个车道最多考虑2辆车，分别是主车前后的第一辆车辆，纵向距离200m以内，不足的补(200, dy, 0)
    # 初始化六个车道的车辆信息
    car_init = [[0, 0, 0] for _ in range(6)]
    car_init[0] = [-200, map_info_neg[0]['center'] - state[0, 1], 0]
    car_init[1] = [200, map_info_neg[0]['center'] - state[0, 1], 0]
    car_init[2] = [-200, map_info_neg[1]['center'] - state[0, 1], 0]
    car_init[3] = [200, map_info_neg[1]['center'] - state[0, 1], 0]
    car_init[4] = [-200, map_info_neg[2]['center'] - state[0, 1], 0]
    car_init[5] = [200, map_info_neg[2]['center'] - state[0, 1], 0]
    # 依次判断每个车属于哪个车道，如果距离小于200m，就更新车辆信息
    for i in range(1, len(state)):
        if map_info_neg[0]['right_bound'] < state[i, 1] < map_info_neg[0]['left_bound'] and 0 < state[i, 0] - state[0, 0] < car_init[1][0]:
            car_init[1] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info_neg[0]['right_bound'] < state[i, 1] < map_info_neg[0]['left_bound'] and 0 > state[i, 0] - state[0, 0] > car_init[0][0]:
            car_init[0] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info_neg[1]['right_bound'] < state[i, 1] < map_info_neg[1]['left_bound'] and 0 < state[i, 0] - state[0, 0] < car_init[3][0]:
            car_init[3] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info_neg[1]['right_bound'] < state[i, 1] < map_info_neg[1]['left_bound'] and 0 > state[i, 0] - state[0, 0] > car_init[2][0]:
            car_init[2] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info_neg[2]['right_bound'] < state[i, 1] < map_info_neg[2]['left_bound'] and 0 < state[i, 0] - state[0, 0] < car_init[5][0]:
            car_init[5] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
        if map_info_neg[2]['right_bound'] < state[i, 1] < map_info_neg[2]['left_bound'] and 0 > state[i, 0] - state[0, 0] > car_init[4][0]:
            car_init[4] = [state[i, 0] - state[0, 0], state[i, 1] - state[0, 1], state[i, 2] - state[0, 2]]
    # 添加车辆状态信息
    for i in range(6):
        states.extend(car_init[i])

    states = torch.Tensor(states).to(device).unsqueeze(0)

    return states
def get_lane_info1(road_info):
    map_info = {}
    map_info[0] = {"left_bound":road_info.discretelanes[0].left_vertices[0, 1],
                   "center":road_info.discretelanes[0].center_vertices[0, 1],
                   "right_bound":road_info.discretelanes[0].right_vertices[0, 1]}
    map_info[1] = {"left_bound": road_info.discretelanes[1].left_vertices[0, 1],
                   "center": road_info.discretelanes[1].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[1].right_vertices[0, 1]}
    map_info[2] = {"left_bound": road_info.discretelanes[2].left_vertices[0, 1],
                   "center": road_info.discretelanes[2].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[2].right_vertices[0, 1]}
    return map_info
def get_lane_info2(road_info):
    map_info = {}
    map_info[0] = {"left_bound":road_info.discretelanes[3].left_vertices[0, 1],
                   "center":road_info.discretelanes[3].center_vertices[0, 1],
                   "right_bound":road_info.discretelanes[3].right_vertices[0, 1]}
    map_info[1] = {"left_bound": road_info.discretelanes[4].left_vertices[0, 1],
                   "center": road_info.discretelanes[4].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[4].right_vertices[0, 1]}
    map_info[2] = {"left_bound": road_info.discretelanes[5].left_vertices[0, 1],
                   "center": road_info.discretelanes[5].center_vertices[0, 1],
                   "right_bound": road_info.discretelanes[5].right_vertices[0, 1]}
    return map_info

def cal_reward(state, action, next_state, end):
    # state和next_state转成numpy
    state = state.cpu().numpy().flatten()
    next_state = next_state.cpu().numpy().flatten()

    total_reward = 0

    # 加速度和转角的变化
    total_reward += -0.0002 * np.abs(action[0]) - 0.02 * np.abs(next_state[1] - state[1]) / 0.04

    # 速度
    if next_state[0] < 10 or next_state[0] > 40:
        total_reward += -10
    else:
        total_reward += (1 * next_state[0] / 30) ** 2

    # 航向角
    if abs(next_state[1]) > 0.5:
        total_reward += -10

    # 碰撞
    if end == 2:
        total_reward += -50

    # 到达终点
    if next_state[8] < next_state[6] < next_state[7]:
        if next_state[5] - 5 < 0 and next_state[7] > 0 and next_state[8] < 0:
            total_reward += 100
    if next_state[9] < next_state[6] < next_state[8]:
        if next_state[5] - 5 < 0 and next_state[8] > 0 and next_state[9] < 0:
            total_reward += 100
    if next_state[10] < next_state[6] < next_state[9]:
        if next_state[5] - 5 < 0 and next_state[9] > 0 and next_state[10] < 0:
            total_reward += 100

    # total_reward = 1

    return total_reward
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, Args):
        super().__init__()
        hidden_size = 256
        self.critic = nn.Sequential(
            layer_init(nn.Linear(Args.obs_space, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(Args.obs_space, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, Args.action_space), std=0.01),
            nn.Tanh(),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, Args.action_space))

        # acc_std = math.log(0.3)  # 第一个维度的初始值
        # steer_std = math.log(0.02)  # 第二个维度的初始值
        # initial_std = torch.tensor([[acc_std, steer_std]])
        # # 在你的网络定义中，将 actor_logstd 初始化为 initial_values
        # self.actor_logstd = nn.Parameter(initial_std)

        # 固定方差
        self.actor_logstd = torch.tensor([[math.log(0.3), math.log(0.02)]]).to(device)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        # print(x)
        # print(action_mean)
        # 动作映射
        action_mean = action_mean * torch.tensor([10, 0.15]).to(device)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 256
        self.discriminator = nn.Sequential(
            layer_init(nn.Linear(args.obs_space + args.action_space, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
            nn.Sigmoid()
)

# 指定输入输出文件夹位置
input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../inputs/inputs_gail'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/outputs_gail'))
# 实例化场景管理模块（ScenairoOrganizer）和场景测试模块（Env）
so = scenarioOrganizer.ScenarioOrganizer()
envs = env.Env()
so.load(input_dir, output_dir)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    agent = Agent(args).to(device)

    # # 从model.pth文件中加载模型参数初始化模型actor_mean的参数
    # saved_model_parameters = torch.load('model_best_new_hid256_450e.pth')
    # # 提取并调整 actor_mean 的参数键名
    # actor_mean_parameters = {}
    # for k, v in saved_model_parameters.items():
    #     if 'fc1' in k:
    #         new_key = k.replace('fc1', '0')
    #     elif 'fc2' in k:
    #         new_key = k.replace('fc2', '2')
    #     elif 'fc3' in k:
    #         new_key = k.replace('fc3', '4')
    #     else:
    #         continue
    #     actor_mean_parameters[new_key] = v
    # # 加载参数到模型中
    # agent.actor_mean.load_state_dict(actor_mean_parameters)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 鉴别器
    D = D().to(device)
    optimizer_d = optim.Adam(D.parameters(), lr=3e-4)
    # optimizer_d = optim.Adam(D.parameters())

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (args.obs_space,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (args.action_space,)).to(device)
    # 0806新增
    actions_clip = torch.zeros((args.num_steps, args.num_envs) + (args.action_space,)).to(device)

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # 失败截断(无价值)
    terminals = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    frame_n = 0
    ep_reward = 0
    action = [0, 0]
    scenario_to_test = so.next()
    scenario_to_test['test_settings']['visualize'] = False
    observation, traj = envs.make(scenario=scenario_to_test)
    goal = [np.mean(observation['test_setting']['goal']['x']), np.mean(observation['test_setting']['goal']['y'])]
    road_info = envs.controller.observation.road_info
    init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']]
    if goal[0] > init_pos[0]:
        # 从左往右
        map_info = get_lane_info1(road_info)
    else:
        # 从右往左
        map_info = get_lane_info2(road_info)
    start_time = time.time()

    if goal[0] > init_pos[0]:
        next_obs = observation_to_state1(observation, frame_n, action, goal, map_info)
    else:
        next_obs = observation_to_state2(observation, frame_n, action, goal, map_info)

    next_done = torch.zeros(args.num_envs).to(device)
    # 失败截断(无价值)
    terminal = torch.zeros(args.num_envs).to(device)

    # for iteration in range(1, args.num_iterations + 1):
    for iteration in range(1, 3000):
        print(f"回合：{iteration}，耗时：{time.time() - start_time:.2f}")
        start_time = time.time()

        # 读取专家轨迹
        expert_traj = np.load('2016lanechanging67.npy')
        # 删除第二三列
        expert_traj = np.delete(expert_traj, [2, 3], axis=1)
        # 读取专家轨迹的观测和动作并转换为tensor
        expert_obs = torch.Tensor(expert_traj[:, :27]).to(device)
        expert_act = expert_traj[:, 27:]
        expert_act[:, 0] = np.clip(expert_act[:, 0], -10, 10)
        expert_act[:, 1] = np.clip(expert_act[:, 1], -0.15, 0.15)
        expert_act = torch.Tensor(expert_act).to(device)

        # 数据增强，在原来的数据基础上复制一份同样的专家轨迹
        expert_obs = torch.cat([expert_obs, expert_obs], dim=0)
        expert_act = torch.cat([expert_act, expert_act], dim=0)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # 失败截断(无价值)
            terminals[step] = terminal

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            action = action.cpu().numpy().reshape(-1)
            # 将动作进行限制，第一个动作[-10,10]，第二个动作[-0.13,0.13] lattice 0.7 dp 0.349
            action_clip = [np.clip(action[0], -10, 10), np.clip(action[1], -0.15, 0.15)]
            observation = envs.step(action_clip)
            actions_clip[step] = torch.tensor(action_clip).to(device).view(-1)
            frame_n += 1
            if goal[0] > init_pos[0]:
                next_obs = observation_to_state1(observation, frame_n, action, goal, map_info)
            else:
                next_obs = observation_to_state2(observation, frame_n, action, goal, map_info)
            # reward = cal_reward(obs[step], action_clip, next_obs, observation['test_setting']['end'])
            # ep_reward += reward
            # ln(discriminator(sf, af))作为reward
            # reward = -torch.log(agent.discriminator(torch.cat([obs[step], actions_clip[step]], dim=1))).item()
            reward = -torch.log(D.discriminator(torch.cat([obs[step], actions[step]], dim=1))).item()

            terminations = np.zeros(args.num_envs)
            truncations = np.zeros(args.num_envs)
            next_obs_np = next_obs.cpu().numpy().flatten()
            # 碰撞
            if observation['test_setting']['end'] == 2:
                terminations = np.ones(args.num_envs)
            # 超出车道线
            if next_obs_np[4] < 0 or next_obs_np[7] > 0:
                terminations = np.ones(args.num_envs)
            # 到达终点的x
            if next_obs_np[2] - 5 < 0:
                terminations = np.ones(args.num_envs)
            # 未超过终点x但到了最大步数
            if observation['test_setting']['end'] == 1 and next_obs_np[2] - 5 > 0:
                truncations = np.ones(args.num_envs)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)
            # 失败截断(无价值)
            terminal = torch.Tensor(terminations).to(device)
            if next_done.cpu().numpy().flatten()[0] == 1:
                # print(f"回合：{iteration}，奖励：{ep_reward:.2f}")
                # 关闭当前matplotlib窗口
                plt.close()
                so.add_result(scenario_to_test, observation['test_setting']['end'])
                frame_n = 0
                action = [0, 0]
                so.load(input_dir, output_dir)
                scenario_to_test = so.next()
                # 每10个回合可视化一次
                if iteration % 50 == 0:
                    scenario_to_test['test_settings']['visualize'] = True
                else:
                    scenario_to_test['test_settings']['visualize'] = False
                observation, traj = envs.make(scenario=scenario_to_test)
                goal = [np.mean(observation['test_setting']['goal']['x']),
                        np.mean(observation['test_setting']['goal']['y'])]
                road_info = envs.controller.observation.road_info
                init_pos = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y']]
                if goal[0] > init_pos[0]:
                    # 从左往右
                    map_info = get_lane_info1(road_info)
                else:
                    # 从右往左
                    map_info = get_lane_info2(road_info)
                if goal[0] > init_pos[0]:
                    next_obs = observation_to_state1(observation, frame_n, action, goal, map_info)
                else:
                    next_obs = observation_to_state2(observation, frame_n, action, goal, map_info)
                next_done = torch.zeros(args.num_envs).to(device)
                # 失败截断(无价值)
                terminal = torch.zeros(args.num_envs).to(device)
                # 写入奖励
                ep_reward = 0

        # 更新discriminator,专家轨迹(0)和生成轨迹(1)的交叉熵
        expert_logit = D.discriminator(torch.cat([expert_obs, expert_act], dim=1))
        expert_loss = torch.nn.functional.binary_cross_entropy(expert_logit, torch.zeros_like(expert_logit))
        generated_logit = D.discriminator(torch.cat([obs.view(-1, args.obs_space), actions_clip.view(-1, args.action_space)], dim=1))
        generated_loss = torch.nn.functional.binary_cross_entropy(generated_logit, torch.ones_like(generated_logit))
        discriminator_loss = expert_loss + generated_loss
        optimizer_d.zero_grad()
        discriminator_loss.backward()
        optimizer_d.step()
        # 打印expert_logit和generated_logit的数值
        print(f"expert_logit: {expert_logit.mean().item()}, generated_logit: {generated_logit.mean().item()}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    # 是否为失败截断
                    nextnonterminal_dw = 1.0 - terminal

                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    # 是否为失败截断
                    nextnonterminal_dw = 1.0 - terminals[t + 1]

                # 人为截断需要计算价值，失败截断价值为0
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal_dw - values[t]
                # 人为截断也要重新计算
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (args.obs_space,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (args.action_space,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 打印self.actor_logstd的数值
        actor_logstd_numpy = torch.exp(agent.actor_logstd[0]).detach().cpu().numpy()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 打印训练轮次
        print(f"------------------iter {iteration}------------------")

    # 保存模型
    torch.save(agent.actor_mean.state_dict(), 'actor_mean.pth')
    # 打印agent.actor_logstd
    print(f"actor_logstd: {actor_logstd_numpy}")
