import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class SimpleWarehouseEnv(gym.Env):
    def __init__(self):
        super(SimpleWarehouseEnv, self).__init__()
        self.grid_size = (5, 5)
        self.goal_position = (4, 4)
        self.charge_station = (0, 4)
        self.obstacles = [(2, 2)]
        self.initial_loads = [(3, 3)]
        self.max_battery = 50
        self.max_load = 1
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(7,), dtype=np.int32)
        self.max_steps = 200
        self.reset()

    def reset(self):
        self.position = (0, 0)
        self.carrying_load = 0
        self.battery = self.max_battery
        self.loads = self.initial_loads.copy()
        self.done = False
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        x, y = self.position
        load_dist = abs(self.loads[0][0] - x) + abs(self.loads[0][1] - y) if self.loads else 0
        goal_dist = abs(self.goal_position[0] - x) + abs(self.goal_position[1] - y)
        charge_dist = abs(self.charge_station[0] - x) + abs(self.charge_station[1] - y)
        return np.array([x, y, self.carrying_load, self.battery, load_dist, goal_dist, charge_dist], dtype=np.int32)

    def step(self, action):
        if self.done: return self._get_obs(), 0, self.done, {}
        self.step_count += 1
        x, y = self.position
        next_x, next_y = x, y
        reward = -0.05

        if action == 0: next_x = max(0, x - 1)
        elif action == 1: next_x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2: next_y = min(self.grid_size[1] - 1, y + 1)
        elif action == 3: next_y = max(0, y - 1)

        if (next_x, next_y) in self.obstacles:
            reward -= 1
            next_x, next_y = x, y

        self.position = (next_x, next_y)

        if action == 4 and self.position in self.loads and self.carrying_load < self.max_load:
            self.loads.remove(self.position)
            self.carrying_load += 1
            reward = 30

        elif action == 5 and self.position == self.goal_position and self.carrying_load > 0:
            self.carrying_load -= 1
            reward = 100
            if not self.loads and self.carrying_load == 0:
                self.done = True
                reward += 200

        elif action == 6 and self.position == self.charge_station:
            gain = self.max_battery - self.battery
            self.battery = self.max_battery
            reward = 5 + 0.5 * gain

        elif action in [4, 5, 6]:
            reward -= 10

        if action in [0, 1, 2, 3]: self.battery -= 1
        if self.battery <= 0:
            self.done = True
            reward = -200
        if self.step_count >= self.max_steps:
            self.done = True
            reward -= 100

        return self._get_obs(), reward, self.done, {}

#  DQN Model 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    

def visualize_simulation():
    env = SimpleWarehouseEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(7, 7).to(device)
    model.load_state_dict(torch.load("dqn_model.pth", map_location=device))
    model.eval()
    state = env.reset()
    done = False

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.grid_size[1] - 0.5)
    ax.set_ylim(-0.5, env.grid_size[0] - 0.5)
    ax.set_xticks(range(env.grid_size[1]))
    ax.set_yticks(range(env.grid_size[0]))
    ax.grid(True)

    # Harita öğeleri
    for (ox, oy) in env.obstacles:
        ax.add_patch(patches.Rectangle((oy - 0.5, ox - 0.5), 1, 1, color="black"))
    gx, gy = env.goal_position
    ax.add_patch(patches.Rectangle((gy - 0.5, gx - 0.5), 1, 1, color="green", alpha=0.5))
    cx, cy = env.charge_station
    ax.add_patch(patches.Rectangle((cy - 0.5, cx - 0.5), 1, 1, color="blue", alpha=0.5))

    # Yük ve robot çizimi
    load_patches = []
    robot_patch = patches.Rectangle((-0.5, -0.5), 1, 1, linewidth=2, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(robot_patch)

    plt.ion()
    plt.show()
    time.sleep(3)
    while not done:
        with torch.no_grad():
            action = model(torch.tensor(state, dtype=torch.float32).to(device)).argmax().item()
        state, reward, done, _ = env.step(action)

        # Robot pozisyonunu güncelle
        rx, ry = env.position
        robot_patch.set_xy((ry - 0.5, rx - 0.5))

        # Yükleri güncelle
        for patch in load_patches: patch.remove()
        load_patches.clear()
        for (lx, ly) in env.loads:
            rect = patches.Rectangle((ly - 0.5, lx - 0.5), 1, 1, color="red", alpha=0.5)
            load_patches.append(rect)
            ax.add_patch(rect)

        plt.title(f"Aksiyon: {action}, Batarya: {env.battery}")
        plt.pause(0.3)

        #Yük başarıyla bırakıldıysa simülasyonu sonlandır
        if done and not env.loads and env.carrying_load == 0:
            print(" Yük başarıyla bırakıldı. Simülasyon sonlandırıldı.")
            plt.ioff()
            break
            

    plt.ioff()
    plt.show()


if __name__=="__main__":
    
    visualize_simulation()