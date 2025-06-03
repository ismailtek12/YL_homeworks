import gym
from gym import spaces
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


class WarehouseEnv(gym.Env):
    def __init__(self):
        super(WarehouseEnv, self).__init__()
        self.grid_size = (7, 7)
        self.goal_position = (5, 3)
        self.charge_station = (1, 5)
        self.obstacles = [(2, 2), (4, 4), (5, 1)]
        self.initial_loads = [(6, 6), (5, 5)]
        self.max_battery = 50
        self.max_load = 2

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size[0]),
            spaces.Discrete(self.grid_size[1]),
            spaces.Discrete(self.max_load + 1),
            spaces.Discrete(self.max_battery + 1)
        ))

        self.reset()

    def reset(self):
        self.position = (0, 0)
        self.carrying_load = 0
        self.battery = self.max_battery
        self.loads = self.initial_loads.copy()
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        x, y = self.position
        return (x, y, self.carrying_load, self.battery)



        
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        x, y = self.position
        next_x, next_y = x, y
        reward = -1

        if action == 0:  # Up
            next_x = max(0, x - 1)
        elif action == 1:  # Down
            next_x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # Right
            next_y = min(self.grid_size[1] - 1, y + 1)
        elif action == 3:  # Left
            next_y = max(0, y - 1)

        if (next_x, next_y) in self.obstacles:
            next_x, next_y = x, y

        self.position = (next_x, next_y)

        if action == 4:  # Pickup
            if self.position in self.loads and self.carrying_load < self.max_load:
                self.loads.remove(self.position)
                self.carrying_load += 1
                reward = 10

        elif action == 5:  # Drop
            if self.position == self.goal_position and self.carrying_load > 0:
                self.carrying_load -= 1
                reward = 50

        elif action == 6:  # Recharge
            if self.position == self.charge_station and self.battery < self.max_battery:
                self.battery = self.max_battery
                reward = 5
                return self._get_obs(), reward, self.done, {}

        if action < 4:
            self.battery -= 1

        if self.battery <= 0:
            self.done = True
            reward = -50

        if self.carrying_load == 0 and len(self.loads) == 0:
            self.done = True

        return self._get_obs(), reward, self.done, {}


    """def render(self, mode='human'):
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        x, y = self.position
        grid[x][y] = 'R'
        for ox, oy in self.obstacles:
            grid[ox][oy] = '#'
        for lx, ly in self.loads:
            grid[lx][ly] = 'L'
        gx, gy = self.goal_position
        grid[gx][gy] = 'G'
        csx, csy = self.charge_station
        grid[csx][csy] = 'C'

        print("\n".join(" ".join(row) for row in grid))
        print(f"Konum: {self.position}, Yük: {self.carrying_load}, Batarya: {self.battery}")"""

        


# Q-learning
env = WarehouseEnv()

Q = np.zeros((10, 10, 2, 101, 7))  # (x, y, load, battery, action)

alpha = 0.1
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.1
decay = 0.995
episodes = 1000

for episode in range(episodes):
    print(f"Episode {episode//10}")
    obs = env.reset()
    done = False

    while not done:
        x, y, load, battery = obs
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[x, y, load, battery, :])

        next_obs, reward, done, _ = env.step(action)
        nx, ny, nload, nbattery = next_obs

        Q[x, y, load, battery, action] += alpha * (
            reward + gamma * np.max(Q[nx, ny, nload, nbattery, :]) - Q[x, y, load, battery, action]
        )

        obs = next_obs

    epsilon = max(min_epsilon, epsilon - decay)

print("Eğitim tamamlandı.")
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

# === Test Etme ===
obs = env.reset()
done = False
total_reward = 0

while not done:
   
   
   
   
   x, y, load, battery = obs
   action = np.argmax(Q[x, y, load, battery, :])
   obs, reward, done, _ = env.step(action)
   env.render()
   print(f"Aksiyon: {action}, Ödül: {reward}\n")
   total_reward += reward
    
   
   
    

print(f"Toplam Ödül: {total_reward}")
