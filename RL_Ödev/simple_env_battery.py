import gym
from gym import spaces
import numpy as np
import random
import pickle


class SimpleWarehouseEnv(gym.Env):
    def __init__(self):
        super(SimpleWarehouseEnv, self).__init__()
        self.grid_size = (5, 5)
        self.goal_position = (0, 4)
        self.obstacles = [(1, 1), (3, 2)]
        self.possible_load_sets = [
            [(4, 0), (4, 1)],
            [(3, 3), (4, 4)],
            [(2, 0), (4, 2)]
        ]
        self.action_space = spaces.Discrete(6)  # Up, Down, Right, Left, Pickup, Drop
        self.observation_space = spaces.Tuple((
            spaces.Discrete(5),  # x
            spaces.Discrete(5),  # y
            spaces.Discrete(2)   # yük taşıyor mu (0-1)
        ))
        self.load_set_index = 0
        self.reset()

    def reset(self, load_set_index=None):
        self.position = (2, 2)
        self.carrying = 0
        self.done = False

        if load_set_index is not None:
            self.load_set_index = load_set_index
        self.loads = self.possible_load_sets[self.load_set_index]
        self.remaining_loads = self.loads.copy()
        return self._get_obs()

    def _get_obs(self):
        x, y = self.position
        return (x, y, self.carrying)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        x, y = self.position
        reward = -1  

        
        new_x, new_y = x, y
        if action == 0:  # Up
            new_x = max(0, x - 1)
        elif action == 1:  # Down
            new_x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # Right
            new_y = min(self.grid_size[1] - 1, y + 1)
        elif action == 3:  # Left
            new_y = max(0, y - 1)

        if (new_x, new_y) not in self.obstacles:
            self.position = (new_x, new_y)

        # Pickup
        if action == 4 and self.position in self.remaining_loads and self.carrying == 0:
            self.remaining_loads.remove(self.position)
            self.carrying = 1
            reward = 10

        # Drop
        if action == 5 and self.position == self.goal_position and self.carrying == 1:
            self.carrying = 0
            reward = 50
            if len(self.remaining_loads) == 0:
                self.done = True

        return self._get_obs(), reward, self.done, {}

    def render(self):
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        x, y = self.position
        grid[x][y] = 'R'
        for ox, oy in self.obstacles:
            grid[ox][oy] = '#'
        for lx, ly in self.remaining_loads:
            grid[lx][ly] = 'L'
        gx, gy = self.goal_position
        grid[gx][gy] = 'G'

        print("\n".join(" ".join(row) for row in grid))
        print(f"Konum: {self.position}, Yük: {self.carrying}")




env = SimpleWarehouseEnv()

# Kullanıcıdan yük seti seçimi
print("Yük setini seçin (0, 1, 2):")
chosen_load_set_index = int(input("Seçiminiz: "))

# Q tablosu
Q = np.zeros((5, 5, 2, 6))

# Parametreler
alpha = 0.1
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.05
decay = 0.99
episodes = 10000

# Eğitim
for episode in range(episodes):
    obs = env.reset(load_set_index=chosen_load_set_index)
    done = False
    step_count = 0

    while not done and step_count < 100:
        x, y, carrying = obs

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[x, y, carrying, :])

        next_obs, reward, done, _ = env.step(action)
        nx, ny, ncarry = next_obs

        Q[x, y, carrying, action] += alpha * (
            reward + gamma * np.max(Q[nx, ny, ncarry, :]) - Q[x, y, carrying, action]
        )

        obs = next_obs
        step_count += 1

    epsilon = max(min_epsilon, epsilon * decay)

print("Eğitim tamamlandı!")

# Q tablosunu kaydet
with open("simple_q_table_3.pkl", "wb") as f:
    pickle.dump(Q, f)



# Test
obs = env.reset(load_set_index=chosen_load_set_index)
done = False
total_reward = 0
step_count = 0
max_steps = 200

while not done and step_count < max_steps:
    x, y, carrying = obs
    action = np.argmax(Q[x, y, carrying, :])
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    step_count += 1
    # env.render()
    # print(f"Aksiyon: {action}, Ödül: {reward}")

if done:
    print("Görev başarıyla tamamlandı.")
else:
    print("Adım sınırına ulaşıldı, görev tamamlanamadı.")

print(f"Toplam Ödül: {total_reward}")

