import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import pickle
import numpy as np
import random
import gym
from gym import spaces


#Yük kombinasyonunu seçme
load_choices=int(input("Hangi yük kombinasyonunu seçmek istiyorsunuz? (0,1,2):"))


# Q tablosu yükleme
if load_choices==0:
    with open(f"simple_q_table.pkl", "rb") as f:
        Q = pickle.load(f)
else:

    with open(f"simple_q_table_{load_choices+1}.pkl", "rb") as f:
        Q = pickle.load(f)


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
            spaces.Discrete(5),     # x
            spaces.Discrete(5),     # y
            spaces.Discrete(2)      # yük taşıyor mu
        ))

        self.reset()

    def reset(self, load_set_index=None):
        self.position = (2, 2)
        self.carrying = 0
        self.done = False
        if load_set_index is not None and 0 <= load_set_index < len(self.possible_load_sets):
            self.loads = self.possible_load_sets[load_set_index]
        else:
            self.loads = random.choice(self.possible_load_sets)
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

        # Hareket
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

            # Eğer tüm yükler teslim edildiyse, bitir
            if len(self.remaining_loads) == 0:
                self.done = True

        # Bitti mi kontrolü
        if self.carrying == 0 and len(self.remaining_loads) == 0:
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
chosen_load_set_index = load_choices 
obs = env.reset(load_set_index=chosen_load_set_index)
done = False
steps = []
total_reward = 0
step_limit = 100

while not done and len(steps) < step_limit:
    x, y, carrying = obs
    action = np.argmax(Q[x, y, carrying, :])
    next_obs, reward, done, _ = env.step(action)
    steps.append((env.position, env.carrying, action))
    obs = next_obs
    total_reward += reward

print(f"Toplam adım: {len(steps)}, Toplam ödül: {total_reward}")



fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

robot_icon = plt.Circle((0, 0), 0.3, color='blue')
ax.add_patch(robot_icon)


for ox, oy in env.obstacles:
    ax.add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1, color='black'))

gx, gy = env.goal_position
ax.add_patch(plt.Rectangle((gy - 0.5, gx - 0.5), 1, 1, color='green', alpha=0.5))


load_patches = []
for lx, ly in env.possible_load_sets[chosen_load_set_index]:
    patch = plt.Rectangle((ly - 0.5, lx - 0.5), 1, 1, color='purple', alpha=0.5)
    ax.add_patch(patch)
    load_patches.append(patch)

info_text = ax.text(0, -1, '', fontsize=10)

def update(frame):
    pos, carrying, action = steps[frame]
    x, y = pos
    robot_icon.center = (y, x)
    info_text.set_text(f"Step: {frame+1} | Pos: {pos} | Load: {carrying} | Action: {action}")
    return robot_icon, info_text

ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=500, repeat=False)
plt.title("Simple Warehouse Agent Animation")
plt.show()

