import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import pickle
import gym
from gym import spaces
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D

# Q Tablosunu Yükle
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)


# Ortam

class WarehouseEnv(gym.Env):
    def __init__(self):
        super(WarehouseEnv, self).__init__()
        self.grid_size = (7, 7)
        self.goal_position = (3, 3)
        self.charge_station = (1, 5)
        self.obstacles = [(2, 2), (4, 4), (5, 1)]
        self.initial_loads = [(6, 6), (5, 5)]
        self.max_battery = 50
        self.max_load = 2

        self.action_space = spaces.Discrete(7)  # Up, Down, Right, Left, Pickup, Drop, Recharge
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
            next_x, next_y = x, y  # Engel varsa geri dön

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

    def render_array(self):
        grid = np.ones((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8) * 255

        for ox, oy in self.obstacles:
            grid[ox, oy] = [0, 0, 0]  # Engeller: Siyah

        for lx, ly in self.loads:
            grid[lx, ly] = [255, 0, 0]  # Yük: Kırmızı

        gx, gy = self.goal_position
        grid[gx, gy] = [0, 255, 0]  # Hedef: Yeşil

        csx, csy = self.charge_station
        grid[csx, csy] = [0, 0, 255]  # Şarj: Mavi

        rx, ry = self.position
        grid[rx, ry] = [255, 255, 0]  # Robot: Sarı

        return grid
    def render(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Engeller
        for ox, oy, oz in self.obstacles:
            ax.scatter(ox, oy, oz, color='red', s=100)  # Engel: Kırmızı

        # Yükler
        for lx, ly, lz in self.initial_loads:
            ax.scatter(lx, ly, lz, color='purple', s=100)  # Yük: Mor

        # Hedef (goal) noktası
        gx, gy, gz = self.goal_position
        ax.scatter(gx, gy, gz, color='green', s=100)  # Hedef: Yeşil

        # Şarj istasyonu
        csx, csy, csz = self.charge_station
        ax.scatter(csx, csy, csz, color='blue', s=100)  # Şarj: Mavi

        # Robotun (araç) pozisyonu
        rx, ry, rz = self.position
        ax.scatter(rx, ry, rz, color='yellow', s=100)  # Robot: Sarı

        # Eksen sınırları ve etiketler
        ax.set_xlim([0, self.grid_size[0]])
        ax.set_ylim([0, self.grid_size[1]])
        ax.set_zlim([0, self.grid_size[2]])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()




# === Test Adımları ===
env = WarehouseEnv()
obs = env.reset()
done = False
steps = []
total_reward = 0

while not done:
    x, y, load, battery = obs
    # En iyi aksiyonu seç (greedy politika)
    action = np.argmax(Q[x, y, load, battery, :])
    
    # Çevreyi güncelle
    next_obs, reward, done, _ = env.step(action)
    total_reward += reward
    steps.append((env.position, env.carrying_load, env.battery, action))
    obs = next_obs

# === Matplotlib Animasyon Ayarları ===
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-0.5, env.grid_size[1] - 0.5)
ax.set_ylim(-0.5, env.grid_size[0] - 0.5)
ax.set_xticks(range(env.grid_size[1]))
ax.set_yticks(range(env.grid_size[0]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

robot_icon = plt.Circle((0, 0), 0.3, color='blue')
ax.add_patch(robot_icon)

obstacle_patches = [plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='black') for x, y in env.obstacles]
for obs_patch in obstacle_patches:
    ax.add_patch(obs_patch)

goal_patch = plt.Rectangle((env.goal_position[1] - 0.5, env.goal_position[0] - 0.5), 1, 1, color='green', alpha=0.4)
ax.add_patch(goal_patch)

charge_patch = plt.Rectangle((env.charge_station[1] - 0.5, env.charge_station[0] - 0.5), 1, 1, color='orange', alpha=0.4)
ax.add_patch(charge_patch)

load_patches = []
for lx, ly in env.initial_loads:
    p = plt.Rectangle((ly - 0.5, lx - 0.5), 1, 1, color='purple', alpha=0.5)
    load_patches.append(p)
    ax.add_patch(p)

info_text = ax.text(0, -1, '', fontsize=10, ha='left')

# === Animasyon Fonksiyonu ===
def update(frame):
    pos, load, battery, action = steps[frame]
    x, y = pos
    robot_icon.center = (y, x)  # Ajanın pozisyonunu güncelle
    print(f"Adım {frame+1} | Pozisyon: {pos} | Yük: {load} | Batarya: {battery}")

    info_text.set_text(f"Adım: {frame+1} | Yük: {load} | Batarya: {battery} | Aksiyon: {action}")
    return robot_icon, info_text
    

ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=500, repeat=False)

plt.title("Depo Robotunun Yük Toplama Animasyonu")
plt.show()


"""def run_multiple_episodes(env, Q, num_episodes=10):
    all_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            x, y, load, battery = obs
            action = np.argmax(Q[x, y, load, battery, :])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        all_rewards.append(episode_reward)
        print(f"🎮 Episode {episode+1} - Toplam Ödül: {episode_reward}")

    print(f"\n📊 Ortalama Ödül: {np.mean(all_rewards)}")
    return all_rewards

run_multiple_episodes(env,Q,num_episodes=5)"""



