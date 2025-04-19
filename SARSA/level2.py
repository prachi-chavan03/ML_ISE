import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ----- GridWorld Environment (Same as Q-learning Level 2) -----
class GridWorld:
    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (9, 9)

        self.obstacles = obstacles if obstacles else []
        while len(self.obstacles) < 20:
            obs = (random.randint(0, size-1), random.randint(0, size-1))
            if obs != self.start and obs != self.goal and obs not in self.obstacles:
                self.obstacles.append(obs)

        self.actions = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def is_done(self):
        return self.state == self.goal

    def step(self, action):
        row, col = self.state
        if action == 'up': row = max(0, row - 1)
        elif action == 'down': row = min(self.size - 1, row + 1)
        elif action == 'left': col = max(0, col - 1)
        elif action == 'right': col = min(self.size - 1, col + 1)

        new_state = (row, col)

        if new_state in self.obstacles:
            reward = -5
            new_state = self.state
        elif new_state == self.goal:
            reward = 10
        else:
            reward = -1

        self.state = new_state
        return new_state, reward, self.is_done()

# ----- SARSA Training Setup -----
size = 10
episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.2

q_table = np.zeros((size, size, 4))
action_map = {a: i for i, a in enumerate(['up', 'down', 'left', 'right'])}
index_map = {i: a for a, i in action_map.items()}

env = GridWorld(size=size)

rewards_per_episode = []
steps_per_episode = []
successes = 0
start_time = time.time()

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    row, col = state
    if random.random() < epsilon:
        action = random.choice(env.actions)
    else:
        action_index = np.argmax(q_table[row, col])
        action = index_map[action_index]

    while not done:
        steps += 1
        action_idx = action_map[action]
        next_state, reward, done = env.step(action)
        n_row, n_col = next_state

        if random.random() < epsilon:
            next_action = random.choice(env.actions)
        else:
            next_action_index = np.argmax(q_table[n_row, n_col])
            next_action = index_map[next_action_index]

        next_action_idx = action_map[next_action]

        q_table[row, col, action_idx] += alpha * (reward + gamma * q_table[n_row, n_col, next_action_idx] - q_table[row, col, action_idx])

        total_reward += reward
        row, col = n_row, n_col
        action = next_action

    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    if env.is_done():
        successes += 1

end_time = time.time()
training_time = end_time - start_time
success_rate = successes / episodes * 100

# ----- Metrics Output -----
avg_steps = np.mean(steps_per_episode)
avg_reward = np.mean(rewards_per_episode)

print(f"\U00002705 Success Rate: {success_rate:.2f}%")
print(f"\u23F1\uFE0F Training Time: {training_time:.2f} seconds")
print(f"\U0001F4CF Avg Steps per Episode: {avg_steps:.2f}")
print(f"\U0001F4B0 Avg Reward per Episode: {avg_reward:.2f}")

# ----- Heatmap -----
plt.figure(figsize=(8, 6))
sns.heatmap(np.max(q_table, axis=2), annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title("SARSA Q-Table Heatmap (Max Q-Values) - Level 2")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()

# ----- Reward Plot -----
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("SARSA: Total Reward per Episode (Level 2)")
plt.grid(True)
plt.show()

# ----- Steps Plot -----
plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode, color='orange')
plt.xlabel("Episodes")
plt.ylabel("Steps Taken")
plt.title("SARSA: Steps per Episode (Level 2)")
plt.grid(True)
plt.show()

# ----- Path Visualization -----
def visualize_robot_path(delay=0.5):
    state = env.reset()
    path = [state]
    fig, ax = plt.subplots()

    for _ in range(50):
        grid = np.zeros((env.size, env.size))
        for pos in path:
            grid[pos] = 0.5
        for obs in env.obstacles:
            grid[obs] = -1
        grid[env.goal] = 2
        grid[state] = 1

        ax.clear()
        ax.imshow(grid, cmap='coolwarm', interpolation='nearest')
        for x in range(env.size):
            ax.axhline(x - 0.5, color='black', linewidth=0.5)
            ax.axvline(x - 0.5, color='black', linewidth=0.5)

        plt.title("\U0001F916 Robot Path Visualization - SARSA")
        plt.pause(delay)

        if env.is_done():
            break

        row, col = state
        best_action_index = np.argmax(q_table[row, col])
        action = index_map[best_action_index]
        next_state, _, _ = env.step(action)
        state = next_state
        path.append(state)

    plt.show()

visualize_robot_path()
