import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- GridWorld Environment for SARSA Level 1 ---
class GridWorld:
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
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

# --- SARSA Parameters ---
episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.2

env = GridWorld()
q_table = np.zeros((env.size, env.size, len(env.actions)))
action_map = {a: i for i, a in enumerate(env.actions)}
index_map = {i: a for a, i in action_map.items()}

rewards_per_episode = []
steps_per_episode = []
successes = 0
start_time = time.time()

# --- SARSA Training Loop ---
for ep in range(episodes):
    state = env.reset()
    row, col = state

    if random.random() < epsilon:
        action = random.choice(env.actions)
    else:
        action = index_map[np.argmax(q_table[row, col])]

    total_reward = 0
    done = False
    steps = 0

    while not done:
        steps += 1
        next_state, reward, done = env.step(action)
        n_row, n_col = next_state

        if random.random() < epsilon:
            next_action = random.choice(env.actions)
        else:
            next_action = index_map[np.argmax(q_table[n_row, n_col])]

        current_q = q_table[row, col, action_map[action]]
        next_q = q_table[n_row, n_col, action_map[next_action]]
        q_table[row, col, action_map[action]] += alpha * (reward + gamma * next_q - current_q)

        total_reward += reward
        state = next_state
        row, col = state
        action = next_action

    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    if env.is_done():
        successes += 1

end_time = time.time()
training_time = end_time - start_time
success_rate = successes / episodes * 100
avg_steps = np.mean(steps_per_episode)
avg_reward = np.mean(rewards_per_episode)

# --- Visualizations ---

# 1. Q-table Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(np.max(q_table, axis=2), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("SARSA: Q-table Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()

# 2. Reward per Episode
plt.figure(figsize=(10, 4))
plt.plot(rewards_per_episode)
plt.title("SARSA: Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

# 3. Steps per Episode
plt.figure(figsize=(10, 4))
plt.plot(steps_per_episode, color='orange')
plt.title("SARSA: Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.show()

# 4. Console Output
print(f"✅ Success Rate: {success_rate:.2f}%")
print(f"⏱️ Training Time: {training_time:.2f} seconds")
print(f"📏 Avg Steps per Episode: {avg_steps:.2f}")
print(f"💰 Avg Reward per Episode: {avg_reward:.2f}")

# 5. Robot Path Visualization
def visualize_robot_path(delay=0.5):
    state = env.reset()
    path = [state]

    fig, ax = plt.subplots()

    for _ in range(50):  # max steps
        grid = np.zeros((env.size, env.size))

        # Path cells
        for pos in path:
            grid[pos] = 0.5
        # Obstacles
        for obs in env.obstacles:
            grid[obs] = -1
        # Goal
        grid[env.goal] = 2
        # Robot
        grid[state] = 1

        ax.clear()
        ax.imshow(grid, cmap='coolwarm', interpolation='nearest')

        for x in range(env.size):
            ax.axhline(x - 0.5, color='black', linewidth=0.5)
            ax.axvline(x - 0.5, color='black', linewidth=0.5)

        plt.title("🤖 SARSA Robot Path Visualization")
        plt.pause(delay)

        if env.is_done():
            break

        row, col = state
        best_action = index_map[np.argmax(q_table[row, col])]
        next_state, _, _ = env.step(best_action)
        state = next_state
        path.append(state)

    plt.show()

# Call the visualizer
visualize_robot_path()
