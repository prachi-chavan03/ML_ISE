import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ----- GridWorld Environment -----
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

# ----- Hyperparameter Grid Search -----
alpha_values = [0.1, 0.3, 0.5]
gamma_values = [0.8, 0.9, 0.99]
epsilon_values = [0.1, 0.2, 0.3]

best_reward = float('-inf')
best_config = None
best_q_table = None
best_steps = None
best_rewards = None

# Reduce the number of episodes to speed up testing
episodes = 10  # Adjust for testing

for alpha in alpha_values:
    for gamma in gamma_values:
        for epsilon in epsilon_values:
            print(f"Testing config - Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")  # Debugging line
            env = GridWorld(size=10)
            q_table = np.zeros((10, 10, 4))
            action_map = {a: i for i, a in enumerate(['up', 'down', 'left', 'right'])}
            index_map = {i: a for a, i in action_map.items()}

            total_rewards = []
            steps_per_episode = []
            successes = 0

            for ep in range(episodes):
                state = env.reset()
                row, col = state
                if random.random() < epsilon:
                    action = random.choice(env.actions)
                else:
                    action_index = np.argmax(q_table[row, col])
                    action = index_map[action_index]

                total_reward = 0
                done = False
                steps = 0

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

                    q_table[row, col, action_idx] += alpha * (
                        reward + gamma * q_table[n_row, n_col, next_action_idx] - q_table[row, col, action_idx]
                    )

                    total_reward += reward
                    row, col = n_row, n_col
                    action = next_action

                total_rewards.append(total_reward)
                steps_per_episode.append(steps)
                if env.is_done():
                    successes += 1

            avg_reward = np.mean(total_rewards)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_config = (alpha, gamma, epsilon)
                best_q_table = q_table.copy()
                best_steps = steps_per_episode[:]
                best_rewards = total_rewards[:]

# ----- Results -----
alpha, gamma, epsilon = best_config
print(f"\n✅ Best Hyperparameters - Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")
print(f"📈 Best Avg Reward: {best_reward:.2f}")
print(f"✅ Best Success Rate: {(np.sum(np.array(best_rewards) > 0) / len(best_rewards)) * 100:.2f}%")
print(f"📉 Best Avg Steps: {np.mean(best_steps):.2f}")

# ----- Visualizations -----
plt.figure(figsize=(10, 5))
plt.plot(best_rewards)
plt.title("SARSA (Tuned): Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(best_steps, color='orange')
plt.title("SARSA (Tuned): Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(np.max(best_q_table, axis=2), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("SARSA Q-Table Heatmap (Max Q-Values)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
