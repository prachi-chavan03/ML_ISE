import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ----- GridWorld Environment -----
class GridWorld:
    def __init__(self, size=5, obstacles=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = obstacles if obstacles else []
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
            new_state = self.state  # stay in place
        elif new_state == self.goal:
            reward = 10
        else:
            reward = -1  # penalty for each move

        self.state = new_state
        return new_state, reward, self.is_done()

# ----- Setup Environment & Q-learning Parameters -----
size = 5
obstacles = [(1, 1), (2, 2), (3, 1)]
env = GridWorld(size=size, obstacles=obstacles)

episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.2

q_table = np.zeros((size, size, len(env.actions)))
action_map = {a: i for i, a in enumerate(env.actions)}
index_map = {i: a for a, i in action_map.items()}

rewards_per_episode = []
steps_per_episode = []
successes = 0
start_time = time.time()

# ----- Q-learning Training Loop -----
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        steps += 1
        row, col = state

        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = random.choice(env.actions)
        else:
            action_index = np.argmax(q_table[row, col])
            action = index_map[action_index]

        next_state, reward, done = env.step(action)
        n_row, n_col = next_state

        # Q-learning Update Rule
        old_value = q_table[row, col, action_map[action]]
        future = np.max(q_table[n_row, n_col])
        q_table[row, col, action_map[action]] += alpha * (reward + gamma * future - old_value)

        total_reward += reward
        state = next_state

    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    if env.is_done():
        successes += 1

end_time = time.time()
training_time = end_time - start_time
success_rate = successes / episodes * 100

# ----- Q-Table & Best Actions -----
print("\n✅ Final Q-Table (Max Q-values):")
print(np.max(q_table, axis=2))

print("\n🔁 Best Actions for Each State:")
best_actions = np.argmax(q_table, axis=2)
for i in range(size):
    for j in range(size):
        print(f"State ({i},{j}): Best -> {index_map[best_actions[i, j]]}")

# ----- Visualize Q-Table Heatmap -----
plt.figure(figsize=(8, 6))
sns.heatmap(np.max(q_table, axis=2), annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title("Q-Table Heatmap (Max Q-Values)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()

# ----- Robot Path Visualization Function -----
def visualize_robot_path(delay=0.5):
    state = env.reset()
    path = [state]

    fig, ax = plt.subplots()

    for _ in range(50):  # Max 50 steps
        grid = np.zeros((env.size, env.size))

        # Mark path
        for pos in path:
            grid[pos] = 0.5  # Gray for path

        # Mark static elements
        for obs in env.obstacles:
            grid[obs] = -1   # Red for obstacles
        grid[env.goal] = 2   # Green/Yellow for goal
        grid[state] = 1      # Blue for current robot position

        ax.clear()
        ax.imshow(grid, cmap='coolwarm', interpolation='nearest')

        # Draw grid lines
        for x in range(env.size):
            ax.axhline(x - 0.5, color='black', linewidth=0.5)
            ax.axvline(x - 0.5, color='black', linewidth=0.5)

        plt.title("🤖 Robot Path Visualization")
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

# ----- Call Robot Path Visualization -----
visualize_robot_path()

# ----- Reward and Step Trend Plots -----
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("📈 Q-Learning: Total Reward per Episode")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode, color='orange')
plt.xlabel("Episodes")
plt.ylabel("Steps Taken")
plt.title("📉 Q-Learning: Steps per Episode")
plt.grid(True)
plt.show()

# ----- Final Results -----
print(f"\n✅ Success Rate: {success_rate:.2f}%")
print(f"⏱️ Training Time: {training_time:.2f} seconds")




