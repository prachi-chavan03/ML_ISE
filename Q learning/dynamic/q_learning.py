import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
from gridworld import GridWorld  # Import GridWorld class

# Setup environment and Q-learning parameters
size = 5
obstacles = [(1, 1), (2, 2), (3, 1)]  # Static obstacles
dynamic_obstacles = [(0, 4), (4, 0)]  # Initial dynamic obstacles

env = GridWorld(size=size, obstacles=obstacles, dynamic_obstacles=dynamic_obstacles)

# Q-learning parameters
episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.2

q_table = np.zeros((size, size, len(env.actions)))
action_map = {a: i for i, a in enumerate(env.actions)}
index_map = {i: a for a, i in action_map.items()}

# Lists to store performance data
rewards_per_episode = []
steps_per_episode = []
successes = 0
start_time = time.time()

# Training loop
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

        # Update Q-table
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

# --- Print Q-Table ---
print("Final Q-Table:")
print(q_table)

# --- Visualize Q-Table Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(np.max(q_table, axis=2), annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title("Q-Table Heatmap (Max Q-Values)")
plt.xlabel("Columns (States)")
plt.ylabel("Rows (States)")
plt.show()

# --- Show Best Action for Each State ---
best_actions = np.argmax(q_table, axis=2)
print("\nBest Actions for Each State:")
for i in range(size):
    for j in range(size):
        print(f"State ({i},{j}): Best Action -> {index_map[best_actions[i, j]]}")

# --- Track and Visualize Robot Path ---
def track_and_visualize_robot_path(delay=0.5):
    state = env.reset()
    robot_path = [state]  # Initialize path list with starting state
    fig, ax = plt.subplots()

    for _ in range(50):  # Max 50 steps
        grid = np.zeros((env.size, env.size))

        # Mark obstacles and goal
        for obs in env.obstacles:
            grid[obs] = -1  # Static obstacles marked as -1 (Black)
        for obs in env.dynamic_obstacles:
            grid[obs] = -2  # Dynamic obstacles marked as -2 (Red)
        grid[env.goal] = 2  # Goal marked as 2 (Green)

        # Mark robot's path (blue)
        for pos in robot_path:
            grid[pos] = 1  # Robot path marked as 1 (Blue)

        ax.clear()

        # Custom color map for visualization
        cmap = plt.cm.coolwarm  # Use default 'coolwarm' colormap for variety

        # Show the grid with custom colors
        ax.imshow(grid, cmap=cmap, interpolation='nearest')

        # Draw grid lines
        for x in range(env.size):
            ax.axhline(x - 0.5, color='black', linewidth=0.5)
            ax.axvline(x - 0.5, color='black', linewidth=0.5)

        plt.title("Robot Navigating with Path")
        plt.pause(delay)

        if env.is_done():
            break

        row, col = state
        best_action_index = np.argmax(q_table[row, col])
        action = index_map[best_action_index]
        next_state, _, _ = env.step(action)
        state = next_state

        # Add the new state to the robot's path
        robot_path.append(state)

    plt.show()

# --- Plot Reward & Steps Per Episode ---
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-Learning: Total Reward per Episode")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode, color='orange')
plt.xlabel("Episodes")
plt.ylabel("Steps Taken")
plt.title("Q-Learning: Steps per Episode")
plt.grid(True)
plt.show()

# --- Print Success Rate and Training Time ---
print(f"✅ Success Rate: {success_rate:.2f}%")
print(f"⏱️ Training Time: {training_time:.2f} seconds")

# Call to visualize robot's path
track_and_visualize_robot_path(delay=0.5)
