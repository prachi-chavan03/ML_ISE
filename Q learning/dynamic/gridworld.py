import random

class GridWorld:
    def __init__(self, size=5, obstacles=None, dynamic_obstacles=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = obstacles if obstacles else []
        self.dynamic_obstacles = dynamic_obstacles if dynamic_obstacles else []  # Dynamic obstacles
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self):
        self.state = self.start
        self.dynamic_obstacles = [(random.randint(0, self.size - 1), random.randint(0, self.size - 1)) for _ in range(2)]  # Starting dynamic obstacles at random positions
        return self.state

    def is_done(self):
        return self.state == self.goal

    def step(self, action):
        row, col = self.state

        if action == 'up':
            row = max(0, row - 1)
        elif action == 'down':
            row = min(self.size - 1, row + 1)
        elif action == 'left':
            col = max(0, col - 1)
        elif action == 'right':
            col = min(self.size - 1, col + 1)

        new_state = (row, col)

        # If new state is an obstacle, stay in place and apply penalty
        if new_state in self.obstacles or new_state in self.dynamic_obstacles:
            reward = -5
            new_state = self.state  # don't move
        elif new_state == self.goal:
            reward = 10
        else:
            reward = -1  # penalty for each move

        # Move dynamic obstacles
        self.move_dynamic_obstacles()

        self.state = new_state
        return new_state, reward, self.is_done()

    def move_dynamic_obstacles(self):
        """ Move dynamic obstacles randomly in the grid """
        for i in range(len(self.dynamic_obstacles)):
            row, col = self.dynamic_obstacles[i]
            direction = random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up' and row > 0:
                row -= 1
            elif direction == 'down' and row < self.size - 1:
                row += 1
            elif direction == 'left' and col > 0:
                col -= 1
            elif direction == 'right' and col < self.size - 1:
                col += 1

            # Update the position of the obstacle
            self.dynamic_obstacles[i] = (row, col)
