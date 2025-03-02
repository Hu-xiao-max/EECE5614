import numpy as np
import matplotlib.pyplot as plt

####################################ENV###################################
n_rows = 18
n_cols = 18

# Black (impassable) cells
wall_cells = {
    (2, 5),
    (3, 5),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16),
    (5, 3),
    (6, 3),(6, 6),(6, 9),(6, 15),
    (7, 3),(7, 6),(7, 9),(7, 12),(7, 13),(7, 14),(7, 15),
    (8, 6),(8, 9),(8, 15),
    (9, 6),(9, 9),(9, 15),
    (10, 1),(10, 2),(10,3),(10,4),(10,6),(10,9),(10,10),(10,15),
    (11,6),(11,10),(11,13),(11,15),(11,16),(11,17),
    (12,3),(12,4),(12,5),(12,6),(12,7),(12,10),(12,13),(12,17),
    (13,7),(13,10),(13,13),(13,17),
    (14,7),(14,10),(14,13),
    (15,7),(15,13),(15,14),(15,15),(15,16),
    (17,1), (17,2), (17,7), (17,8), (17,9),(17,10), (17,11), (17,12),
}

# Bump cells
bump_cells = {
    (1, 11),(1, 12),
    (2, 1),(2, 2),(2, 3),
    (5, 1),(5, 9),(5, 17),
    (6, 17),
    (7, 2),(7, 10),(7, 11),(7, 17),
    (8, 17),
    (12, 11),(12, 12),
    (14, 1),(14, 2),
    (15, 17),(15, 18),
    (16, 7)
}

oil_cells = {
    (2, 8), (2, 16),
    (4, 2),
    (5, 6),
    (10, 18),
    (15, 10),
    (16, 10),
    (17, 14),(17, 7),
    (18, 7)
}

# Goal cell (Terminal State)
goal_cell = (3, 13)

# Actions: Up, Down, Left, Right
ACTIONS = {
    'U': (-1, 0),  # Move down (row+1)
    'D': (+1, 0),  # Move up (row-1)
    'L': (0, -1),  # Move left (col-1)
    'R': (0, +1)   # Move right (col+1)
}

goal_cell = (3, 13)
ACTIONS = {'U': (-1, 0), 'D': (+1, 0), 'L': (0, -1), 'R': (0, +1)}


def next_state_and_reward(state, action):
    if state == goal_cell:
        return state
    row, col = state
    dr, dc = ACTIONS[action]
    next_row, next_col = row + dr, col + dc
    if (next_row < 1 or next_row > n_rows or
            next_col < 1 or next_col > n_cols or
            (next_row, next_col) in wall_cells):
        return state
    return (next_row, next_col)


def get_reward(state, next_state):
    action_reward = -1
    hitting_wall_reward = -0.8 if state == next_state else 0
    bump_reward = -10 if next_state in bump_cells else 0
    oil_reward = -5 if next_state in oil_cells else 0
    goal_reward = 300 if next_state == goal_cell else 0
    return action_reward + hitting_wall_reward + bump_reward + oil_reward + goal_reward

############################################################################################################################
# ---------------------------
# 3. Policy Iteration
# ---------------------------


p, gamma, theta = 0.025, 0.99, 0.01
# p, gamma, theta = 0.4, 0.99, 0.01
# p, gamma, theta = 0.025, 0.5, 0.01



policy = {(row, col): 'L' for row in range(1, n_rows + 1) for col in range(1, n_cols + 1)
          if (row, col) not in wall_cells and (row, col) != goal_cell}
V = np.zeros((n_rows, n_cols))
i=0

while True:
    i=i+1
    print(i)
    while True:
        delta = 0
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                if (row, col) in wall_cells or (row, col) == goal_cell:
                    continue
                state = (row, col)
                v_old = V[row - 1, col - 1]
                action = policy[state]
                next_state = next_state_and_reward(state, action)
                reward = get_reward(state, next_state)
                if action == 'U' or 'D':
                    next_state_side1 = next_state_and_reward(state, 'L')
                    reward_side1 = get_reward(state, next_state_side1)
                    next_state_side2 = next_state_and_reward(state, 'R')
                    reward_side2 = get_reward(state, next_state_side2)
                else:
                    next_state_side1 = next_state_and_reward(state, 'U')
                    reward_side1 = get_reward(state, next_state_side1)
                    next_state_side2 = next_state_and_reward(state, 'D')
                    reward_side2 = get_reward(state, next_state_side2)

                state_value = (
                        (1 - p) * (reward + gamma * V[next_state[0] - 1, next_state[1] - 1])
                        + (0.5 * p) * (reward_side1 + gamma * V[next_state_side1[0] - 1, next_state_side1[1] - 1])
                        + (0.5 * p) * (reward_side2 + gamma * V[next_state_side2[0] - 1, next_state_side2[1] - 1])
                )

                V[row - 1, col - 1] = state_value
                delta = max(delta, abs(v_old - V[row - 1, col - 1]))
        if delta < theta:
            break

    # Policy Improvement
    policy_stable = True
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            if (row, col) in wall_cells or (row, col) == goal_cell:
                continue
            state = (row, col)
            old_action = policy[state]
            action_values = {}
            for action in ACTIONS:
                next_state = next_state_and_reward(state, action)
                reward = get_reward(state, next_state)
                action_values[action] = reward + gamma * V[next_state[0] - 1, next_state[1] - 1]
            best_action = max(action_values, key=action_values.get)
            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False

    if policy_stable:
        break























#####################################VISUALIZATION#####################################
# ---------------------------
# 4. Compute Optimal Path
# ---------------------------

def get_optimal_path(start):
    path = [start]
    current = start
    while current != goal_cell:
        best_action = policy[current]
        next_state = next_state_and_reward(current, best_action)
        if next_state == current:
            break
        path.append(next_state)
        current = next_state
    return path


start = (15, 4)
optimal_path = get_optimal_path(start)
print("Optimal Path:", optimal_path)


grid = np.ones((n_rows, n_cols, 3))  # White background

# Assign colors
for r, c in wall_cells:
    grid[r-1, c-1] = [0, 0, 0]  # Black for walls
for r, c in oil_cells:
    grid[r-1, c-1] = [1, 0, 0]  # Red for oil
for r, c in bump_cells:
    grid[r-1, c-1] = [1, 0.5, 0]  # Orange for bumps
grid[goal_cell[0]-1, goal_cell[1]-1] = [0, 1, 0]  # Green for goal

# Mark the start position in blue
start_position = start
grid[start_position[0]-1, start_position[1]-1] = [0, 0, 1]  # Blue for start
# Create figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(grid, origin="upper")

# Draw grid lines
for x in range(n_cols + 1):
    ax.axvline(x - 0.5, color="black", linewidth=0.5)
for y in range(n_rows + 1):
    ax.axhline(y - 0.5, color="black", linewidth=0.5)

# Plot state values
for row in range(1, n_rows + 1):
    for col in range(1, n_cols + 1):
        if (row, col) not in wall_cells:  # Only show values for non-wall states
            value = V[row - 1, col - 1]
            ax.text(col - 1, row - 1, f"{value:.1f}", fontsize=6, ha='center', va='center', color='black')

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])



# Show the visualization
plt.title("State Value Function (V) Visualization in Maze")
plt.show()



start = (15, 4)



grid = np.ones((n_rows, n_cols, 3))  # White background

# Assign colors
for r, c in wall_cells:
    grid[r-1, c-1] = [0, 0, 0]  # Black for walls
for r, c in oil_cells:
    grid[r-1, c-1] = [1, 0, 0]  # Red for oil
for r, c in bump_cells:
    grid[r-1, c-1] = [1, 0.5, 0]  # Orange for bumps
grid[goal_cell[0]-1, goal_cell[1]-1] = [0, 1, 0]  # Green for goal

# Mark the start position in blue
start_position = start
grid[start_position[0]-1, start_position[1]-1] = [0, 0, 1]  # Blue for start
# Create figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(grid, origin="upper")

# Draw grid lines
for x in range(n_cols + 1):
    ax.axvline(x - 0.5, color="black", linewidth=0.5)
for y in range(n_rows + 1):
    ax.axhline(y - 0.5, color="black", linewidth=0.5)

# Plot state values
for row in range(1, n_rows + 1):
    for col in range(1, n_cols + 1):
        if (row, col) not in wall_cells:  # Only show values for non-wall states
            key = tuple([row, col])
            value = policy.get(key)
            ax.text(col - 1, row - 1, f"{value}", fontsize=6, ha='center', va='center', color='black')

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])



# Show the visualization
plt.title("Policy in Maze")
plt.show()



# Example optimal path (assuming policy iteration was performed earlier)
#optimal_path = [(15, 4), (14, 4), (13, 4), (13, 3), (13, 2), (12, 2), (11, 2), (11, 3), (11, 4), (11, 5), (10, 5), (9, 5), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (7, 1), (6, 1), (6, 2), (5, 2), (4, 2), (3, 2), (3, 3), (3, 4), (2, 4), (1, 4), (1, 5), (1, 6), (2, 6), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13)]

# ---------------------------
# 4. Visualization of the Maze with the Optimal Path
# ---------------------------

# Create a grid representation with white background
grid = np.ones((n_rows, n_cols, 3))  # White background

# Assign colors
for r, c in wall_cells:
    grid[r-1, c-1] = [0, 0, 0]  # Black for walls
for r, c in oil_cells:
    grid[r-1, c-1] = [1, 0, 0]  # Red for oil
for r, c in bump_cells:
    grid[r-1, c-1] = [1, 0.5, 0]  # Orange for bumps
grid[goal_cell[0]-1, goal_cell[1]-1] = [0, 1, 0]  # Green for goal

# Mark the start position in blue
start_position = start
grid[start_position[0]-1, start_position[1]-1] = [0, 0, 1]  # Blue for start

# Plot the maze
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(grid, origin="upper")

# Draw grid lines
for x in range(n_cols + 1):
    ax.axvline(x - 0.5, color="black", linewidth=0.5)
for y in range(n_rows + 1):
    ax.axhline(y - 0.5, color="black", linewidth=0.5)

# Plot the optimal path
path_x, path_y = zip(*optimal_path)
ax.plot([y-1 for y in path_y], [x-1 for x in path_x], marker="o", color="blue", markersize=5, linestyle="-")

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Show the visualization
plt.title("Maze with Walls (Black), Oil (Red), Bumps (Orange), Start (Blue), and Grid Lines")
plt.show()