import itertools
import numpy as np
import pandas as pd

##############################################################################
# 1. Define states and actions
##############################################################################
state_space = [list(i) for i in itertools.product([0, 1], repeat=4)]
action_space = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]
num_states = len(state_space)
num_actions = len(action_space)

##############################################################################
# 2. Example transition function (M(a))
#    (same logic you used, or load from somewhere else)
##############################################################################
C = np.array([
    [ 0,  0, -1,  0],
    [ 1,  0, -1, -1],
    [ 0,  1,  0,  0],
    [-1,  1,  1,  0]
])

def v(x):
    return (x > 0).astype(int)

def compute_transition_matrix(action, p=0.1):
    """
    Return an N×N transition matrix for the given action under noise p.
    """
    T = np.zeros((num_states, num_states))
    for i, s_i in enumerate(state_space):
        # Next state w/o noise:
        s_expected = np.bitwise_xor(v(C @ s_i), action)
        for j, s_j in enumerate(state_space):
            # Hamming distance:
            dist = np.sum(np.abs(s_j - s_expected))
            T[i,j] = (p**dist)*((1 - p)**(4 - dist))
    return T

transition_matrices = {
    tuple(a): compute_transition_matrix(a, p=0.1)
    for a in action_space
}

##############################################################################
# 3. Reward function, R_{ss}^a for each action a
#    We'll build an N×N matrix whose (i,j) entry is the "immediate" reward
#    for going from state i to j via action a.
##############################################################################
def reward_function(s_next, action):
    """
    Example: +5 for each '1' in s_next, minus 1 for each '1' in action
    """
    return 5*np.sum(s_next) - np.sum(np.abs(action))

reward_matrices = {}
for a in action_space:
    a_key = tuple(a)
    # Build NxN, where entry [i,j] = reward_function( state_space[j], a )
    R_ss = np.zeros((num_states, num_states))
    for i, s_i in enumerate(state_space):
        for j, s_j in enumerate(state_space):
            R_ss[i, j] = reward_function(s_j, a)
    reward_matrices[a_key] = R_ss

##############################################################################
# 4. From each R_{ss}^a, build the Nx1 expected-reward vector by using:
#    R^a = (M(a) ∘ R^a_{ss}) * 1_{N×1}
##############################################################################
expected_reward_vectors = {}
ones = np.ones(num_states)  # 1_{N×1} for summation
for a in action_space:
    a_key = tuple(a)
    M_a = transition_matrices[a_key]      # NxN
    R_ss_a = reward_matrices[a_key]       # NxN
    # Elementwise (Hadamard) product:
    hadamard = M_a * R_ss_a              # NxN
    # Summation across columns gives Nx1 vector of expected rewards:
    # (row i) -> sum_j [ M_a[i,j] * R_ss_a[i,j] ]
    r_a = hadamard.dot(ones)
    expected_reward_vectors[a_key] = r_a

##############################################################################
# 5. Matrix-form value iteration:
#    V_{k+1} = max_a [ r^a + gamma * M(a) * V_k ]
##############################################################################
gamma = 0.9
theta = 1e-2
V = np.zeros(num_states)  # initial guess

while True:
    # For each action a, build Q^a = r^a + gamma * M(a) * V
    # We'll stack them in Q_all, shape (num_states, num_actions)
    Q_all = np.zeros((num_states, num_actions))
    for a_idx, a in enumerate(action_space):
        a_key = tuple(a)
        r_a = expected_reward_vectors[a_key]    # Nx1
        M_a = transition_matrices[a_key]        # NxN
        Q_all[:, a_idx] = r_a + gamma * M_a.dot(V)

    V_new = np.max(Q_all, axis=1)
    delta = np.max(np.abs(V_new - V))
    V = V_new
    if delta < theta:
        break

##############################################################################
# 6. Recover the optimal policy
##############################################################################
best_actions = np.argmax(Q_all, axis=1)  # which col is max
optimal_policy = {
    tuple(state_space[i]): action_space[best_actions[i]]
    for i in range(num_states)
}

policy_df = pd.DataFrame(optimal_policy.items(), columns=["State", "Optimal Action"])
print(policy_df)

# Optionally save to CSV
policy_df.to_csv("optimal_policy.csv", index=False)
print("Optimal policy saved as 'optimal_policy.csv'")


import random
# Define function to sample next state based on transition matrix M(a)
def sample_next_state(current_state, action, transition_matrices):
    """Sample next state based on transition probabilities from M(a)."""
    current_index = state_space.index(list(current_state))  # Get index of current state
    action_key = tuple(action)
    transition_probs = transition_matrices[action_key][current_index]  # Get transition probabilities
    next_state_index = np.random.choice(len(state_space), p=transition_probs)  # Sample next state
    return state_space[next_state_index]


initial_state = random.choice(state_space)
current_state = initial_state



print(best_actions)
total_reward = 0
for _ in range(100):
    episode_reward = 0
    for _ in range(200):
        i = decimal_state = int("".join(map(str, current_state)), 2)

        #action = action_space[best_actions[i]]

        action = action_space[0]

        # Sample next state using M(a)
        next_state = sample_next_state(current_state, action, transition_matrices)

        # Move to next state
        current_state = next_state

        reward = current_state[0] + current_state[1] + current_state[2] + current_state[3]

        episode_reward +=reward
    total_reward +=episode_reward

print(total_reward/200/100)











