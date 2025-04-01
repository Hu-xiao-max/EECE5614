import itertools
import numpy as np
import pandas as pd
import random

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
# 2. Transition function (M(a))
##############################################################################
C = np.array([
    [ 0,  0, -1,  0],
    [ 1,  0, -1, -1],
    [ 0,  1,  0,  0],
    [-1,  1,  1,  0]
])

def v(x):
    return (x > 0).astype(int)

def compute_transition_matrix(action, p=0.05):
    T = np.zeros((num_states, num_states))
    for i, s_i in enumerate(state_space):
        s_expected = np.bitwise_xor(v(C @ s_i), action)
        for j, s_j in enumerate(state_space):
            dist = np.sum(np.abs(s_j - s_expected))
            T[i, j] = (p ** dist) * ((1 - p) ** (4 - dist))
    return T

transition_matrices = {tuple(a): compute_transition_matrix(a, p=0.05) for a in action_space}

##############################################################################
# 3. Reward function R_{ss}^a
##############################################################################
def reward_function(s_next, action):
    return 5 * np.sum(s_next) - np.sum(np.abs(action))

reward_matrices = {}
for a in action_space:
    a_key = tuple(a)
    R_ss = np.zeros((num_states, num_states))
    for i, s_i in enumerate(state_space):
        for j, s_j in enumerate(state_space):
            R_ss[i, j] = reward_function(s_j, a)
    reward_matrices[a_key] = R_ss

##############################################################################
# 4. Compute expected reward vector R^a
##############################################################################
expected_reward_vectors = {}
ones = np.ones(num_states)
for a in action_space:
    a_key = tuple(a)
    M_a = transition_matrices[a_key]
    R_ss_a = reward_matrices[a_key]
    expected_reward_vectors[a_key] = (M_a * R_ss_a).dot(ones)

##############################################################################
# 5. Policy Iteration
##############################################################################
gamma = 0.9
theta = 1e-2
policy = np.random.choice(num_actions, num_states)
V = np.zeros(num_states)

policy_stable = False
while not policy_stable:
    while True:
        delta = 0
        for i in range(num_states):
            a_key = tuple(action_space[policy[i]])
            r_a = expected_reward_vectors[a_key]
            M_a = transition_matrices[a_key]
            v_new = r_a[i] + gamma * np.dot(M_a[i, :], V)
            delta = max(delta, abs(v_new - V[i]))
            V[i] = v_new
        if delta < theta:
            break

    policy_stable = True
    for i in range(num_states):
        old_action = policy[i]
        best_action = np.argmax([expected_reward_vectors[tuple(a)][i] + gamma * np.dot(transition_matrices[tuple(a)][i, :], V) for a in action_space])
        if old_action != best_action:
            policy_stable = False
        policy[i] = best_action

##############################################################################
# 6. Recover the optimal policy
##############################################################################
optimal_policy = {
    tuple(state_space[i]): action_space[policy[i]]
    for i in range(num_states)
}

policy_df = pd.DataFrame(optimal_policy.items(), columns=["State", "Optimal Action"])
print(policy_df)
policy_df.to_csv("optimal_policy.csv", index=False)
print("Optimal policy saved as 'optimal_policy.csv'")

##############################################################################
# 7. Simulate policy execution
##############################################################################
def sample_next_state(current_state, action, transition_matrices):
    current_index = state_space.index(list(current_state))
    action_key = tuple(action)
    transition_probs = transition_matrices[action_key][current_index]
    next_state_index = np.random.choice(len(state_space), p=transition_probs)
    return state_space[next_state_index]

initial_state = random.choice(state_space)
current_state = initial_state

total_reward = 0
for _ in range(100):
    episode_reward = 0
    for _ in range(200):
        key = tuple(current_state)
        action = optimal_policy.get(key)
        next_state = sample_next_state(current_state, action, transition_matrices)
        current_state = next_state
        reward = sum(current_state)
        episode_reward += reward
    total_reward += episode_reward

print(total_reward / (200 * 100))
