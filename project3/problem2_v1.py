import numpy as np
import itertools
import random
import copy
import matplotlib.pyplot as plt

##############################################################################
# 1. Environment Setup
##############################################################################
class Problem2Env:
    """
    Environment for the 4-bit state (16 possible states) and 4 possible actions.
    Next state follows x_{t+1} = v(M x_t) XOR action, with noise p in each bit.
    Reward:  R(s_{t+1}, a_t) = 5 * sum(s_{t+1}) - sum(a_t).
    """
    def __init__(self, p=0.1):
        """
        p: Bernoulli noise parameter
        """
        self.p = p
        
        # Define the 4-bit state space: all combinations of 0/1 in length 4
        self.state_space = [list(s) for s in itertools.product([0,1], repeat=4)]
        self.num_states   = len(self.state_space)  # 16
        
        # Define the 4 possible actions (no more no-op [0,0,0,0]):
        # flipping exactly one bit
        self.action_space = [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]
        self.num_actions  = len(self.action_space)  # 4

        # Connectivity matrix M (same as original)
        self.C = np.array([
            [ 0,  0, -1,  0],
            [ 1,  0, -1, -1],
            [ 0,  1,  0,  0],
            [-1,  1,  1,  0]
        ])

        # We'll store the "current state" internally in reset() / step()
        self.current_state = None

    def v(self, x):
        """Heaviside step on each component:  v(x_i) = 1 if x_i>0, else 0."""
        return (x > 0).astype(int)
    
    def reset(self):
        """Reset environment to a random initial state."""
        self.current_state = random.choice(self.state_space)
        return self.current_state

    def step(self, action):
        """
        Execute an action (4-bit vector).
        Return: next_state, reward, done, info
         - done = False always (no terminal states)
         - info = {}
        """
        s = self.current_state
        
        # 1) Deterministic next state from M*s
        s_arr = np.array(s)
        a_arr = np.array(action)
        deterministic_next = np.bitwise_xor(self.v(self.C @ s_arr), a_arr)
        
        # 2) Incorporate Bernoulli noise p
        next_state = self._sample_state_with_noise(deterministic_next)
        
        # 3) Compute reward
        r = 5 * np.sum(next_state) - np.sum(a_arr)

        # 4) Update current state
        self.current_state = list(next_state)
        
        return list(next_state), r, False, {}

    def _sample_state_with_noise(self, deterministic_next):
        """Sample a next state from the 16 possibilities based on
           Hamming distance to 'deterministic_next' with noise p."""
        det_list = list(deterministic_next)
        
        # Probability distribution over the 16 states
        probs = []
        for s in self.state_space:
            dist = np.sum(np.abs(np.array(s) - np.array(det_list)))
            p_s = (self.p ** dist) * ((1 - self.p) ** (4 - dist))
            probs.append(p_s)
        
        probs = np.array(probs)
        probs /= probs.sum()  # normalize
        next_idx = np.random.choice(range(self.num_states), p=probs)
        return np.array(self.state_space[next_idx])


##############################################################################
# 2. Matrix-Form Value Iteration
##############################################################################
def matrix_value_iteration(env, gamma=0.9, theta=1e-3):
    """
    Matrix-based value iteration for the environment with 4 actions.
    """
    state_space  = env.state_space
    action_space = env.action_space
    num_states   = env.num_states
    num_actions  = env.num_actions
    C = env.C
    p = env.p

    def v(x):
        return (x > 0).astype(int)
    
    def compute_transition_matrix(action):
        T = np.zeros((num_states, num_states))
        for i, s_i in enumerate(state_space):
            s_expected = np.bitwise_xor(v(C @ s_i), action)
            for j, s_j in enumerate(state_space):
                dist = np.sum(np.abs(np.array(s_j) - s_expected))
                T[i, j] = (p ** dist) * ((1 - p) ** (4 - dist))
        return T

    # Precompute T^a
    transition_matrices = {}
    for a in action_space:
        transition_matrices[tuple(a)] = compute_transition_matrix(a)

    # Build reward matrices R^a_{i,j} = R(state_space[j], a)
    def reward_function(s_next, a):
        return 5 * sum(s_next) - sum(a)
    
    reward_matrices = {}
    for a in action_space:
        a_key = tuple(a)
        R_ss = np.zeros((num_states, num_states))
        for i, s_i in enumerate(state_space):
            for j, s_j in enumerate(state_space):
                R_ss[i, j] = reward_function(s_j, a)
        reward_matrices[a_key] = R_ss

    # Expected reward vector r^a(i) = sum_j [ T^a(i,j) * R^a_{i,j} ]
    expected_reward_vectors = {}
    ones = np.ones(num_states)
    for a in action_space:
        a_key = tuple(a)
        T_a = transition_matrices[a_key]
        R_a = reward_matrices[a_key]
        hadamard = T_a * R_a
        r_a = hadamard.dot(ones)
        expected_reward_vectors[a_key] = r_a

    # Value iteration
    V = np.zeros(num_states)
    while True:
        Q_all = np.zeros((num_states, num_actions))
        for a_idx, a in enumerate(action_space):
            a_key = tuple(a)
            r_a = expected_reward_vectors[a_key]
            T_a = transition_matrices[a_key]
            Q_all[:, a_idx] = r_a + gamma * T_a.dot(V)

        V_new = np.max(Q_all, axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    
    # Recover policy
    best_actions = np.argmax(Q_all, axis=1)
    optimal_policy = {}
    for i, s in enumerate(state_space):
        optimal_policy[tuple(s)] = action_space[best_actions[i]]

    return V, optimal_policy


##############################################################################
# 3. Step-Based RL Algorithms
##############################################################################
def epsilon_greedy(Q, state_idx, epsilon=0.1):
    """
    Epsilon-greedy action selection from Q-table for a given state index.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])  # random action index
    else:
        return np.argmax(Q[state_idx])

def state_to_index(env, s):
    """
    Convert a 4-bit state list [0,1,0,1] into its index in env.state_space.
    """
    return env.state_space.index(s)

def q_learning(env, num_episodes=200, max_steps=100, gamma=0.9,
               alpha=0.1, epsilon=0.1):
    """
    Tabular Q-learning for the given environment (4 actions).
    """
    Q = np.zeros((env.num_states, env.num_actions))
    rewards_per_episode = []

    for _ in range(num_episodes):
        s = env.reset()
        s_idx = state_to_index(env, s)
        total_reward = 0

        for _ in range(max_steps):
            a_idx = epsilon_greedy(Q, s_idx, epsilon)
            a = env.action_space[a_idx]

            s_next, r, done, _ = env.step(a)
            s_next_idx = state_to_index(env, s_next)

            # Q-learning update
            Q[s_idx, a_idx] += alpha * (
                r + gamma * np.max(Q[s_next_idx]) - Q[s_idx, a_idx]
            )

            s_idx = s_next_idx
            total_reward += r
            if done:
                break
        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode

def sarsa(env, num_episodes=200, max_steps=100, gamma=0.9,
          alpha=0.1, epsilon=0.1):
    """
    Tabular SARSA for the given environment (4 actions).
    """
    Q = np.zeros((env.num_states, env.num_actions))
    rewards_per_episode = []

    for _ in range(num_episodes):
        s = env.reset()
        s_idx = state_to_index(env, s)
        a_idx = epsilon_greedy(Q, s_idx, epsilon)

        total_reward = 0
        for _ in range(max_steps):
            a = env.action_space[a_idx]
            s_next, r, done, _ = env.step(a)
            s_next_idx = state_to_index(env, s_next)

            a_next_idx = epsilon_greedy(Q, s_next_idx, epsilon)

            # SARSA update
            Q[s_idx, a_idx] += alpha * (
                r + gamma * Q[s_next_idx, a_next_idx] - Q[s_idx, a_idx]
            )

            s_idx = s_next_idx
            a_idx = a_next_idx
            total_reward += r
            if done:
                break
        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode

def sarsa_lambda(env, num_episodes=200, max_steps=100, gamma=0.9,
                 alpha=0.1, epsilon=0.15, lam=0.95):
    """
    Tabular SARSA(λ) with accumulating traces (4 actions).
    """
    Q = np.zeros((env.num_states, env.num_actions))
    rewards_per_episode = []

    for _ in range(num_episodes):
        s = env.reset()
        s_idx = state_to_index(env, s)
        a_idx = epsilon_greedy(Q, s_idx, epsilon)

        # Initialize eligibility trace
        E = np.zeros((env.num_states, env.num_actions))

        total_reward = 0
        for _ in range(max_steps):
            a = env.action_space[a_idx]
            s_next, r, done, _ = env.step(a)
            s_next_idx = state_to_index(env, s_next)
            a_next_idx = epsilon_greedy(Q, s_next_idx, epsilon)

            # TD error
            delta = r + gamma * Q[s_next_idx, a_next_idx] - Q[s_idx, a_idx]

            # Update eligibility
            E[s_idx, a_idx] += 1

            # Update Q and E for all states/actions
            Q += alpha * delta * E
            E *= gamma * lam

            s_idx = s_next_idx
            a_idx = a_next_idx
            total_reward += r
            if done:
                break

        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode

def tabular_actor_critic(env, num_episodes=200, max_steps=100, gamma=0.9,
                         alpha_w=0.1, alpha_theta=0.1, epsilon=0.1):
    """
    A simple tabular actor-critic with 4 actions.
      - Value function V(s) stored in w[s]
      - Policy stored in a Q-like 'theta[s,a]'
      - We'll do an epsilon-greedy approach to pick actions from theta.
    """
    w = np.zeros(env.num_states)                # value function
    theta = np.zeros((env.num_states, env.num_actions))  # policy parameters
    rewards_per_episode = []

    def get_action_probs(s_idx):
        # Epsilon-greedy w.r.t. argmax(theta[s_idx])
        if np.random.rand() < epsilon:
            probs = np.ones(env.num_actions) / env.num_actions
        else:
            best_a = np.argmax(theta[s_idx])
            probs = np.zeros(env.num_actions)
            probs[best_a] = 1.0
        return probs

    for _ in range(num_episodes):
        s = env.reset()
        s_idx = state_to_index(env, s)
        total_reward = 0

        for _2 in range(max_steps):
            # Sample action from policy
            action_probs = get_action_probs(s_idx)
            a_idx = np.random.choice(env.num_actions, p=action_probs)
            a = env.action_space[a_idx]

            s_next, r, done, _ = env.step(a)
            s_next_idx = state_to_index(env, s_next)

            # Critic update (TD error)
            delta = r + gamma * w[s_next_idx] - w[s_idx]
            w[s_idx] += alpha_w * delta

            # Actor update
            for a_i in range(env.num_actions):
                if a_i == a_idx:
                    theta[s_idx, a_i] += alpha_theta * delta * (1 - action_probs[a_i])
                else:
                    theta[s_idx, a_i] -= alpha_theta * delta * (action_probs[a_i])

            s_idx = s_next_idx
            total_reward += r
            if done:
                break
        rewards_per_episode.append(total_reward)

    return w, theta, rewards_per_episode


##############################################################################
# 4. Utility Functions for Multiple Runs & Policy Extraction
##############################################################################
def run_multiple_runs(
    algo_func,
    env_class,
    num_runs=10,
    num_episodes=1000,
    max_steps=100,
    **algo_kwargs
):
    """
    Runs `algo_func` for `num_runs` independent runs, each time creating a new
    environment from `env_class`. 
    Returns:
      - all_returns: shape (num_runs, num_episodes) with reward per episode
      - all_Q_or_params: list of final learned parameters (Q or (w,theta)) from each run
    """
    all_returns = []
    all_Q_or_params = []
    
    for run_idx in range(num_runs):
        env = env_class(p=0.1)  # p=0.1 as specified
        result = algo_func(env, num_episodes, max_steps, **algo_kwargs)
        
        # Q-learning, SARSA, SARSA-lambda => (Q, rewards)
        # Actor-Critic => (w, theta, rewards)
        if len(result) == 2:
            Q, rewards = result
            all_Q_or_params.append(Q)
            all_returns.append(rewards)
        else:
            w, theta, rewards = result
            all_Q_or_params.append((w, theta))
            all_returns.append(rewards)

    return np.array(all_returns), all_Q_or_params

def extract_policy_from_Q(env, Q):
    """
    Given a Q-table (shape [num_states, num_actions]),
    return a list of the best action (4-bit vector) for each state (16 total).
    """
    policy = []
    for s_idx in range(env.num_states):
        best_a_idx = np.argmax(Q[s_idx])
        best_a = env.action_space[best_a_idx]
        policy.append(best_a)
    return policy

def extract_policy_from_actor_critic(env, theta):
    """
    Given actor parameters theta (shape [num_states, num_actions]),
    return a list of the best action for each state.
    """
    policy = []
    for s_idx in range(env.num_states):
        best_a_idx = np.argmax(theta[s_idx])
        best_a = env.action_space[best_a_idx]
        policy.append(best_a)
    return policy

def print_policies(env, all_policies):
    """
    Print the final policies as a 10x16 matrix.
    Each row corresponds to one independent experiment (10 runs),
    and each column corresponds to one state (16 states).
    The action vector for each state is printed as: "a1 a2 a3 a4".
    """
    for policy in all_policies:
        row = []
        for action in policy:
            # Convert the action vector to a space-separated string
            action_str = " ".join(str(a) for a in action)
            row.append(action_str)
        # Print the row with each action vector separated by a delimiter
        print(" | ".join(row))

##############################################################################
# 5. Example Main (Compare Q-Learning, SARSA, SARSA-lambda, Actor-Critic)
##############################################################################
if __name__ == "__main__":
    # 1) Create environment class with 4 actions
    def env_factory(p=0.1):
        return Problem2Env(p=p)

    # 2) Run matrix-based Value Iteration for reference
    env_ref = env_factory()
    V_star, pi_star = matrix_value_iteration(env_ref, gamma=0.9)
    print("=== Matrix-based Value Iteration (Reference) ===")
    for i in range(5):
        s_tuple = tuple(env_ref.state_space[i])
        print(f"State {s_tuple} -> V={V_star[i]:.2f}, π(s)={pi_star[s_tuple]}")
    print()

    # 3) Compare Q-Learning, SARSA, SARSA-lambda, Actor-Critic
    num_runs = 10
    num_episodes = 1000
    max_steps = 100
    gamma = 0.9
    alpha = 0.25
    epsilon = 0.15

    # Example: Q-Learning
    print("=== Q-Learning: 10 independent runs ===")
    qlearn_returns, qlearn_params = run_multiple_runs(
        q_learning, env_factory,
        num_runs=num_runs,
        num_episodes=num_episodes,
        max_steps=max_steps,
        gamma=gamma,
        alpha=alpha,
        epsilon=epsilon
    )
    mean_qlearning = np.mean(qlearn_returns, axis=0)

    # Extract final policy from each run
    env_for_eval = env_factory()
    qlearn_policies = [extract_policy_from_Q(env_for_eval, Q) for Q in qlearn_params]

    # SARSA
    print("=== SARSA: 10 independent runs ===")
    sarsa_returns, sarsa_params = run_multiple_runs(
        sarsa, env_factory,
        num_runs=num_runs,
        num_episodes=num_episodes,
        max_steps=max_steps,
        gamma=gamma,
        alpha=alpha,
        epsilon=epsilon
    )
    mean_sarsa = np.mean(sarsa_returns, axis=0)
    sarsa_policies = [extract_policy_from_Q(env_for_eval, Q) for Q in sarsa_params]

    # SARSA(λ)
    print("=== SARSA(λ): 10 independent runs ===")
    sarsa_lam_returns, sarsa_lam_params = run_multiple_runs(
        sarsa_lambda, env_factory,
        num_runs=num_runs,
        num_episodes=num_episodes,
        max_steps=max_steps,
        gamma=gamma,
        alpha=alpha,
        epsilon=epsilon,
        lam=0.95
    )
    mean_sarsa_lam = np.mean(sarsa_lam_returns, axis=0)
    sarsa_lam_policies = [extract_policy_from_Q(env_for_eval, Q) for Q in sarsa_lam_params]

    # Actor-Critic
    print("=== Tabular Actor-Critic: 10 independent runs ===")
    ac_returns, ac_params = run_multiple_runs(
        tabular_actor_critic, env_factory,
        num_runs=num_runs,
        num_episodes=num_episodes,
        max_steps=max_steps,
        gamma=gamma,
        alpha_w=alpha,         # example
        alpha_theta=0.05,      # example
        epsilon=epsilon
    )
    mean_ac = np.mean(ac_returns, axis=0)
    ac_policies = []
    for (w, theta) in ac_params:
        ac_policies.append(extract_policy_from_actor_critic(env_for_eval, theta))

    # 4) Plot average learning curves

    # Plot Q-Learning
    plt.figure(figsize=(8, 5))
    plt.plot(mean_qlearning, label='Q-Learning', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('Q-Learning')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot SARSA
    plt.figure(figsize=(8, 5))
    plt.plot(mean_sarsa, label='SARSA', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('SARSA')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot SARSA(λ)
    plt.figure(figsize=(8, 5))
    plt.plot(mean_sarsa_lam, label='SARSA(λ)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('SARSA(λ)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Actor-Critic
    plt.figure(figsize=(8, 5))
    plt.plot(mean_ac, label='Actor-Critic', color='black')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('Actor-Critic')
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(mean_qlearning, label='Q-Learning')
    plt.plot(mean_sarsa, label='SARSA')
    plt.plot(mean_sarsa_lam, label='SARSA(λ)')
    plt.plot(mean_ac, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('Comparison of RL Algorithms (4 Actions)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 5) Show final average reward
    print("\n=== Final Average Reward (last 50 episodes) ===")
    print("Q-Learning   =", np.mean(mean_qlearning[-50:]))
    print("SARSA        =", np.mean(mean_sarsa[-50:]))
    print("SARSA(λ)     =", np.mean(mean_sarsa_lam[-50:]))
    print("Actor-Critic =", np.mean(mean_ac[-50:]))

    # (Optional) Print the final policy from each run
    print('***************q learning***************')
    print_policies(env_for_eval, qlearn_policies)
    print('***************sarsa***************')
    print_policies(env_for_eval, sarsa_policies)
    print('***************sarsa lam***************')
    print_policies(env_for_eval, sarsa_lam_policies)
    print('***************ac***************')
    print_policies(env_for_eval, ac_policies)