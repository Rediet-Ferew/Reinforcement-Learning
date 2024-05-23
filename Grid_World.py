import gymnasium as gym
import numpy as np

# Set up the FrozenLake environment with custom parameters
env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
env.reset()

# Helper function to visualize the policy
def print_policy(policy, grid_size):
    policy_symbols = {
        0: '←', 1: '↓', 2: '→', 3: '↑'
    }
    policy_grid = np.array([policy_symbols[action] for action in policy]).reshape(grid_size)
    for row in policy_grid:
        print(' '.join(row))

# Value Iteration
def value_iteration(env, gamma=0.99, theta=1e-9):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    while True:
        delta = 0
        for s in range(n_states):
            q_sa = [sum([prob * (reward + gamma * V[s_])
                         for prob, s_, reward, _ in env.unwrapped.P[s][a]])
                    for a in range(n_actions)]
            best_action_value = max(q_sa)
            delta = max(delta, np.abs(V[s] - best_action_value))
            V[s] = best_action_value
            policy[s] = np.argmax(q_sa)
        
        if delta < theta:
            break

    return policy, V

# Policy Iteration
def policy_iteration(env, gamma=0.99):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.random.choice(n_actions, size=n_states)
    V = np.zeros(n_states)
    
    def policy_eval(policy):
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                a = policy[s]
                V[s] = sum([prob * (reward + gamma * V[s_])
                            for prob, s_, reward, _ in env.unwrapped.P[s][a]])
                delta = max(delta, np.abs(v - V[s]))
            if delta < 1e-10:
                break

    while True:
        policy_eval(policy)
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            q_sa = [sum([prob * (reward + gamma * V[s_])
                         for prob, s_, reward, _ in env.unwrapped.P[s][a]])
                    for a in range(n_actions)]
            new_action = np.argmax(q_sa)
            policy[s] = new_action
            if new_action != old_action:
                policy_stable = False
        if policy_stable:
            break

    return policy, V

# Q-Learning
def q_learning(env, num_episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    def epsilon_greedy_policy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(n_actions)
        else:
            return np.argmax(Q[state])

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(state)
            next_state, reward, done, _, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            state = next_state

    policy = np.argmax(Q, axis=1)
    return policy, Q

# Run the algorithms
grid_size = (4, 4)  # for FrozenLake-v1 4x4
policy_vi, _ = value_iteration(env)
policy_pi, _ = policy_iteration(env)
policy_ql, _ = q_learning(env)

# Display the policies
print("Value Iteration Policy:")
print_policy(policy_vi, grid_size)

print("\nPolicy Iteration Policy:")
print_policy(policy_pi, grid_size)

print("\nQ-Learning Policy:")
print_policy(policy_ql, grid_size)
