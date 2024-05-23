import numpy as np

class MultiArmedBanditEnv:
    def __init__(self, k):
        self.k = k
        self.probabilities = np.random.rand(k)  # Random probabilities for each arm
        self.best_arm = np.argmax(self.probabilities)

    def reset(self):
        return 0  # single state

    def step(self, action):
        reward = np.random.binomial(1, self.probabilities[action])
        return 0, reward, False, {}

    def optimal_strategy(self):
        return self.best_arm, self.probabilities[self.best_arm]

class EpsilonGreedyAgent:
    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.k)
        else:
            return np.argmax(self.values)

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def run(self, env, n_steps):
        total_reward = 0
        for step in range(n_steps):
            action = self.select_action()
            _, reward, _, _ = env.step(action)
            self.update(action, reward)
            total_reward += reward
        return total_reward
class UCBAgent:
    def __init__(self, k, c=2):
        self.k = k
        self.c = c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.total_counts = 0

    def select_action(self):
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts + 1) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        self.total_counts += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def run(self, env, n_steps):
        total_reward = 0
        for step in range(n_steps):
            action = self.select_action()
            _, reward, _, _ = env.step(action)
            self.update(action, reward)
            total_reward += reward
        return total_reward
k = 10  # Number of arms
n_steps = 1000  # Number of steps

env = MultiArmedBanditEnv(k)

# Epsilon-Greedy Agent
epsilon_greedy_agent = EpsilonGreedyAgent(k, epsilon=0.1)
epsilon_greedy_reward = epsilon_greedy_agent.run(env, n_steps)

# UCB Agent
ucb_agent = UCBAgent(k, c=2)
ucb_reward = ucb_agent.run(env, n_steps)

print(f"Epsilon-Greedy Total Reward: {epsilon_greedy_reward}")
print(f"UCB Total Reward: {ucb_reward}")

# Optimal strategy
optimal_action, optimal_reward = env.optimal_strategy()
print(f"Optimal Action: {optimal_action}, Optimal Reward Rate: {optimal_reward * n_steps}")
