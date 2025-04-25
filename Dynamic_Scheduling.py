# dynamic_scheduling.py
import gym
import numpy as np
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

# Simplified delivery environment
class DeliveryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)  # 4 possible next locations
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(2,))
        self.state = np.array([50, 50])  # Start at depot
        self.locations = np.random.rand(4, 2) * 100
        self.step_count = 0
        self.max_steps = 10

    def step(self, action):
        self.step_count += 1
        next_loc = self.locations[action]
        distance = np.sqrt(np.sum((self.state - next_loc) ** 2))
        reward = -distance  # Minimize distance
        self.state = next_loc
        done = self.step_count >= self.max_steps
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([50, 50])
        self.step_count = 0
        return self.state

# Train DQN
env = DeliveryEnv()
model = DQN('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=1000)

# Simulate and visualize
state = env.reset()
rewards = []
for _ in range(10):
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)
    rewards.append(reward)
    if done:
        break

plt.plot(np.cumsum(rewards))
plt.title('Cumulative Rewards (Dynamic Scheduling)')
plt.xlabel('Step')
plt.ylabel('Cumulative Reward')
plt.savefig('visualizations/dynamic_scheduling.png')
plt.close()

if __name__ == "__main__":
    print(f"Total Reward: {sum(rewards):.2f}")