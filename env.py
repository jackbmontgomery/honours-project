import gymnasium as gym
import numpy as np 
class CartPoleWrapper():
    def __init__(self):

        self.env = gym.make('CartPole-v1')

        self.observation_dimension  = self.env.observation_space.shape[-1] + 1 # Plus reward
        self.control_dimension      = 1

    def _parse_obs(self, obs, reward):
        # To array
        obs = obs.tolist()
        # Append reward
        obs.append(reward)

        return np.reshape(obs, (self.observation_dimension,))

    def reset(self, seed=None):

        obs, info = self.env.reset(seed=seed)

        # To array
        obs = self._parse_obs(obs, 1)

        return obs, info
    
    def sample_action(self):
        action = self.env.action_space.sample()

        return np.reshape(action, (self.control_dimension,))
    
    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._parse_obs(obs, reward)

        return obs, terminated, truncated, info
