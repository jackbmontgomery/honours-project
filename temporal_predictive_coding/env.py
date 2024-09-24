import gymnasium as gym
import numpy as np 

class CartPole:
    def __init__(self):
        """
        Wrapper for the OpenAI Cartpole Environment.
        Modifications:
            1. No longer any reward that is observed in the step
            2. The only observations are the position of the cart and the angle of the pole
        """

        self.env = gym.make('CartPole-v1')
        self.observation_dimension  = 2
        self.control_dimension      = 1

    def _parse_obs(self, obs):

        # Hardcoded selection of the position observation indicies

        postition_indexes = np.array([1, 3], dtype=np.int8)

        obs = [obs[i] for i in postition_indexes]

        return np.reshape(obs, (self.observation_dimension,))

    def reset(self, seed=None):

        obs, info = self.env.reset(seed=seed)

        obs = self._parse_obs(obs)

        return obs, info
    
    def sample_action(self):
        action = self.env.action_space.sample()

        return np.reshape(action, (self.control_dimension,))
    
    def step(self, action):

        if action == -1:
            action += 1

        obs, _, terminated, truncated, info = self.env.step(action)

        obs = self._parse_obs(obs)

        return obs, terminated, truncated, info
    
class BayesianThermostat:

    def __init__():
        pass