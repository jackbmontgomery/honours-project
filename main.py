import torch.nn as nn
import torch
import predictive_coding as pc
import random
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
from gym.wrappers import TimeLimit
from tqdm import tqdm
import wandb
import copy

class NeuralNetwork(nn.Module):

    def __init__(self, predictive_coding, num_obs, num_act, bias=True, pc_layer_at='before_acf', hidden_size=128, num_hidden=1, acf='Sigmoid'):

        super(NeuralNetwork, self).__init__()

        self.predictive_coding = predictive_coding
        self.num_act = num_act

        model = []

        # input layer
        model.append(nn.Linear(num_obs, hidden_size, bias=bias))
        if self.predictive_coding and pc_layer_at == 'before_acf':
            model.append(pc.PCLayer())
        model.append(eval('nn.{}()'.format(acf)))
        if self.predictive_coding and pc_layer_at == 'after_acf':
            model.append(pc.PCLayer())

        for _ in range(num_hidden):

            # hidden layer
            model.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            if self.predictive_coding and pc_layer_at == 'before_acf':
                model.append(pc.PCLayer())
            model.append(eval('nn.{}()'.format(acf)))
            if self.predictive_coding and pc_layer_at == 'after_acf':
                model.append(pc.PCLayer())

        # output layer
        model.append(nn.Linear(hidden_size, num_act, bias=bias))

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)

class PredictiveQAgent():

    def __init__(
            self, 
            env: gym.Env, 
            buffer_size: int = 1000,
            batch_size: int = 64,
            device: str = "cpu", 
            eps_min: float = 0.05,
            eps_decay: float = 0.999,
            gamma: float = 0.99,
            load_target_network_period: int = 100
        ):

        self.device     = device
        self.gamma    = gamma
        self.env        = env

        # Q-Network
        
        self.q_network      = NeuralNetwork(predictive_coding=True, num_obs=env.observation_space.shape[0], num_act=env.action_space.n)

        self.target_network = NeuralNetwork(predictive_coding=True, num_obs=env.observation_space.shape[0], num_act=env.action_space.n)
        self.target_network.load_state_dict(self.q_network.state_dict(), strict=False)

        self.load_target_network_period = load_target_network_period

        # Trainer

        self.pc_trainer = pc.PCTrainer(model=self.q_network, plot_progress_at=[])

        # Replay Buffer
        self.batch_size     = batch_size
        self.buffer_size    = buffer_size
        self.memory         = ReplayBuffer(
            buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            handle_timeout_termination=False
        )

        # Epsilon Greedy:

        self.eps_min    = eps_min
        self.eps_decay  = eps_decay
        self.eps = 1.0

    def sample_action(self, obs: torch.Tensor):

        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        if random.random() < self.eps:
            action = self.env.action_space.sample()

        else:
            with torch.no_grad():
                action_q_values = self.q_network(obs)
                action = torch.argmax(action_q_values).numpy()

        return action
    
    def _can_learn(self):
        return self.memory.size() >= self.batch_size
    
    def learn(self):

        batch = self.memory.sample(self.batch_size)

        # Compute target

        with torch.no_grad():
            action_q_values_next    = self.target_network(batch.next_observations)
            max_next_value          = action_q_values_next.max(dim=1).values

        target = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * max_next_value

        self.q_network.train()

        def loss_fn(outputs, actions, target):
            predicted = outputs.gather(1, actions).flatten()
            loss = (predicted - target).pow(2).sum() * 0.5
            return loss
        
        self.pc_trainer.train_on_batch(
            batch.observations,
            loss_fn=loss_fn,
            loss_fn_kwargs={
                'actions': batch.actions,
                'target': target
            })
    
    def train(self, num_episodes: int = 100, max_episode_steps: int = 1000):

        env = TimeLimit(self.env, max_episode_steps=max_episode_steps)

        total_training_steps = 0

        for e in range(num_episodes):

            obs, _ = env.reset()

            done            = False
            episode_rewards = 0

            while not done:

                total_training_steps += 1
                
                action = self.sample_action(torch.from_numpy(obs))

                next_obs, reward, termination, truncation, info = env.step(action)

                self.memory.add(obs, next_obs, action, reward, termination, info)

                if self._can_learn():
                    self.learn()

                if total_training_steps % self.load_target_network_period == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict(), strict=False)

                obs = next_obs
                done            = termination or truncation
                episode_rewards += reward

            print(f'Episode: {e} - Reward: {episode_rewards}')

env = gym.make("CartPole-v1")
agent = PredictiveQAgent(env=env)
agent.train()