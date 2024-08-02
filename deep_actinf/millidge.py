# import gym
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# from torch.autograd import Variable

# MEM_SIZE = 100000
# BATCH_SIZE = 200
# STATE_SIZE = 4
# ACTION_SIZE = 2

# class History:
#     def __init__(self, nS, nA, gamma):
#         self.nS = nS
#         self.nA = nA
#         self.gamma = gamma
#         self.states = []
#         self.actions = []
#         self.rewards = []

# def remember(memory, state, action, reward, next_state, done):
#     if len(memory) == MEM_SIZE:
#         memory.popleft()
#     memory.append((state, action, reward, next_state, done))

# def value_loss(x, y):
#     return nn.MSELoss()(x, y)

# def replay(opt_v, valuenet, deep_value_net, memory):
#     batch_size = min(BATCH_SIZE, len(memory))
#     minibatch = random.sample(memory, batch_size)

#     x = np.zeros((batch_size, STATE_SIZE), dtype=np.float32)
#     y = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
#     for i, (state, action, reward, next_state, done) in enumerate(minibatch):
#         target = reward
#         if not done:
#             target += 0.99 * np.max(deep_value_net(torch.FloatTensor(next_state)).detach().numpy())
        
#         target_f = valuenet(torch.FloatTensor(state)).detach().numpy()
#         target_f[action] = target

#         x[i] = state
#         y[i] = target_f

#     x = torch.FloatTensor(x)
#     y = torch.FloatTensor(y)
#     qhats = valuenet(x)
#     loss = value_loss(qhats, y)
#     opt_v.zero_grad()
#     loss.backward()
#     opt_v.step()
#     return loss.item()

# def replay_expectation(opt_v, valuenet, deep_value_net, memory, policynet):
#     batch_size = min(BATCH_SIZE, len(memory))
#     minibatch = random.sample(memory, batch_size)

#     x = np.zeros((batch_size, STATE_SIZE), dtype=np.float32)
#     y = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
#     for i, (state, action, reward, next_state, done) in enumerate(minibatch):
#         target = reward
#         if not done:
#             target += 0.99 * np.sum(nn.Softmax(dim=0)(policynet(torch.FloatTensor(next_state))) * deep_value_net(torch.FloatTensor(next_state)).detach().numpy())
        
#         target_f = valuenet(torch.FloatTensor(state)).detach().numpy()
#         target_f[action] = target

#         x[i] = state
#         y[i] = target_f

#     x = torch.FloatTensor(x)
#     y = torch.FloatTensor(y)
#     qhats = valuenet(x)
#     loss = value_loss(qhats, y)
#     opt_v.zero_grad()
#     loss.backward()
#     opt_v.step()
#     return loss.item()

# def sample_action(probs):
#     cprobs = np.cumsum(probs)
#     sampled = cprobs > np.random.rand()
#     return np.argmax(sampled)

# def mean_ac_loss(history, policynet, valuenet):
#     nS, nA = history.nS, history.nA
#     M = len(history.states) // nS
#     states = np.reshape(history.states, (M, nS))
#     p = nn.Softmax(dim=1)(policynet(torch.FloatTensor(states)))
#     V = valuenet(torch.FloatTensor(states))
#     ploss = -torch.mean(torch.sum(p * nn.LogSoftmax(dim=1)(V), dim=1))
#     return ploss.item()

# def mean_mean_ac_loss(histories, policynet, valuenet):
#     return np.mean([mean_ac_loss(hist, policynet, valuenet) for hist in histories])

# def main(γ=0.99, episodes=15000, render=True, infotime=50):
#     env = gym.make("CartPole-v1")
#     seed = -1
#     if seed > 0:
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         env.seed(seed)

#     valuenet = nn.Sequential(
#         nn.Linear(STATE_SIZE, 100),
#         nn.ReLU(),
#         nn.Linear(100, ACTION_SIZE)
#     )
#     policynet = nn.Sequential(
#         nn.Linear(STATE_SIZE, 100),
#         nn.ReLU(),
#         nn.Linear(100, ACTION_SIZE)
#     )
#     deep_value_net = deepcopy(valuenet)
#     opt_p = optim.Adam(policynet.parameters(), lr=0.001)
#     opt_v = optim.Adam(valuenet.parameters(), lr=0.001)
#     nS, nA = STATE_SIZE, ACTION_SIZE
#     avgreward = 0
#     histories = []
#     ep_rewards = []
#     vlosses = []
#     plosses = []
#     tlosses = []
#     memory = deque(maxlen=MEM_SIZE)
#     for episode in range(1, episodes + 1):
#         state = env.reset()
#         episode_rewards = 0
#         history = History(nS, nA, γ)
#         for t in range(10000):
#             p = policynet(torch.FloatTensor(state))
#             p = nn.Softmax(dim=0)(p)
#             action = sample_action(p.detach().numpy())

#             next_state, reward, done, _ = env.step(action - 1)
#             history.states.extend(state)
#             history.actions.append(action)
#             history.rewards.append(reward)
#             remember(memory, state, action, reward, next_state, done)
#             state = next_state
#             episode_rewards += reward
#             if done:
#                 break
#         ep_rewards.append(episode_rewards)
#         if len(memory) > BATCH_SIZE:
#             vloss = replay(opt_v, valuenet, deep_value_net, memory)
#             vlosses.append(vloss)
#         if episode % infotime == 0:
#             print(f"Episode: {episode}, Reward: {episode_rewards}, Avg Reward: {np.mean(ep_rewards[-infotime:])}")