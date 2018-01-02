import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import gym
import random
from collections import deque
from gym.envs.registration import registry, register, make, spec
import matplotlib.pyplot as plt


class DQN():
    # DQN Agent
    def __init__(self,
                 env,
                 epsilon,
                 learning_rate=0.01,
                 gamma=0.95,
                 weight_decay=0.001,
                 batch_size=64,
                 replay_size=1000,
                 train_delay=16
                 ):
        # init experience replay
        # init some parameters
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.final_epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.train_delay = train_delay
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.train_loss = []
        self.count = 0

        self.create_Q_network()
        self.create_training_method()

    def create_Q_network(self):
        # network weights
        self.layer1 = torch.nn.Linear(self.state_dim, 24)
        nn.init.xavier_uniform(self.layer1.weight)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(24, 48)
        nn.init.xavier_uniform(self.layer2.weight)
        self.layer3 = torch.nn.Linear(48, self.action_dim)
        nn.init.xavier_uniform(self.layer3.weight)
        self.value_net = nn.Sequential(self.layer1, self.relu, self.layer2,
                                       self.relu, self.layer3)

    # def forward(self, input):
    #     return self.net(input)error

    def create_training_method(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = [0 for i in range(self.action_dim)]
        one_hot_action[action] = 1
        self.count += 1
        # state = Variable(torch.FloatTensor(state.tolist()))
        # print('type(action) = ', type(action))
        # one_hot_action = Variable(torch.LongTensor(one_hot_action))
        # reward = Variable(torch.FloatTensor([reward]))
        # next_state = Variable(torch.FloatTensor(next_state.tolist()))
        self.replay_buffer.append((state, one_hot_action, reward, next_state,
                                   done))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()

        # if len(self.replay_buffer) >= BATCH_SIZE:
        if len(self.replay_buffer) >= self.batch_size:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        # print('type(minibatch) = ', type(minibatch))
        # print('minibatch = \n', minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        state_batch = Variable(torch.FloatTensor(state_batch))
        action_batch = Variable(torch.FloatTensor(action_batch))
        # reward_batch = Variable(torch.FloatTensor(reward_batch))
        next_state_batch = Variable(torch.FloatTensor(next_state_batch))

        # Step 2: calculate y
        y_batch = []
        next_Q_value_batch = self.value_net(next_state_batch)
        next_Q_value_batch = next_Q_value_batch.data
        for i in range(0, self.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                # print('torch.max(next_Q_value_batch[i]) = ', torch.max(next_Q_value_batch[i]))
                y_batch.append(reward_batch[i] +
                               self.gamma * torch.max(next_Q_value_batch[i]))

        Q_value_batch = self.value_net(state_batch)
        # print('Q_value_batch.size() = ', Q_value_batch.size())
        Q_action = torch.sum(torch.mul(Q_value_batch, action_batch), dim=1)
        y_batch = Variable(torch.FloatTensor(y_batch), requires_grad=False)
        loss = self.criterion(Q_action, y_batch)
        self.train_loss.append(loss.data[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.time_step % 1000 == 0:
            print('loss = {:.5f}, timestep = {}'.format(
                loss.data[0], self.time_step + 1))

    def egreedy_action(self, state):
        # print('Q_value.size() = ', Q_value.size())
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            # action.dtype='int'
            # action = action.astype('int')
            # print('1egreedy_action:type(action) = ', type(action))
            # return random.randint(0, self.action_dim - 1)
        else:
            state = Variable(torch.FloatTensor(state.tolist()))
            # Q_value = self.value_net(state)[0]
            Q_value = self.value_net(state).data
            _, action = torch.max(Q_value, 0)
            # print('action = ', action)
            # action.dtype='int'
            # action = action.astype(int)
            action = int(action[0])
            # print('type(action) = ', type(action))

            # print('2egreedy_action:type(action) = ', type(action))
            # return action
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        # print('egreedy_actiontype(action) = ', action)
        return action

    def get_action(self, state):
        state = Variable(torch.FloatTensor(state.tolist()))
        _, action = torch.max(self.value_net(state).data, 0)
        # print('get_action:action = ', action)
        action = int(action[0])
        return action
    
    def visualize_loss(self, pic_name):
        fig, ax = plt.subplots()
        ax.plot([i+1 for i in range(len(self.train_loss))], self.train_loss)
        title = 'loss during training'
        ax.set(xlabel='training numbers',
               ylabel='training error',
               title=title)
        ax.grid()
        fig.savefig(pic_name)
        # plt.show()
    '''
    def visualize_reward(self):
        fig, ax = plt.subplots()
        # ax.plot([i+1 for i in range(len(self.train_loss))], self.train_reward)
        title = 'reward during training'
        ax.set(xlabel='training numbers',
               ylabel='test reward during training',
               title=title)
        ax.grid()
        fig.savefig(pic_name)
        # plt.show()
    '''