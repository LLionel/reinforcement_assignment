'''
NUM_EPISODES = 2000
MAX_T = 20010
ALPHA = 0.1
GAMMA = 0.99

EXPLORATION_RATE = 0.5
EXPLORATION_RATE_DECAY = 0.9

'''
import gym
import numpy as np
import random
import math
import pickle as pkl
import matplotlib.pyplot as plt

# NUM_EPISODES = 1000
# MAX_T = 20010
# ALPHA = 0.5
# GAMMA = 0.99

# EXPLORATION_RATE = 0.5
# EXPLORATION_RATE_DECAY = 0.9

# env = gym.make('CartPole-v0')
# env = env.unwrapped
# NUM_ACTIONS = env.action_space.n

# CART_POS = np.linspace(-2.4, 2.4, 12)
# CART_VEL = np.linspace(-1, 1, 5)
# POLE_ANGLE = np.linspace(-0.3, 0.3, 24)
# ANG_RATE = np.linspace(-1, 1, 5)


class QLearning():

    def __init__(self,
                 env,
                 bins,
                 episodes=10000,
                 num_decay=None,
                 exploration_rate=0.5,
                 exploration_rate_decay=0.9,
                 alpha=0.5,
                 alpha_decay=0.9,
                 gamma=0.99,
                 reward_func=None):
        self.env = env
        self.bins = bins
        self.episodes = episodes
        self.num_decay = num_decay
        self.exploration_rate = exploration_rate
        self.exploration_rate_base = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.alpha = alpha
        self.alpha_base = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.num_action = env.action_space.n
        self.Q = {}
        self.reward_func = reward_func

    def set_env(self, env):
        self.env = env

    def discretization(self, value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    def to_state(self, obs):
        # x, theta, v, omega = obs
        # c_p, c_v, p_a, p_v = obs
        state = []
        for i, o in enumerate(obs):
            state.append(self.discretization(o, self.bins[i]))
        state = tuple(state)
        # state = (self.to_bins(c_p, CART_POS),
        #          self.to_bins(c_v, CART_VEL),
        #          self.to_bins(p_a, POLE_ANGLE),
        #          self.to_bins(p_v, ANG_RATE))
        # print('state = \n', state)2
        # state = (to_bins(c_p, CART_POS), to_bins(p_a, POLE_ANGLE))
        # state = to_bins(p_a, POLE_ANGLE)
        # print('POLE_ANGLE = ', state)
        return state

    def get_action(self, state, mode='greedy'):
        p = np.random.uniform(0, 1)
        # print p
        if p < self.exploration_rate and mode == 'greedy':
            # print('greedy')
            return random.choice([i for i in range(self.num_action)])
        x = []
        for action in range(self.num_action):
            if (state, action) not in self.Q:
                self.Q[(state, action)] = 0
            x.append(self.Q[(state, action)])
        return np.argmax(x)

    def train(self,
              train_episodes=None,
              test_interval=1,
              verbose=True):
        if not train_episodes:
            train_episodes = self.episodes
        timesteps = np.zeros((train_episodes//test_interval, 2))
        for episode in range(train_episodes):
            # agent = QLearning()
            obs = self.env.reset()
            state = self.to_state(obs)
            timestep = 0
            while True:
                # env.render()
                timestep += 1
                action = self.get_action(state)
                obs, reward, done, _ = self.env.step(action)
                # total_rewards += reward
                if self.reward_func:
                    reward = self.reward_func(obs, self.env)
                '''
                x, x_dot, theta, theta_dot = obs
                x_threshold = self.env.observation_space.high[0]
                theta_threshold_radians = self.env.observation_space.high[2]
                r1 = (x_threshold - abs(x)) / x_threshold-0.8
                r2 = (theta_threshold_radians - abs(theta)) / theta_threshold_radians - 0.5
                reward = r1+r2
                '''
                state_prime = self.to_state(obs)
                action_prime = self.get_action(state_prime, 'non_greedy')

                if (state_prime, action_prime) not in self.Q:
                    self.Q[(state_prime, action_prime)] = 0.1 * np.random.randn(1)
                    # Q[(state_prime, action_prime)] = 0
                if (state, action) not in self.Q:
                    self.Q[(state, action)] = 0.1 * np.random.randn(1)
                    # Q[(state, action)] = 0

                # Q[(state, action)] = (1 - ALPHA) * Q[(state, action)] + ALPHA * (
                #     reward + GAMMA * Q[(state_prime, action_prime)])
                self.Q[(state, action)] += self.alpha * (reward +
                    self.gamma * self.Q[(state_prime, action_prime)] -
                    self.Q[(state, action)])
                state = state_prime

                if done:
                    if verbose:
                        print("train:Episode %d completed in %d" % (episode, timestep))
                    break
            if episode % test_interval == 0:
                test_timesteps = self.test(1, True)
                # timesteps[episode // test_interval, :] = episode, test_timesteps[0, 1]
                timesteps[episode // test_interval, :] = episode, test_timesteps[0, 1]
                # timesteps[episode // 100, 1] = test_timesteps[0]
            
            if self.num_decay:
                decay_segment_length = train_episodes // self.num_decay
                self.alpha = self.alpha_base * (self.alpha_decay**(episode // decay_segment_length))
                self.exploration_rate = self.exploration_rate_base * (self.exploration_rate_decay **
                                                    (episode // decay_segment_length)) 
            '''
            # test
            obs = self.env.reset()
            cnt = 0
            avg = 0
            while True:
                # env.render()
                state = self.to_state(obs)
                action = self.get_action(state, 'non_greedy')
                obs, reward, done, _ = self.env.step(action)
                cnt += 1
                # if cnt > 500000 and cnt % 500000 == 0:
                    # print('wasai!! cnt = ', cnt)
                if done:
                    avg += cnt
                    timesteps[episode] = cnt
                    if verbose:
                        print("test :Episode %d completed in %d" % (episode, cnt))
                    break
            '''
        return timesteps

    def test(self, test_episodes=100, verbose=True):
        timesteps = np.zeros((test_episodes, 2), dtype=int)
        avg = 0
        for episode in range(test_episodes):
            cnt = 0
            obs = self.env.reset()
            total_rewards = 0
            while True:
                # env.render()
                cnt += 1
                state = self.to_state(obs)
                action = self.get_action(state, 'non_greedy')
                obs, reward, done, _ = self.env.step(action)
                total_rewards += reward
                # if cnt > 500000 and cnt % 500000 == 0:
                    # print('wasai!! cnt = ', cnt)
                if done:
                    avg += cnt
                    timesteps[episode, :] = episode, total_rewards
                    if verbose:
                        print("test :Episode %d completed in %d" % (episode, cnt))
                    break
        return timesteps
    
    def visualize(self, pic_name, timesteps, mode='train'):
        fig, ax = plt.subplots()
        ax.plot(timesteps[:, 0], timesteps[:, 1])
        title = 'rewards over episodes during training'
        if mode != 'train':
            title = 'rewards over episodes during test'
        ax.set(xlabel='epidode',
               ylabel='rewards',
               title=title)
        ax.grid()
        fig.savefig(pic_name)
        plt.show()

    def save_Qtable(self, filename):
        f = open(filename, 'wb')
        pkl.dump(self.Q, f, 3)
    # def infor(self):
    #     # 最大值，平均值，方差
    #     return (np.max(self.timesteps),
    #             np.mean(self.timesteps),
    #             np.std(self.timesteps))
        
# print('average timesteps = ', avg/NUM_EPISODES)
# print('max timestep = ', np.max(timesteps))
# print('means = {}, std = {}'.format(np.mean(timesteps), np.std(timesteps)))
