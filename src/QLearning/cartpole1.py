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
from gym.envs.registration import registry, register, make
from MyQLearning import QLearning
import pickle as pkl


NUM_EPISODES = 10000
MAX_T = 20010
ALPHA = 0.3
GAMMA = 0.99
NUM_DECAY = 10
EXPLORATION_RATE = 0.6
EXPLORATION_RATE_DECAY = 0.9

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=20000,
    # reward_threshold=195.0,
)


def reward_func(obs, env):
    x, x_dot, theta, theta_dot = obs
    x_threshold = env.observation_space.high[0]
    theta_threshold_radians = env.observation_space.high[2]
    r1 = (x_threshold - abs(x)) / x_threshold-0.8
    r2 = (theta_threshold_radians - abs(theta)) / theta_threshold_radians - 0.5
    return r1+r2


def cartpole():
    # CART_POS = np.linspace(-2.4, 2.4, 12)
    # CART_VEL = np.linspace(-1, 1, 5)
    # POLE_ANGLE = np.linspace(-0.3, 0.3, 24)
    # ANG_RATE = np.linspace(-1, 1, 5)
    CART_POS = np.linspace(-2.4, 2.4, 10)
    CART_VEL = np.linspace(-1, 1, 5)
    POLE_ANGLE = np.linspace(-0.3, 0.3, 24)
    ANG_RATE = np.linspace(-1.5, 1.5, 6)
    env = gym.make('CartPole-v2')
    bins = (CART_POS, CART_VEL, POLE_ANGLE, ANG_RATE)
    qlearning = QLearning(env,
                          bins,
                          num_decay=NUM_DECAY,
                          exploration_rate=EXPLORATION_RATE,
                          exploration_rate_decay=EXPLORATION_RATE_DECAY,
                          alpha=ALPHA,
                          gamma=GAMMA,
                          reward_func=reward_func)
    train_timesteps = qlearning.train(train_episodes=NUM_EPISODES, test_interval=20)
    test_evn = gym.wrappers.Monitor(env, './videos/CartPole-v1', force=True)
    qlearning.set_env(test_evn)
    test_timesteps = qlearning.test(test_episodes=20)
    qlearning.visualize('train_cartpole.png', train_timesteps, mode='train')
    qlearning.visualize('test_cartpole.png', test_timesteps, mode='test')
    print('max rewards = ', np.max(test_timesteps[:, 1]))
    print('means = {}, std = {}'.format(np.mean(test_timesteps[:, 1]), np.std(test_timesteps[:, 1])))
    if np.mean(test_timesteps) > 10000:
        qlearning.save_Qtable('./QLearning/cartpole_Q.pkl')
    print('alpha, exploration_rate', qlearning.alpha, qlearning.exploration_rate)


def main():
    cartpole()


if __name__ == '__main__':
    main()