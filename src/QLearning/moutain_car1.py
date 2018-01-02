import gym
import numpy as np
import random
import math
from gym.envs.registration import registry, register, make
from MyQLearning import QLearning
import pickle as pkl


NUM_EPISODES = 50000
# NUM_EPISODES = 5
TEST_EPISODES = 100
# TEST_EPISODES = 2
MAX_T = 2010
# ALPHA_BASE = 0.1
ALPHA_BASE = 0.05
ALPHA = ALPHA_BASE
ALPHA_DECAY_RATE = 0.9
GAMMA = 0.99

EXPLORATION_RATE_BASE = 0.6
EXPLORATION_RATE = EXPLORATION_RATE_BASE
EXPLORATION_RATE_DECAY = 0.95
NUM_DECAY = 5

register(
    id='MountainCar-v1',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=2000,
    # reward_threshold=-110.0,
)


def mountain_car():
    env = gym.make('MountainCar-v1')
    test_evn = gym.wrappers.Monitor(env, './videos/MountainCar-v1', force=True)
    NUM_ACTIONS = env.action_space.n
    CART_POS = np.linspace(-1.2, 0.6, 10)
    CART_VEL = np.linspace(-0.07, 0.07, 8)
    # CART_POS = np.linspace(-1.2, 0.6, 18)
    # CART_VEL = np.linspace(-0.07, 0.07, 10)
    bins = (CART_POS, CART_VEL)
    qlearning = QLearning(env,
                          bins,
                          num_decay=NUM_DECAY,
                          exploration_rate=EXPLORATION_RATE,
                          exploration_rate_decay=EXPLORATION_RATE_DECAY,
                          alpha=ALPHA,
                          gamma=GAMMA)
    test_interval = 50
    train_timesteps = qlearning.train(train_episodes=NUM_EPISODES, test_interval=test_interval)
    test_evn = gym.wrappers.Monitor(env, './videos/MountainCar-v1', force=True)
    qlearning.set_env(test_evn)
    test_timesteps = qlearning.test(test_episodes=100)
    qlearning.visualize('train_mountain.png', train_timesteps, mode='train')
    qlearning.visualize('test_mountain.png', test_timesteps, mode='test')
    print('max rewards = ', np.max(test_timesteps[:, 1]))
    print('means = {}, std = {}'.format(np.mean(test_timesteps[:, 1]), np.std(test_timesteps[:, 1])))
    if np.mean(test_timesteps) < 100:
        qlearning.save_Qtable('./QLearning/Mountain_Q.pkl')


def main():
    mountain_car()


if __name__ == '__main__':
    main()
