import gym
import numpy as np
import random
import math
from gym.envs.registration import registry, register, make
from MyQLearning import QLearning
import pickle as pkl


NUM_EPISODES = 5000
# NUM_EPISODES = 5
TEST_EPISODES = 100
# TEST_EPISODES = 2
MAX_T = 2010
# ALPHA_BASE = 0.1
ALPHA_BASE = 0.2
ALPHA = ALPHA_BASE
ALPHA_DECAY_RATE = 0.8
GAMMA = 0.99

EXPLORATION_RATE_BASE = 0.4
EXPLORATION_RATE = EXPLORATION_RATE_BASE
EXPLORATION_RATE_DECAY = 0.95
NUM_DECAY = 10

register(
    id='Acrobot-v2',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=2000,
    # reward_threshold=-110.0,
)


def mountain_car():
    env = gym.make('Acrobot-v2')
    
    NUM_ACTIONS = env.action_space.n
    obs1 = np.linspace(-1, 1, 10)
    obs2 = np.linspace(-1, 1, 10)
    obs3 = np.linspace(-1, 1, 10)
    obs4 = np.linspace(-1, 1, 10)
    obs5 = np.linspace(-12.56637061, 12.56637061, 25)
    obs6 = np.linspace(-28.27433388, 28.27433388, 56)
    bins = (obs1, obs2, obs3, obs4, obs5, obs6)
    qlearning = QLearning(env,
                          bins,
                        #   num_decay=NUM_DECAY,
                          exploration_rate=EXPLORATION_RATE,
                          exploration_rate_decay=EXPLORATION_RATE_DECAY,
                          alpha=ALPHA,
                          gamma=GAMMA)
    train_timesteps = qlearning.train(train_episodes=NUM_EPISODES, test_interval=10)
    test_evn = gym.wrappers.Monitor(env, './videos/Acrobot-v1', force=True)
    qlearning.set_env(test_evn)
    test_timesteps = qlearning.test(test_episodes=100)
    qlearning.visualize('train_acrobot.png', train_timesteps, mode='train')
    qlearning.visualize('test_acrobot.png', test_timesteps, mode='test')
    print('max rewards = ', np.max(test_timesteps[:, 1]))
    print('means = {}, std = {}'.format(np.mean(test_timesteps[:, 1]), np.std(test_timesteps[:, 1])))
    if np.mean(test_timesteps) < 100:
        qlearning.save_Qtable('./QLearning/Acrobot-v1.pkl')


def main():
    mountain_car()


if __name__ == '__main__':
    main()
