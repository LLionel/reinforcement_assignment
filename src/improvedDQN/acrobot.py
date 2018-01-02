import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import gym
import random
from collections import deque
from gym.envs.registration import registry, register, make, spec
from improvedDQN import DQN
import matplotlib.pyplot as plt
import pickle as P
# import tensorflow as tf

register(
    id='Acrobot-v2',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=2000,
    # reward_threshold=-110.0,
)



# env = gym.make('CartPole-v1')

# Hyper Parameters for DQN
GAMMA = 0.99  # discount factor for target Q
EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
REPLAY_SIZE = 500  # experience replay buffer size
BATCH_SIZE = 64  # size of minibatch
LEARNING_RATE = 0.02
STEP = 2001
# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'Acrobot-v2'
TRAIN_EPISODE = 3000  # Episode limitation
TEST_EPISODE = 20
TEST = 10  # The number of experiment test every 100 episode
# set random seed
# SEED = 7
# torch.manual_seed(SEED)


def visualize(rewards, pic_name):
    fig, ax = plt.subplots()
    ax.plot([i+1 for i in range(len(rewards))], rewards)
    title = 'training error w.r.t. training numbers'
    ax.set(xlabel='training numbers',
           ylabel='average rewards',
           title=title)
    ax.grid()
    fig.savefig(pic_name)
    plt.show()


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env,
                ENV_NAME,
                EPSILON,
                LEARNING_RATE,
                GAMMA,
                batch_size=BATCH_SIZE,
                train_delay=16)
    # training_test_rewards = np.zeros(NUM_ITERATION, 2)
    # for i in range(NUM_ITERATION):
    training_test_rewards = []
    for episode in range(TRAIN_EPISODE):
        # initialize task
        # print('episode {}'.format(episode))
        # print('*'*20)
        state = env.reset()
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            '''
            x, x_dot, theta, theta_dot = state
            x_threshold = env.observation_space.high[0]
            theta_threshold_radians = env.observation_space.high[2]
            r1 = (x_threshold - abs(x)) / x_threshold - 0.8
            r2 = (theta_threshold_radians - abs(theta)
                ) / theta_threshold_radians - 0.5
            reward = r1 + r2
            '''
            # Define reward for agent
            # reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                # print('TRAIN:timesteps={}'.format(step))
                break
        
        # Test every 100 episodes
        if episode % 10 == 0:
            total_reward = 0
            # for i in range(TEST):
            state = env.reset()
            timestep = 0
            while True:
                env.render()
                timestep += 1
                action = agent.get_action(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                # x, x_dot, theta, theta_dot = state
                # x_threshold = env.observation_space.high[0]
                # theta_threshold_radians = env.observation_space.high[2]
                # r1 = (x_threshold - abs(x)) / x_threshold - 0.8
                # r2 = (theta_threshold_radians - abs(theta)
                #     ) / theta_threshold_radians - 0.5
                # reward = r1 + r2
                total_reward += reward
                if done:
                    training_test_rewards.append(total_reward)
                    print('TEST :timesteps={}'.format(timestep))
                    break
            # ave_reward = total_reward / TEST
            # print('episode: {}, Evaluation Average Reward:{}'
            #     .format(episode, ave_reward))
            # if ave_reward >= 200:
            #     break
        
    # save results for uploading
    # env.monitor.start('DQN/videos/CartPole-v1', force=True)
    test_rewards = []
    test_evn = gym.wrappers.Monitor(
        env, './videos/improvedDQN_acrobot-v0', force=True)
    for episode in range(TEST_EPISODE):
        total_reward = 0
        state = test_evn.reset()
        timestep = 0
        while True:
            test_evn.render()
            timestep += 1
            action = agent.get_action(state)  # direct action for test
            state, reward, done, _ = test_evn.step(action)
            total_reward += reward
            if done:
                # test_rewards[episode, :] = episode, total_reward
                test_rewards.append(total_reward)
                print('TEST:episode: {}, timesteps: {}, reward:{}'.format(episode, timestep, total_reward))
                break
    test_evn.close()
    print('max rewards = ', np.max(test_rewards))
    print('means = {}, std = {}'.format(np.mean(test_rewards), np.std(test_rewards)))
    agent.visualize_loss('improvedDQN_acrobot_loss')
    visualize(training_test_rewards, 'improvedDQN_acrobot_training_test_rewards.png')
    visualize(test_rewards, 'improvedDQN_acrobot_test_rewards.png')
    f = open('./improvedDQN/improvedDQN_acrobot_test_reward.pkl',  'wb')
    P.dump(test_rewards, f, 3)
    f.close()
    f = open('./improvedDQN/improvedDQN_acrobot_training_test_reward.pkl',  'wb')
    P.dump(training_test_rewards, f, 3)
    f.close()


if __name__ == '__main__':
    main()
