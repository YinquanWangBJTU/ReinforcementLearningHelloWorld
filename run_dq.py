import gym
from ReinforcementLearning.Dueling_DQN.DuelingDQN import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 25
N_F = env.observation_space.shape[0]
N_A = 25

d_dqn = DuelingDQN(n_features=N_F, n_actions=N_A, memory_size=MEMORY_SIZE)


def train(RL):
    total_steps = 0
    acc_r = [0]
    observation = env.reset()
    while True:
        # if total_steps-MEMORY_SIZE > 9000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10      # normalize to a range of (-1, 0)
        acc_r.append(reward + acc_r[-1])
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps-MEMORY_SIZE > 15000:
            break

        observation = observation_
        total_steps += 1
    return RL.cost_hist, acc_r


cost, reward_list = train(d_dqn)
plt.plot(range(len(cost)), cost)
plt.show()

plt.plot(range(len(reward_list)), reward_list)
plt.show()