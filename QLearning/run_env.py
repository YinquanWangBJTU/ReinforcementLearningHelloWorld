from Q_learning.env import ENV
from Q_learning.Q_learning_myself_1 import QLearning
import matplotlib.pyplot as plt

RENDER = False
env = ENV(grid_number=7, origin_position=1)

action_space = env.action_space
action_dim = env.action_dim
environment_space = env.environment_space
state_space = env.state_dim
destination_position = env.destination_position


q_learning = QLearning(action_list=action_space, environment_list=environment_space,
                       destination_position=destination_position, learning_rate=0.01,
                       reward_decay=0.9, e_greedy=0.9)

count_list = []
for i in range(1000):
    observation = env.reset()
    print(q_learning.epsilon)
    while True:
        action = q_learning.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        q_learning.learn(observation, action, reward, observation_)

        if done:
            print('总迭代次数:', env.step_counter)
            count_list.append(env.step_counter)
            break

        if RENDER is True: env.render()

        observation = observation_

plt.plot(range(len(count_list)), count_list)
plt.xlabel('Episode')
plt.ylabel('Step count')
plt.show()