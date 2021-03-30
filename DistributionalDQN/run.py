import gym
from ReinforcementLearning.Dueling_DQN.DistributionalDQN.DistributionalDQNMyself import DistributionalDQN
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

DisDQN = DistributionalDQN(
    n_features=observation_space,
    n_actions=action_space,
    learning_rate=0.001,
    gamma=0.9,
    e_greedy=0.9,
    v_max=100,
    v_min=0,
    atoms=11
)

reward_list = []
for i in range(2000):
    observation = env.reset()
    total_reward = 0
    total_step = 0
    while True:
        action = DisDQN.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        DisDQN.store_transition(observation, action, reward, observation_)

        if DisDQN.memory_count >= DisDQN.memory_size:
            DisDQN.learn()

        total_reward += reward
        total_step += 1

        if done:
            reward_list.append(total_reward)
            print('Episode: {} | Reward: {} | Step: {}'.format(i, total_reward, total_step))
            break


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(len(reward_list)), reward_list)
plt.show()
