import gym
from ReinforcementLearning.Dueling_DQN.PRIORITIZED_EXPERIENCE_REPLAY.Prioritized_experience_replay\
    import PrioritizedExperienceReplay
from ReinforcementLearning.Dueling_DQN.PRIORITIZED_EXPERIENCE_REPLAY.DoubleDQN import DoubleDQN
import numpy as np

env = gym.make('MountainCar-v0')
env = env.unwrapped

MemorySize = 10000

per = PrioritizedExperienceReplay(memory_size=MemorySize, batch_size=32, n_features=env.observation_space.shape[0])
d_dqn = DoubleDQN(n_features=env.observation_space.shape[0], n_actions=env.action_space.n)

step_list = []
total_step = 0
for i in range(300):
    total_reward = 0
    step = 0
    observation = env.reset()
    while True:
        action = d_dqn.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if done: reward = 10
        total_reward += 0.9 * reward

        single_q_next = d_dqn.sess.run(d_dqn.q_next, feed_dict={d_dqn.state_: observation_[np.newaxis, :]})
        single_q_eval, single_q_eval4next = d_dqn.sess.run([d_dqn.q_eval, d_dqn.q_eval],
                                                           feed_dict={d_dqn.state_: observation_[np.newaxis, :],
                                                                      d_dqn.state: observation[np.newaxis, :]})
        action_index = np.argmax(single_q_eval4next[0])

        single_q_target = single_q_next[0][action_index]
        td_error = abs(reward + 0.9 * single_q_target - single_q_eval[0][action])

        per.store_transition(observation, action, reward, observation_, td_error)

        if total_step > MemorySize:
            sample, sample_index = per.sample()
            sample = d_dqn.learn(sample)
            per.memory[sample_index, :] = sample

        if done:
            print('episode:{}, reward:{}, step:{}'.format(i, total_reward, step))
            break

        total_step += 1
        step += 1
        observation = observation_
