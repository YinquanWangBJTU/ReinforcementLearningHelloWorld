from ReinforcementLearning.Q_Network.Q_Network_myself import DeepQNetwork
from ReinforcementLearning.Q_Network.maze_env import Maze
from ReinforcementLearning.Q_Network.replaybuffer import ReplayBuffer
import os
import tensorflow as tf
import shutil


def run_maze():
    step = 0
    for episode in range(300):  # 迭代300次
        # initial observation
        observation = env.reset()  # 获取初始状态

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            replay_buffer.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                sample, batch_size = replay_buffer.sample(32)
                RL.learn(sample, batch_size=batch_size)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    observation_space = env.n_features
    action_space = env.n_actions

    replay_buffer = ReplayBuffer(memory_size=500, n_features=observation_space)
    memory_size = replay_buffer.memory_size

    RL = q_network = DeepQNetwork(observation_space, action_space)

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()

    OUTPUT_GRAPH = True
    LOG_DIR = './log'

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, RL.sess.graph)

