import pandas as pd
import numpy as np


class QLearning:
    def __init__(self, action_list, environment_list, destination_position,
                 learning_rate, reward_decay, e_greedy, e_greedy_increment=0.001):
        self.action_list = action_list
        self.action_dim = len(action_list)
        self.environment_list = environment_list
        self.destination_position = destination_position
        self.e_greedy_increment = e_greedy_increment

        self.state_dim = len(environment_list)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(index=environment_list,
                                    data=np.zeros(shape=(self.state_dim, self.action_dim)),
                                    columns=self.action_list)

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            action_index = np.argmax(self.q_table.loc[observation, :])
            action = self.action_list[action_index]
        else:
            action = np.random.choice(self.action_list)
        return action

    def learn(self, observation, action, reward, observation_):
        q_predict = self.q_table.loc[observation, action]
        if observation_ == self.destination_position:
            q_target = reward
        else:
            q_target = reward + self.gamma * self.q_table.loc[observation_, :].max()
        self.q_table.loc[observation, action] = self.q_table.loc[observation, action] + self.lr * (q_target - q_predict)

        self.epsilon += self.e_greedy_increment
        print(self.q_table)
        # print(self.q_table)

