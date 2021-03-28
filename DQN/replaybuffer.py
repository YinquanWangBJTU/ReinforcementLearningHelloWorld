import numpy as np


class ReplayBuffer:
    def __init__(self, memory_size, n_features):
        self.memory_size = memory_size
        self.memory = np.zeros(shape=(self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

    def store_transition(self, observation, action, reward, observation_):
        transition = np.hstack((observation, action, reward, observation_))
        store_index = self.memory_counter % self.memory_size
        self.memory[store_index, :] = transition
        self.memory_counter += 1

    def sample(self, batch_size):
        if self.memory_counter > self.memory_size:
            batch_index = np.random.choice(self.memory_size, size=batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, size=batch_size)
        sample = self.memory[batch_index, :]
        return sample, batch_size
