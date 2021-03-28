import tensorflow as tf
import numpy as np


class PrioritizedExperienceReplay:
    def __init__(self, memory_size, batch_size, n_features, alpha=1, beta=1):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.n_features = n_features

        self.memory = np.zeros(shape=(memory_size, n_features * 2 + 2))

        self.memory_count = 0

    def store_transition(self, observation, action, reward, observation_, td_error):
        transition = np.hstack((observation, action, reward, observation_, td_error))
        memory_index = self.memory_count % self.memory_size
        self.memory[memory_index, :] = transition

    def sample(self):
        sample_probs = self.memory[:, -1] / np.sum(self.memory[:, -1])
        sample_index = np.random.choice(size=self.batch_size, p=)