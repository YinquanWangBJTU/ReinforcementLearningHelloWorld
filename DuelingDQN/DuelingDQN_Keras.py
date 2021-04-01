import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import gym
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, 32)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample)
        )
        states = np.array(states).reshape(32, -1)
        next_states = np.array(next_states).reshape(32, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class DuelingDQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.001
        self.opt = tf.keras.optimizers.RMSprop(self.lr)
        self.loss = tf.keras.losses.MeanSquaredError()

    def create_model(self):
        input_state = Input((self.state_dim, ))
        h1 = Dense(32, activation='relu')(input_state)
        h2 = Dense(32, activation='relu')(h1)
        outputs = []
        state_value = Dense(1, activation=None)(h2)
        action_value = Dense(self.action_dim, activation=None)(h2)
        outputs.append(state_value)
        outputs.append(action_value)
        return tf.keras.Model(input_state, outputs)

