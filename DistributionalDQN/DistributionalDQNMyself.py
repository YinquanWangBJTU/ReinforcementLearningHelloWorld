import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Softmax, Reshape
from tensorflow.keras.optimizers import Adam
import gym
import random
import numpy as np
from collections import deque
import math


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, state_, done):
        self.buffer.append([state, action, reward, state_, done])

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, action, rewards, states_, dones = map(
            np.asarray, zip(*sample)
        )
        states = np.array(states).reshape(batch_size, -1)
        states_ = np.array(states_).reshape(batch_size, -1)
        return states, action, rewards, states_, dones

    def size(self):
        return len(self.buffer)


class ActionValueModel:
    def __init__(self, state_dim, action_dim, z, atoms):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z = z
        self.atoms = atoms

        self.opt = tf.keras.optimizers.Adam(0.0001)
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim, ))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        return tf.keras.Model(input_state, outputs)

    def train(self, x, y):
        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)


class Agent:
    def __init__(self, env, v_max, v_min, atoms, gamma):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.buffer = ReplayBuffer()
        self.batch_size = 8
        self.v_max = v_max
        self.v_min = v_min
        self.atoms = atoms
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.gamma = gamma
        self.q = ActionValueModel(state_dim=self.state_dim, action_dim=self.action_dim,
                                  z=self.z, atoms=self.atoms)
        self.q_target = ActionValueModel(state_dim=self.state_dim, action_dim=self.action_dim,
                                         z=self.z, atoms=self.atoms)
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, states_, dones = self.buffer.sample(self.batch_size)
        z = self.q.predict(states_)
        z_ = self.q_target.predict(states_)

        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order='F')
        action_ = np.argmax(q, axis=1)
        m_prob = [np.zeros(shape=(self.batch_size, self.atoms)) for i in range(self.action_dim)]
        for i in range(self.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(self.atoms):
                    Tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(l)] += z_[action_[i]][i][int(l)] * (u - bj)
                    m_prob[actions[i]][i][int(u)] += z_[action_[i]][i][int(u)] * (bj - l)
        self.q.train(states, m_prob)

    def train(self):
        for i in range(500):
            done, total_reward, steps = False, 0, 0
            state = self.env.reset()
            while not done:
                action = self.q.get_action(state, steps)
                next_state, reward, done, info = self.env.step(action)

                self.buffer.store_transition(state, action, -1 if done else 0, next_state, done)

                if self.buffer.size() > 1000:
                    self.replay()

                if steps % 5 == 0:
                    self.target_update()

                state = next_state
                steps += 1
                total_reward += reward

            print('Episode: {}| Reward: {}| Steps: {}'.format(i, total_reward, steps))


def run():
    env = gym.make('CartPole-v0')
    agent = Agent(env=env, v_max=5, v_min=-5, atoms=8, gamma=0.99)
    agent.train()


if __name__ == '__main__':
    run()