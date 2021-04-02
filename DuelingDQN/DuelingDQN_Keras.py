import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from collections import deque
import gym
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
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim, ))
        h1 = Dense(32, activation='relu')(input_state)
        h2 = Dense(32, activation='relu')(h1)
        state_value = Dense(1, activation=None)(h2)
        action_value = Dense(self.action_dim, activation=None)(h2)
        outputs = state_value + (action_value - tf.reduce_mean(action_value, axis=1, keepdims=True))
        return tf.keras.Model(input_state, outputs)

    def predict(self, state):
        return self.model.predict(state)

    def train(self, x, y):
        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            q_eval = self.model(x)
            loss = self.loss(q_eval, y)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action(self, state, step):
        state = state[np.newaxis, :]
        eps = 1. / ((step / 10) + 1)
        if np.random.uniform() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)
    
    def get_optimal_action(self, state):
        q_eval = self.predict(state)
        action = np.argmax(q_eval)
        return action


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.gamma = 0.9
        self.buffer = ReplayBuffer(capacity=1000)
        self.q = DuelingDQN(state_dim=self.state_dim, action_dim=self.action_dim)
        self.q_target = DuelingDQN(state_dim=self.state_dim, action_dim=self.action_dim)
    
    def replace_op(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)
    
    def learn(self):
        states, actions, rewards, next_states, done = self.buffer.sample()

        q_next = self.q_target.predict(next_states)
        q_eval = self.q.predict(states)

        q_target = np.array(q_eval).copy()
        actions = np.array(actions).astype(int)
        q_target[range(32), actions] = rewards + self.gamma * np.max(q_next, axis=1)
        self.q.train(states, q_target)


def run():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    agent = Agent(env)
    for i in range(200):
        total_steps = 0
        total_rewards = 0
        state = agent.env.reset()
        while True:
            action = agent.q.get_action(state, total_steps)
            next_state, reward, done, info = agent.env.step(action)
            agent.buffer.store_transition(state, action, reward, next_state, done)

            if agent.buffer.size() > 100:
                agent.learn()

            if total_steps % 5 == 0:
                agent.replace_op()

            state = next_state
            total_steps += 1
            total_rewards += reward

            if done:
                print('EP{} reward={}'.format(i, total_rewards))
                break


if __name__ == '__main__':
    run()