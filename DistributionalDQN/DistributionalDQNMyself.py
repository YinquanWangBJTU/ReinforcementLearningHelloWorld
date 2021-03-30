import tensorflow as tf
import numpy as np
import math


class DistributionalDQN:
    def __init__(self, n_features, n_actions, learning_rate, gamma, e_greedy, v_max, v_min, atoms):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.v_max = v_max
        self.v_min = v_min
        self.atoms = atoms
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.z = tf.cast(np.array(self.z), tf.float64)

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.memory_size = 500
        self.memory = np.zeros(shape=(self.memory_size, n_features * 2 + 2))
        self.memory_count = 0

        self.batch_size = 32

        e_params = tf.get_collection('eval_net')
        t_params = tf.get_collection('target_net')
        self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.replace_count = 100
        self.train_count = 0

    def _build_net(self):
        self.state = tf.placeholder(tf.float64, [None, self.n_features], name='state')
        self.action = tf.placeholder(tf.float64, [None, 1], name='action')
        self.m_input = tf.placeholder(tf.float64, [None, self.atoms], name='m_input')
        w_init = tf.random_normal_initializer(0.0, 0.3)
        b_init = tf.constant_initializer(0.5)
        with tf.variable_scope('eval_net'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=24,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                activation=tf.nn.relu,
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=24,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                activation=tf.nn.relu,
                name='l2'
            )
            with tf.variable_scope('distribution'):
                self.distribution = tf.layers.dense(
                    inputs=l2,
                    units=self.atoms,
                    kernel_initializer=w_init,
                    bias_initializer=b_init,
                    activation=None,
                    name='action_distribution'
                )

            with tf.variable_scope('loss'):
                self.loss = -tf.reduce_sum(self.m_input * tf.log(self.distribution))

            with tf.variable_scope('train'):
                self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            with tf.variable_scope('Q_eval'):
                self.Q_eval = tf.reduce_sum(self.z * self.distribution)

        self.state_ = tf.placeholder(tf.float64, [None, self.n_features], name='state_')
        with tf.variable_scope('target_net'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=24,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                activation=tf.nn.relu,
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=24,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                activation=tf.nn.relu,
                name='l2'
            )
            with tf.variable_scope('distribution_target'):
                self.distribution_target = tf.layers.dense(
                        inputs=l2,
                        units=self.atoms,
                        kernel_initializer=w_init,
                        bias_initializer=b_init,
                        activation=None,
                        name='action_distribution'
                    )
            with tf.variable_scope('Q_target'):
                self.Q_target = tf.reduce_sum(self.z * self.distribution_target)

    def choose_action(self, observation):
        if np.random.uniform() <= self.epsilon:
            Q_value = [self.sess.run(self.Q_eval, feed_dict={self.state: observation[np.newaxis, :]})
                       for a in range(self.n_actions)]
            action = np.argmax(Q_value)
            return action
        else:
            action = np.random.randint(0, self.n_actions)
            return action

    def learn(self):
        if self.train_count % self.replace_count == 0:
            self.sess.run(self.replace_op)

        sample = self.sample()
        observation = sample[:, :self.n_features]
        action = sample[:, self.n_features]
        reward = sample[:, self.n_features + 1]
        observation_ = sample[:, -self.n_features:]

        list_Q = [self.sess.run(self.Q_eval, feed_dict={self.state: observation_})
                  for a in range(self.n_actions)]
        print(list_Q)
        list_Q = np.vstack(list_Q)
        q = np.sum(np.multiply(list_Q, np.array(self.z)), axis=1)
        print(q.shape)
        q = q.reshape((self.batch_size, self.n_actions), order='F')
        print(q)
        action_ = np.argmax(q, axis=1)
        print(action_)

        m = np.zeros(shape=(self.batch_size, self.atoms))
        p = self.sess.run(self.distribution, feed_dict={self.state: observation_[np.newaxis, :],
                                                        self.action: [[action_]]})

        for i in range(self.atoms):
            Tz = min(self.v_max, max(self.v_min, reward + self.gamma * self.z[i]))

            bj = (Tz - self.v_min) / self.delta_z

            l, u = math.floor(bj), math.ceil(bj)
            pj = p[i]

            m[int(l)] += pj * (u - bj)
            m[int(u)] += pj * (bj - l)
        self.sess.run(self.train_op, feed_dict={self.state: observation[np.newaxis, :], self.action: [[action]],
                                                self.m_input: [m]})
        self.train_count += 0

    def store_transition(self, observation, action, reward, observation_):
        transition = np.hstack((observation, action, reward, observation_))
        memory_index = self.memory_count % self.memory_size
        self.memory[memory_index, :] = transition
        self.memory_count += 1

    def sample(self):
        batch_index = np.random.randint(low=0, high=self.memory_size, size=self.batch_size)
        sample = self.memory[batch_index, :]
        return sample
