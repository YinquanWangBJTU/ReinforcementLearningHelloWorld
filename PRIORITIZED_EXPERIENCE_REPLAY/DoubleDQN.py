import tensorflow as tf
import numpy as np


class DoubleDQN:
    def __init__(self, n_features, n_actions, learning_rate=0.001, e_greedy=0.9,
                e_greedy_increment=0.01, gamma=0.9):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy
        self.e_greedy_increment = e_greedy_increment
        self.epsilon = 0 if self.e_greedy_increment is not None else e_greedy

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        e_params = tf.get_collection('eval_params_collections')
        t_params = tf.get_collection('target_params_collections')

        self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.train_counter = 0
        self.replace_iter = 10

    def _build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')
        self.weights = tf.placeholder(tf.float32, [None, 1], name='weights')
        w_init = tf.random_normal_initializer(0.0, 0.3)
        b_init = tf.constant_initializer(0.5)
        c_names = [tf.GraphKeys.GLOBAL_VARIABLES, 'eval_params_collections']
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 20], initializer=w_init, collections=c_names)
                b1 = tf.get_variable('b1', [1, 20], initializer=b_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [20, self.n_actions], initializer=w_init, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('loss'):
                self.td_error = tf.reduce_sum(tf.abs(self.q_target-self.q_eval), axis=1)
                self.loss = tf.reduce_mean(self.weights * self.td_error)

            with tf.variable_scope('train'):
                self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        c_names = [tf.GraphKeys.GLOBAL_VARIABLES, 'target_params_collections']
        self.state_ = tf.placeholder(tf.float32, [None, self.n_features], name='state_')
        with tf.variable_scope('target_net'):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 20], collections=c_names)
                b1 = tf.get_variable('b1', [1, 20], collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [20, self.n_actions], collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.choice(range(self.n_actions))
        return action

    def learn(self, sample):
        if self.train_counter % self.replace_iter == 0:
            self.sess.run(self.replace_op)

        sample_observation = sample[:, :self.n_features]
        sample_observation_ = sample[:, -(self.n_features+1): -1]
        sample_action = sample[:, self.n_features].astype(int)
        sample_reward = sample[:, self.n_features + 1]
        sample_weights = sample[:, -1]

        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval], feed_dict={self.state_: sample_observation_,
                                                                                   self.state: sample_observation_})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.state: sample_observation})
        q_target = q_eval.copy()

        max_eval_action = np.argmax(q_eval4next, axis=1)
        batch_index = np.arange(32, dtype=np.int32)

        q_target[batch_index, sample_action] = sample_reward + self.gamma * q_next[batch_index, max_eval_action]

        self.sess.run([self.train, self.loss], feed_dict={self.state: sample_observation,
                                                          self.q_target: q_target,
                                                          self.weights: np.array(sample_weights).reshape(-1, 1)})

        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.train_counter += 1

        td_error = self.sess.run(self.td_error, feed_dict={self.state: sample_observation,
                                                           self.q_target: q_target})

        sample[:, -1] = td_error
        return sample
