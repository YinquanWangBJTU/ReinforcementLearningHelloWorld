import tensorflow as tf
import numpy as np


class DuelingDQN:
    def __init__(self, n_features, n_actions, learning_rate=0.001, e_greedy=0.9, e_greedy_increment=0.001,
                 reward_decay=0.9, memory_size=500, batch_size=32):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = 0 if e_greedy_increment is not None else e_greedy
        self.epsilon_max = e_greedy
        self.e_greedy_increment = e_greedy_increment
        self.gamma = reward_decay

        self.replace_iter = 300
        self.train_counter = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros(shape=(self.memory_size, self.n_features * 2 + 2))
        self.memory_counter = 0

        self._build_net()
        self.sess = tf.Session()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess.run(tf.global_variables_initializer())

        self.cost_hist = []

    def _build_net(self):
        w_init = tf.random_normal_initializer(0.0, 0.3)
        b_init = tf.constant_initializer(0.1)
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='target')
        c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('state_value_net'):
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, 10], initializer=w_init, collections=c_names)
                    b1 = tf.get_variable('b1', [1, 10], initializer=b_init, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [10, 1], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_init, collections=c_names)
                    self.eval_state_value = tf.matmul(l1, w2) + b2

            with tf.variable_scope('action_value_net'):
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, 10], initializer=w_init, collections=c_names)
                    b1 = tf.get_variable('b1', [1, 10], initializer=b_init, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [10, self.n_actions], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                    self.eval_action_value = tf.matmul(l1, w2) + b2

            with tf.variable_scope('q_eval'):
                self.q_eval = self.eval_state_value + self.eval_action_value - tf.reduce_mean(self.eval_action_value,
                                                                                              axis=1, keep_dims=True)

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

            with tf.variable_scope('train'):
                self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.state_next = tf.placeholder(tf.float32, [None, self.n_features], name='state_next')
        c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('q_target'):
            with tf.variable_scope('state_value_net'):
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, 10], initializer=w_init, collections=c_names)
                    b1 = tf.get_variable('b1', [1, 10], initializer=b_init, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.state_next, w1) + b1)

                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [10, 1], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_init, collections=c_names)
                    self.target_state_value = tf.matmul(l1, w2) + b2

            with tf.variable_scope('action_value_net'):
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [self.n_features, 10], initializer=w_init, collections=c_names)
                    b1 = tf.get_variable('b1', [1, 10], initializer=b_init, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.state_next, w1) + b1)

                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [10, self.n_actions], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                    self.target_action_value = tf.matmul(l1, w2) + b2

            with tf.variable_scope('q_target'):
                self.q_next = self.target_state_value + self.target_action_value - \
                              tf.reduce_mean(self.target_action_value, axis=1, keep_dims=True)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.train_counter % self.replace_iter == 0:
            self.sess.run(self.replace_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter >= self.memory_size:
            batch_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, self.batch_size)
        batch_sample = self.memory[batch_index, :]

        sample_action = batch_sample[:, self.n_features]
        sample_reward = batch_sample[:, self.n_features + 1]
        sample_observation = batch_sample[:, :self.n_features]
        sample_observation_ = batch_sample[:, -self.n_features:]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.state_next: sample_observation_,
                                                  self.state: sample_observation})

        q_target = q_eval.copy()
        target_action = sample_action.astype(int)
        index = np.arange(self.batch_size, dtype=np.int32)
        q_target[index, target_action] = sample_reward + self.gamma * np.max(q_next, axis=1)

        _, cost = self.sess.run([self.train, self.loss], feed_dict={self.state: sample_observation,
                                                                    self.q_target: q_target})

        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.train_counter += 1

        self.cost_hist.append(cost)

    def store_transition(self, observation, action, reward, observation_):
        transition = np.hstack((observation, action, reward, observation_))
        store_index = self.memory_counter % self.memory_size
        self.memory[store_index, :] = transition
        self.memory_counter += 1

