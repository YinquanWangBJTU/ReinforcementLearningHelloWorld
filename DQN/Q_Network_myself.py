import tensorflow as tf
import numpy as np


class DeepQNetwork:
    def __init__(self, n_features, n_actions, learning_rate=0.001, e_greedy=0.9, reward_decay=0.9,
                 e_greedy_increment=0.01, tau=0.99):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e_greedy_increment = e_greedy_increment
        self.e_greedy = e_greedy
        self.tau = tau
        self.epsilon = 0 if e_greedy_increment is not None else self.e_greedy
        self.learn_step_counter = 0
        self.replace_target_iter = 300

        self._build_net()

        e_params = tf.get_collection('eval_params_collections')
        t_params = tf.get_collection('target_params_collections')

        self.replace_op = [tf.assign(t, (1-self.tau) * t + self.tau * e)
                           for t, e in zip(t_params, e_params)]
        # self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_list = []

    def _build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='target')
        with tf.variable_scope('eval_net'):
            c_names = ['eval_params_collections', tf.GraphKeys.GLOBAL_VARIABLES]
            w_init = tf.random_normal_initializer(0.0, 0.3)
            b_init = tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 10], initializer=w_init,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, 10], initializer=b_init,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [10, self.n_actions], initializer=w_init,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init,
                                     collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

            with tf.variable_scope('train'):
                self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.state_next = tf.placeholder(tf.float32, [None, self.n_features], name='state_next')
        with tf.variable_scope('target_net'):
            c_names = ['target_params_collections', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 10], initializer=w_init,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, 10], initializer=b_init,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [10, self.n_actions], initializer=w_init,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init,
                                     collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            action_index = np.argmax(action_value)
        else:
            action_index = np.random.randint(0, self.n_actions)
        return action_index

    def learn(self, sample, batch_size):
        self.sess.run(self.replace_op)
        # if self.learn_step_counter % self.replace_target_iter == 0:
        #     self.sess.run(self.replace_op)
        #     print('\ntarget_params_replaced\n')

        state = sample[:, :self.n_features]
        state_next = sample[:, -self.n_features:]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.state_next: state_next,
                                                  self.state: state}
                                       )

        q_target = q_eval.copy()
        target_act = sample[:, self.n_features].astype(int)
        reward = sample[:, self.n_features + 1]
        index = np.arange(batch_size, dtype=np.int32)
        q_target[index, target_act] = reward + self.gamma * np.max(q_next, axis=1)

        _, cost = self.sess.run([self.train, self.loss], feed_dict={self.state: state,
                                                                    self.q_target: q_target})
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        self.learn_step_counter += 1
        self.cost_list.append(cost)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
