# -*- coding: utf-8 -*-
import os
import random
import threading
import numpy as np
import tensorflow as tf

import default.cartpole as cartpole
import default.mountaincar as mountaincar

from collections import deque

#Shared global hyper-parameters
T = 0                            # Global shared counter
TMAX = 5000000                   # Max iteration of global shared counter    
THREADS = 12                     # Number of running thread
N_STEP = 7                       # Number of steps before update
GAMMA = 0.99                     # Decay rate of past observations
ENTROPY_REGU = 0.001             # Entropy regulation term: beta, default: 0.001
INIT_LEARNING_RATE = 0.0001      # Initial learning rate
# Optimizer
OPT_DECAY = 0.99                 # Discouting factor for the gradient, default: 0.99
OPT_MOMENTUM = 0.0               # A scalar tensor, default: 0.0
OPT_EPSILON = 0.005              # value to avoid zero denominator, default: 0.01
ENVIRONMENT = 1
if ENVIRONMENT == 1:
    ACTIONS = 2
    STATES = 4
    MAX_SCORE = 3000
    CHECKPOINT_DIR = 'default/save_a3c/cartpole'
elif ENVIRONMENT == 2:
    ACTIONS = 3
    STATES = 2
    MAX_SCORE = 10
    CHECKPOINT_DIR = 'default/save_a3c/mountaincar'

class netCreator(object):
    def __init__(self):
        with tf.device("/cpu:0"):
            # Placeholder
            self.s = tf.placeholder("float", [None, STATES])  # Make as input layer of network
     
            # Network weights   
            self.W_fc1 = self._weight_variable([STATES, 300])
            self.b_fc1 = self._bias_variable([300])

            self.W_fc2 = self._weight_variable([300, 200])
            self.b_fc2 = self._bias_variable([200])
            
            self.W_fc3 = self._weight_variable([200, 100])
            self.b_fc3 = self._bias_variable([100])
            
            # weight for policy output layer
            self.W_fc4 = self._weight_variable([100, ACTIONS])
            self.b_fc4 = self._bias_variable([ACTIONS])
            
            # weight for value output layer
            self.W_fc5 = self._weight_variable([100, 1])
            self.b_fc5 = self._bias_variable([1])

            # region make fc hiddent layers
            h_fc1 = tf.nn.relu(tf.matmul(self.s, self.W_fc1) + self.b_fc1)
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3)
            # endregion make fc hiddent layers

            # region output layer
            # policy
            self.pi = tf.nn.softmax(tf.matmul(h_fc3, self.W_fc4) + self.b_fc4)
            self.log_pi = tf.log(self.pi)
            self.argmax_pi = tf.argmax(self.pi, dimension=1)
            # value
            self.v = tf.matmul(h_fc3, self.W_fc5) + self.b_fc5
            # endregion output layer
    
    def loss_func(self):
        #global ENTROPY_REGU, OPT_DECAY, OPT_MOMENTUM, OPT_EPSILON
        with tf.device("/cpu:0"):
            # Entropy loss
            entropy = -tf.reduce_sum(self.pi * self.log_pi)
            # Policy loss
            self.a = tf.placeholder("float", [None, ACTIONS])
            self.diff = tf.placeholder("float", [None])
            self.pi_loss = -(tf.reduce_sum(tf.mul(self.log_pi, self.a), 1) * self.diff + ENTROPY_REGU * entropy)
            # Value loss
            self.r = tf.placeholder("float", [None])
            self.v_loss = tf.reduce_mean(tf.square(self.r - self.v))
            
            # Optimizer
            self.lr = tf.placeholder("float")
            self.optimizer = tf.train.RMSPropOptimizer(self.lr, OPT_DECAY, OPT_MOMENTUM, OPT_EPSILON)
            self.opt_v = self.optimizer.minimize(self.v_loss)
            self.opt_pi = self.optimizer.minimize(self.pi_loss)  # TODO: wait.. should we maximize?
    
    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)  
        
    def forward_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict = {self.s : [s_t]})
        return pi_out[0]

    def forward_value(self, sess, s_t):
        v_out = sess.run(self.v, feed_dict = {self.s : [s_t]})
        return v_out[0][0] # output is scalar
        
    def sync_from(self, src_netowrk, name=None):
        src_policy_vars = src_netowrk.get_policy_param()
        src_value_vars = src_netowrk.get_value_param()
        dst_policy_vars = self.get_policy_param()
        dst_value_vars = self.get_value_param()

        sync_ops = []

        with tf.device("/cpu:0"):    
            with tf.op_scope([], name, "netCreator") as name:
                for(src_policy_var, dst_policy_var) in zip(src_policy_vars, dst_policy_vars):
                    sync_op = tf.assign(dst_policy_var, src_policy_var)
                    sync_ops.append(sync_op)
                for(src_value_var, dst_value_var) in zip(src_value_vars, dst_value_vars):
                    sync_op = tf.assign(dst_value_var, src_value_var)
                    sync_ops.append(sync_op)
                return tf.group(*sync_ops, name=name)

    def get_policy_param(self):
        return [self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3,
                self.W_fc4, self.b_fc4]
    def get_value_param(self):
        return [self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3,
                self.W_fc5, self.b_fc5]
        
    def train(self, sess, states, actions, R, td, learningRate):
        with tf.device("/gpu:0"):
            sess.run(self.opt_v, feed_dict = {self.s: states, self.r: R, self.lr: learningRate})
            sess.run(self.opt_pi, feed_dict = {self.s: states, self.a: actions, self.diff: td, self.lr: learningRate})

    def predict(self, sess, states):
        feed_dict = {self.s: states}
        a, V = sess.run([self.argmax_policy, self.V], feed_dict)
        return a, V

class trainThread(object):
    def __init__(self, num, global_network, initial_lr, environment):
        
        print "THREAD ", num, "STARTING...", "LEARNING POLICY => INITIAL_LEARNING_RATE:", initial_lr
        
        self.num = num
        
        self.thread_network = netCreator()
        self.thread_network.loss_func()
        
        self.sync_net = self.thread_network.sync_from(global_network)
        
        # Open communicate with environment
        self.environment = environment
        if self.environment == 1:
            self.env_state = cartpole.CartPole()
        elif self.environment == 2:
            self.env_state = mountaincar.MountainCarND()
        self.env_state._reset()
        
        self.initial_lr = initial_lr
        self.t = 1
        self.step_score = 0
        
    def choose_action(self, pi_values):
        totals = []
        running_total = 0.0
        for rate in pi_values:
            running_total += rate
            value = running_total
            totals.append(value)
        
        rnd = random.random() * running_total
        for i in range(len(totals)):
            if totals[i] >= rnd:
                return i
        # fail safe
        return np.argmax(pi_values)
    
    def actorLearner(self, sess):     
        global T
        # Reset gradients
        states = []
        actions = []
        rewards = []
        values = []
        R = []
        td = []
        
        sess.run(self.sync_net) # Sync with global network
        
        # Grab a state
        s_t, terminal = self.env_state._state()

        t_start = self.t
        # Act until terminal or we did 'n_step' steps     
        for i in range(N_STEP):
            pi_ = self.thread_network.forward_policy(sess, s_t)
            v_t = self.thread_network.forward_value(sess, s_t)
            
            # Action selection based on softmax
            a_index = self.choose_action(pi_)
            a_t = np.zeros([ACTIONS])
            a_t[a_index] = 1
            
            # Run one step of sim
            s_t1, r_t, terminal = self.env_state._step(a_index)
            
            # Accumulate gradients
            states.append(s_t)
            actions.append(a_t)
            rewards.append(r_t)
            values.append(v_t)
            
            # Update counters
            if self.environment == 1:
                self.step_score += r_t
            elif self.environment == 2:
                self.step_score += 0
            self.t += 1
            T += 1
            s_t = s_t1
            
            if (self.num == 0) and (self.t % 100) == 0:
                print "P:", pi_, "/ V", v_t, "/ ACTION", a_t
            
            if terminal:
                print "THREAD:", self.num, "/ TIME", T, "/ TIMESTEP", self.t, "/ SCORE", self.step_score         
                if self.environment == 1:
                    self.step_score = 0
                elif self.environment == 2:
                    self.step_score += 1
                # Reset state
                self.env_state._reset()
                break

        # bootstrap if last state not terminal
        R_t = 0.0 if terminal else self.thread_network.forward_value(sess, s_t)
        
        states.reverse()
        rewards.reverse()
        actions.reverse()
        values.reverse()  
        steps_done = self.t - t_start
        for i in range(steps_done):  # [t-1 ..., t_start] but shifted to start at 0
            R.append(rewards[i] + GAMMA * R_t)
            td.append(R[i] - values[i])
            
        cur_lr = self._anneal_learning_rate(T)
                       
        return self.step_score, states, actions, R, td, cur_lr
    
    def _anneal_learning_rate(self, global_t):
        global TMAX
        learning_rate = self.initial_lr * (TMAX - global_t) /TMAX
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

def igniter(thread_index, coord, lock):
    global T, TMAX, MAX_SCORE
    training_thread = threads_checkin[thread_index]
    score = 0
    while not coord.should_stop():
        if T > TMAX:
            coord.request_stop()
        if score > MAX_SCORE:
            coord.request_stop()
            
        score, states, actions, R, td, cur_lr = training_thread.actorLearner(sess)
        lock.acquire()
        global_network.train(sess, states, actions, R, td, cur_lr)
        lock.release()
        
# Globally shared network
global_network = netCreator()
global_network.loss_func()

# Global experiences for reply
experiences = deque()

lock = threading.Lock()
threads_checkin = list()
for i in range(THREADS):
    training_threads = trainThread(i, global_network, INIT_LEARNING_RATE, ENVIRONMENT)
    threads_checkin.append(training_threads)
  
# Initialize session and variables
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

coord = tf.train.Coordinator()

checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print "Successfully loaded:", checkpoint.model_checkpoint_path

if __name__ == "__main__":
    # Start n concurrent actor threads
    threads = list()
    for i in range(THREADS):
        t = threading.Thread(target=igniter, args=(i, coord, lock,))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    coord.join(threads)
        
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)  

    saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = T)
