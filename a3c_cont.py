# -*- coding: utf-8 -*-
import os
import time
import math
import random
import threading
import numpy as np
import tensorflow as tf
import default.env_cartpole_cont as env

#Shared global hyper-parameters
T = 0                               # Global shared counter
TMAX = 5000000                      # Max iteration of global shared counter    
THREADS = 12                        # Number of running thread
N_STEP = 5                          # Number of steps before update
WISHED_SCORE = 3000                 # Stopper of iterative learning
GAMMA = 0.99                        # Decay rate of past observations
ACTIONS = 1                         # Number of valid actions
STATES = 4                          # Number of state
ENTROPY_BETA = 0.001               # Entropy regulation term: beta, default: 0.001
# Initial Learning rate
INITIAL_ALPHA_LOW = 1e-5            # Lowest learning rate, default: 1e-5
INITIAL_ALPHA_HIGH = 1e-4           # Hight learning rate, default: 1e-4
# Optimizer
OPT_DECAY = 0.99                    # Discouting factor for the gradient, default: 0.99
OPT_MOMENTUM = 0.0                  # A scalar tensor, default: 0.0
OPT_EPSILON = 0.01                 # value to avoid zero denominator, default: 0.005
CHECKPOINT_DIR = 'default/save_a3c_cont'

class netCreator(object):
    def __init__(self):
        with tf.device("/cpu:0"):
            # Placeholder
            self.s = tf.placeholder("float", [None, STATES])  # Make as input layer of network
            self.a = tf.placeholder("float", [None]) # Action holder
            self.diff = tf.placeholder("float", [None])       # Temporary difference (R-V) (input for policy)
            self.r = tf.placeholder("float", [None])          # R term
            self.lr = tf.placeholder("float", [])             # Adaptable learning rate
            
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
            
            self.W_fc5 = self._weight_variable([100, ACTIONS])
            self.b_fc5 = self._bias_variable([ACTIONS])
            
            # weight for value output layer
            self.W_fc6 = self._weight_variable([100, 1])
            self.b_fc6 = self._bias_variable([1])

            # region make fc hiddent layers
            h_fc1 = tf.nn.relu(tf.matmul(self.s, self.W_fc1) + self.b_fc1)
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3)
            # endregion make fc hiddent layers

            # region output layer
            # policy
            self.mu = tf.matmul(h_fc3, self.W_fc4) + self.b_fc4
            self.sigma2 = tf.nn.softplus(tf.matmul(h_fc3, self.W_fc5) + self.b_fc5)
            self.log_sigma2 = tf.log(self.sigma2)
            # value
            self.v = tf.matmul(h_fc3, self.W_fc6) + self.b_fc6
            # endregion output layer
    
    def loss_func(self):
        #global ENTROPY_REGU, OPT_DECAY, OPT_MOMENTUM, OPT_EPSILON
        with tf.device("/cpu:0"):
            # TODO check diferential policy entropy
#            entropy = -tf.reduce_sum(self.pi * self.log_pi)
            entropy = -0.5 * (tf.log(2. * np.pi * self.sigma2) + 1.)
            
            # Policy loss
            D = tf.to_float(tf.size(self.a))
            x_prec = tf.exp(-self.log_sigma2)
            x_diff = tf.sub(self.a, self.mu)
            x_power = tf.square(x_diff) * x_prec * -0.5
            gaussian_nll = (tf.reduce_sum(self.log_sigma2) + D * tf.log(2. * np.pi)) / 2. - tf.reduce_sum(x_power)
            self.pi_loss = tf.mul(gaussian_nll, tf.stop_gradient(self.diff)) + ENTROPY_BETA * entropy
            
            # Value loss
            self.v_loss = tf.reduce_mean(tf.square(self.r - self.v))
            
            # Optimizer
            self.optimizer = tf.train.RMSPropOptimizer(self.lr, OPT_DECAY, OPT_MOMENTUM, OPT_EPSILON)
            self.opt_v = self.optimizer.minimize(self.v_loss)
            self.opt_pi = self.optimizer.minimize(self.pi_loss)
    
    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)  
        
    def forward_policy(self, sess, s_t):
        mu_out = sess.run(self.mu, feed_dict = {self.s : [s_t]})
        sigma2_out = sess.run(self.sigma2, feed_dict = {self.s : [s_t]})
        return mu_out[0], sigma2_out[0]

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
                self.W_fc4, self.b_fc4,
                self.W_fc5, self.b_fc5]
    def get_value_param(self):
        return [self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3,
                self.W_fc6, self.b_fc6]
        
    def train(self, sess, states, actions, R, td, learningRate):
        sess.run(self.opt_v, feed_dict = {self.s: states, self.r: R, self.lr: learningRate})
        sess.run(self.opt_pi, feed_dict = {self.s: states, self.a: actions, self.diff: td, self.lr: learningRate})

    def predict(self, sess, states):
        feed_dict = {self.s: states}
        a, V = sess.run([self.argmax_policy, self.V], feed_dict)
        return a, V

class trainThread(object):
    def __init__(self, num, lock, global_network, initial_lr):
        
        print "THREAD ", num, "STARTING...", "LEARNING POLICY => INITIAL_LEARNING_RATE:", initial_lr
        
        self.num = num
        self.lock = lock
        
        self.thread_network = netCreator()
        self.thread_network.loss_func()
        
        self.sync_net = self.thread_network.sync_from(global_network)
        
        # Open communicate with environment
        self.lock.acquire()
        self.env_state = env.CartPole()
        self.env_state.initialState()
        self.lock.release()
        
        self.initial_lr = initial_lr
        self.t = 1
        self.step_score = 0
        self.episodic_score = 0
        
        time.sleep(num/5)
        
    def choose_action(self, mu, sigma2):
#        print("mean", mu, "std", np.sqrt(sigma2))
        sample_action = np.random.normal(mu, np.sqrt(sigma2))
        # Clip actions to the range
        if sample_action < -1.:
            sample_action = -1.0
        elif sample_action > 1.:
            sample_action = 1.0
            
        return sample_action
    
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
        self.lock.acquire()
        s_t, terminal = self.env_state.getState()
        self.lock.release()

        t_start = self.t
        # Act until terminal or we did 'n_step' steps     
        while not(terminal or self.t - t_start == N_STEP):
            mu_, sigma2_ = self.thread_network.forward_policy(sess, s_t)
            v_t = self.thread_network.forward_value(sess, s_t)
            a_t = self.choose_action(mu_, sigma2_)
            self.lock.acquire()
            s_t1, r_t, terminal = self.env_state.oneStep(a_t) # Run one step of sim 
            self.lock.release()
            
            if (self.num == 0) and (self.t % 100) == 0:
                print "Mean:", mu_, "/ Variance:", sigma2_, "/ Action:", a_t, "/ V", v_t
                
            # Accumulate gradients
            states.append(s_t)
            actions.append(a_t)
            rewards.append(r_t)
            values.append(v_t)
            
            # Update counters
            self.step_score += r_t
            self.t += 1
            T += 1
            s_t = s_t1
           
        self.episodic_score = self.step_score
        
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
            
        if terminal:
            print "THREAD:", self.num, "/ T", T, "/ TSTEP", self.t, "/ LRATE", cur_lr, "/ SCORE", self.episodic_score           
            self.step_score = 0
            # Reset state
            self.lock.acquire()
            self.env_state.initialState()
            self.lock.release()
                       
        return self.episodic_score, states, actions, R, td, cur_lr
    
    def _anneal_learning_rate(self, global_t):
        global TMAX
        learning_rate = self.initial_lr * (TMAX - global_t) /TMAX
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


def log_uniform(lo, hi):
    log_lo = math.log(lo)
    log_hi = math.log(hi)  
#    v = log_lo * (1-0.5) + log_hi * 0.5
#    return math.exp(v)
    return math.exp(random.uniform(log_lo, log_hi))

def igniter(thread_index):
    global T, TMAX
    training_thread = threads_checkin[thread_index]
    score = 0
    while T < TMAX and score < WISHED_SCORE:
        # apply gradients
        # TODO: Considering tensorflow handles the 'batch', no need to accumulate and then update
        score, states, actions, R, td, cur_lr = training_thread.actorLearner(sess)
        global_network.train(sess, states, actions, R, td, cur_lr)

# Globally shared network
global_network = netCreator()
global_network.loss_func()
    
lock = threading.Lock()
threads_checkin = list()
for i in range(THREADS):
    initial_lr = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH)
    training_threads = trainThread(i, lock, global_network, initial_lr)
    threads_checkin.append(training_threads)
  
# Initialize session and variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print "Successfully loaded:", checkpoint.model_checkpoint_path

if __name__ == "__main__":
    # Start n concurrent actor threads
    threads = list()
    for i in range(THREADS):
        t = threading.Thread(target=igniter, args=(i,))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    for x in threads:
        x.join()
        
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)  

    saver.save(sess, CHECKPOINT_DIR + '/' + 'cartpole', global_step = T)
