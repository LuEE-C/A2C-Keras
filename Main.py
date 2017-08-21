# Initial implementation from https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
# Reworked with strong inspiration from https://github.com/Kaixhin/NoisyNet-A3C
# Changed to synchronous myself.

import numpy as np

import tensorflow as tf

import gym, time, threading

from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
from NoisyDense import NoisyDense
from keras.optimizers import RMSprop

# -- constants
ENV = 'CartPole-v1'

FRAMES_RUN_TIME = 200000
RENDER = False

THREADS = 8
OPTIMIZERS = 4
THREAD_DELAY = 0.001
SIGMA_INIT=0.02

GAMMA = 0.99

N_STEP_RETURN = 10
GAMMA_N = GAMMA ** N_STEP_RETURN

MIN_BATCH = 32

LOSS_V = .5  # v loss coefficient


def value_loss():
    def val_loss(y_true, y_pred):
        advantage = y_true - y_pred
        return K.mean(LOSS_V * K.square(advantage))
    return val_loss


def policy_loss(actual_value, predicted_value):
    advantage = actual_value - predicted_value

    def pol_loss(y_true, y_pred):
        log_prob = K.log(K.sum(y_pred * y_true, axis=1, keepdims=True) + 1e-10)
        return -log_prob * K.stop_gradient(advantage)
    return pol_loss


# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):

        state_input = Input(shape=(NUM_STATE,))
        actual_value = Input(shape=(1,))

        x = Dense(256, activation='relu')(state_input)
        x = Dense(256, activation='relu')(x)

        out_actions = NoisyDense(NUM_ACTIONS, activation='softmax', name='out_actions', sigma_init=SIGMA_INIT)(x)
        out_value = NoisyDense(1, name='out_value', sigma_init=SIGMA_INIT)(x)

        model = Model(inputs=[state_input, actual_value], outputs=[out_actions, out_value , actual_value])
        model.compile(optimizer=RMSprop(),
                      loss=[policy_loss(actual_value=actual_value, predicted_value=out_value),
                                                 value_loss(),
                                                 'mae'])

        model.summary()

        return model


    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        self.model.train_on_batch([s, r], [a, v, r])

        self.model.get_layer('out_actions').sample_noise()
        self.model.get_layer('out_value').sample_noise()

    def print_average_weight(self):
        print(np.mean(self.model.get_layer('out_actions').get_weights()[1]))

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)
        if len(self.train_queue[0]) > MIN_BATCH:
            self.optimize()
    def predict(self, s):
        p, v, r = self.model.predict([s, np.zeros(shape=(s.shape[0], 1))])
        return p, v

    def predict_p(self, s):
        p, v, r = self.model.predict([s, np.zeros(shape=(s.shape[0], 1))])
        return p

    def predict_v(self, s):
        p, v, r = self.model.predict([s, np.zeros(shape=(s.shape[0], 1))])
        return v



class Agent:
    def __init__(self, brain):
        self.frames = 0
        self.brain = brain
        self.memory = []
        self.R = 0.


    def stop_signal(self):
        if self.frames >= FRAMES_RUN_TIME:
            return True
        return False


    def act(self, s):
        self.frames += 1
        s = np.array([s])
        p = self.brain.predict_p(s)[0]

        # Making sure all is nice
        p[p<0.0000001] = 0.0000001
        p[p>0.9999999] = 0.9999999

        a = np.random.choice(NUM_ACTIONS, p=p)

        return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

# ---------
class Environment():
    stop_signal = False

    def __init__(self, render=False):

        self.render = render
        self.env = gym.make(ENV)

    def make_agent(self, brain):
        self.agent = Agent(brain)

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)
        if R == 500:
            print(self.agent.frames)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()
            self.stop_signal = self.agent.stop_signal()

    def stop(self):
        self.stop_signal = True



if __name__ == '__main__':
    # -- main
    env_test = Environment(render=RENDER)
    NUM_STATE = env_test.env.observation_space.shape[0]
    NUM_ACTIONS = env_test.env.action_space
    print(env_test.env.action_space, env_test.env.observation_space)
    print(NUM_ACTIONS, NUM_STATE)
    NUM_ACTIONS = 2
    NONE_STATE = np.zeros(NUM_STATE)

    brain = Brain()
    env_test.make_agent(brain)
    env_test.run()

    print("Training finished")