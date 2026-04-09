import numpy as np
import random
from collections import defaultdict

RL_ACTIONS = ["Strong", "Normal", "Safe", "Invert"]

def default_q():
    return np.zeros(len(RL_ACTIONS))

class RLAgent:
    def __init__(self):
        self.q_table = defaultdict(default_q)
        self.last_state = None
        self.last_action = None

    def step(self, conf, drift, streak, reward=None):
        loss_level = min(streak, 3)
        low_conf = int(conf < 0.6)
        drift_flag = int(drift)
        state = (low_conf, drift_flag, loss_level)

        if reward is not None and self.last_state is not None:
            old_value = self.q_table[self.last_state][self.last_action]
            next_max = np.max(self.q_table[state])
            self.q_table[self.last_state][self.last_action] = old_value + 0.1 * (reward + 0.9 * next_max - old_value)

        action = random.randint(0, len(RL_ACTIONS) - 1) if random.random() < 0.1 else np.argmax(self.q_table[state])
        self.last_state, self.last_action = state, action
        return RL_ACTIONS[action]