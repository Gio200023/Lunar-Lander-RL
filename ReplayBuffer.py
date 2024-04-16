import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state','done'))


class ReplayMemory(object):
    """Replay Memory class for the replay buffer

        Functions:
        sample: sample a batch of transitions
        push: save a transition
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clean(self):
        self.memory.clear()