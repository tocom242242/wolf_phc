import copy
import numpy as np
import ipdb
from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):

    @abstractmethod
    def select_action(self, **kwargs):
        pass

class NormalPolicy(Policy):
    """
        与えられた確率分布に従って選択
    """
    def __init__(self):
        super(NormalPolicy, self).__init__()


    def select_action(self, pi):
        action = np.random.choice(np.arange(len(pi)), p=pi)

        return action
