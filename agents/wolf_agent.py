from abc import ABCMeta, abstractmethod
import numpy as np
import ipdb

class Agent(metaclass=ABCMeta):
    """Abstract Agent Class"""

    def __init__(self, alpha=None, policy=None):
        """
        :param alpha:
        :param policy:
        """
        self.alpha = alpha
        self.policy = policy
        self.rewards = []

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def get_reward(self, reward):
        pass


class WoLFAgent(Agent):
    """
        Policy hill-climbing algorithm(PHC)
        http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
    """
    def __init__(self, delta=0.0001, action_list=None, **kwargs):
        super().__init__(**kwargs)
        self.action_list = action_list  # 選択肢
        self.last_action_id = None
        self.q_values = self._init_q_values()   # 期待報酬値の初期化
        self.conter = 0
        self.pi = [(1.0/len(action_list)) for idx in range(len(action_list))]
        self.pi_a = [(1.0/len(action_list)) for idx in range(len(action_list))]
        self.pi_history = [self.pi[0]]
        self.high_delta = 0.0004
        self.row_delta = 0.0002

    def _init_q_values(self):
        q_values = {}
        q_values = np.repeat(0.0, len(self.action_list))
        return q_values

    def act(self, q_values=None):
        action_id = self.policy.select_action(self.pi)    # 行動選択
        self.last_action_id = action_id
        action = self.action_list[action_id]
        return action

    def get_reward(self, reward):
        self.rewards.append(reward)
        self.q_values[self.last_action_id] = self._update_q_value(reward)   # 期待報酬値の更新
        self._update_pi_a()
        self._update_pi()

    def _update_q_value(self, reward):
        return ((1.0 - self.alpha) * self.q_values[self.last_action_id]) + (self.alpha * reward) # 通常の指数移動平均で更新

    def _update_pi_a(self):
       self.conter += 1
       for aidx, _ in enumerate(self.pi):
           self.pi_a[aidx] = self.pi_a[aidx] + (1/self.conter)*(self.pi[aidx]-self.pi_a[aidx])
           if self.pi_a[aidx] > 1: self.pi_a[aidx] = 1
           if self.pi_a[aidx] < 0: self.pi_a[aidx] = 0

    def _update_pi(self):
       delta = self.decide_delta()
       max_action_id = np.argmax(self.q_values)
       for aidx, _ in enumerate(self.pi):
           if aidx == max_action_id:
               update_amount = delta
           else:
               update_amount = ((-delta)/(len(self.action_list)-1))
           self.pi[aidx] = self.pi[aidx] + update_amount
           if self.pi[aidx] > 1: self.pi[aidx] = 1
           if self.pi[aidx] < 0: self.pi[aidx] = 0
       self.pi_history.append(self.pi[0])

    def decide_delta(self):
        expected_value = 0
        expected_value_average = 0
        for aidx, _ in enumerate(self.pi):
            expected_value += self.pi[aidx]*self.q_values[aidx]
            expected_value_average += self.pi_a[aidx]*self.q_values[aidx]

        if expected_value > expected_value_average: # win
            return self.row_delta
        else:   # row
            return self.high_delta
