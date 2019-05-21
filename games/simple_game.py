from games.game import Game
from tqdm import tqdm
import numpy as np

class SimpleGame(Game):
    """
        シンプルなmatrix game
    """
    def __init__(self, nb_eps=1, nb_steps=10000,agents=None):
        self.agents = agents
        self.nb_steps = nb_steps
        self.nb_eps = nb_eps
        self.reward_matrix = self._create_reward_table()

    def _reset_agents(self):
        for agent in self.agents:
            from_s = agent.state
            to_s = agent.init_state()
            self.env.force_move(int(from_s), int(to_s))

    def run(self):
        all_log = []
        for eps in range(self.nb_eps):
            social_rewards = []
            for step in tqdm(range(self.nb_steps)):
                a0, a1 = self.agents[0].act(), self.agents[1].act()
                r0, r1 = self.reward_matrix[a0][a1]
                social_rewards.append(r0+r1)
                self.agents[0].get_reward(r0)
                self.agents[1].get_reward(r1)

        social_rewards = np.array(social_rewards)
        return {"social_rewards":social_rewards}


    def _create_reward_table(self):
        reward_matrix = [
                            [[1, -1], [-1, 1]],
                            [[-1, 1], [1, -1]]
                        ]

        return reward_matrix
