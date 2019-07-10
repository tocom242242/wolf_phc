import numpy as np
import matplotlib.pyplot as plt
from wolf_agent import WoLFAgent 
from matrix_game import MatrixGame
import pandas as pd

if __name__ == '__main__':
    nb_episode = 1000

    actions = np.arange(2)
    agent1 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002) 
    agent2 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002)

    game = MatrixGame()
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        _, r1, r2 = game.step(action1, action2)

        agent1.observe(reward=r1)
        agent2.observe(reward=r2)

    print(agent1.pi)
    print(agent2.pi)
    plt.plot(np.arange(len(agent1.pi_history)),agent1.pi_history, label="agent1's pi(0)")
    plt.plot(np.arange(len(agent2.pi_history)),agent2.pi_history, label="agent2's pi(0)")

    plt.ylim(0, 1)
    plt.xlabel("episode")
    plt.ylabel("pi(0)")
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
