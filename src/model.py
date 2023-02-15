import mesa
from agent import SocialAgent
import random


class NewsModel(mesa.Model):
    def __init__(self, N):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.truth = random.random()
        for i in range(self.num_agents):
            a = SocialAgent(i, self, i)
            self.schedule.add(a)

    '''
    Add something about the Bernoulli clock here
    '''

    def step(self):
        self.schedule.step()

    def get_agent(self, agent_id):
        agents = self.schedule.agents
        try:
            return agents[agents.index(agent_id)]
        except ValueError:
            return None
