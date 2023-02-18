import random

import mesa
import networkx as nx
from tqdm import tqdm

from parameter import AgentParameter
from social_agent import BotFollower, LeftWingBot, RegularAgent, RightWingBot
from metric import Metric


class SocialModel(mesa.Model):
    def __init__(self, G: nx.Graph, agent_parameters: list[AgentParameter]):
        # self.num_agents = N
        self.G = G
        self.schedule = mesa.time.RandomActivation(self)
        self.truth = random.random()
        for i, unique_id in enumerate(tqdm(self.G.nodes())):
            a = BotFollower(unique_id, self, agent_parameters[i])
            self.schedule.add(a)

    '''
    Add something about the Bernoulli clock here
    '''

    def step(self):
        self.schedule.step()
        print(
            f'There exists {Metric.misinformation(self)} level of misinformation within the society')

    def get_agent(self, unique_id):
        '''Helper function for retrieving specific agent

        Args:
            unique_id (int)

        Returns:
            Agent: agent with that unique_id 
        '''
        try:
            agent = [
                a for a in self.schedule.agents if a.unique_id == unique_id][0]
            return agent
        except:
            return None

    def get_agents(self):
        return self.schedule.agents
