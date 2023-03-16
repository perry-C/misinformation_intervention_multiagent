import random

import mesa
import networkx as nx
import numpy as np
from mesa import DataCollector
from tqdm import tqdm

import config
from metric import Metric
from parameter import AgentParameter
from social_agent import BotFollower, LeftWingBot, RegularAgent, RightWingBot


class SocialModel(mesa.Model):
    def __init__(self, G: nx.Graph, agents_params: list[AgentParameter]):
        self.G = G
        self.agents_params = agents_params
        self.schedule = mesa.time.RandomActivation(self)
        self.num_bot_followers = round(
            self.G.number_of_nodes() * config.bot_follower_percentage)
        self.num_l_bot_followers = round(
            self.num_bot_followers * random.random())
        self.num_r_bot_followers = self.num_bot_followers - self.num_l_bot_followers
        self.num_regular_agents = self.G.number_of_nodes() - self.num_bot_followers
        self.l_bot_id = self.G.number_of_nodes()
        self.r_bot_id = self.G.number_of_nodes()+1

        for _ in range(self.num_l_bot_followers):
            a = BotFollower(self.get_id(), self,
                            self.agents_params.pop(), self.l_bot_id)
            self.schedule.add(a)

        for _ in range(self.num_r_bot_followers):
            a = BotFollower(self.get_id(), self,
                            self.agents_params.pop(), self.r_bot_id)
            self.schedule.add(a)

        for _ in range(self.num_regular_agents):
            a = RegularAgent(self.get_id(), self, self.agents_params.pop())
            self.schedule.add(a)

        l_bot = LeftWingBot(self.l_bot_id, self, self.agents_params.pop())
        r_bot = RightWingBot(self.r_bot_id, self, self.agents_params.pop())

        self.schedule.add(l_bot)
        self.schedule.add(r_bot)

        self.data_collector = DataCollector(model_reporters={'polarization': lambda m: Metric.polarization(m),
                                                             'misinformation': lambda m: Metric.misinformation(m),
                                                             'average_opinion_all': lambda m: Metric.average_opinion_all(m),
                                                             'average_opinion_left': lambda m: Metric.average_opinion_left(m),
                                                             'average_opinion_right': lambda m: Metric.average_opinion_right(m),
                                                             },
                                            # agent_reporters={'opinion': lambda a: a.calculate_opinion()}
                                            )

    def step(self):

        self.data_collector.collect(self)
        self.schedule.step()

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

    def get_id(self):
        return self.schedule.get_agent_count()
