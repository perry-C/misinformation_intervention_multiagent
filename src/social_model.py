import random

import mesa
import networkx as nx
import numpy as np
from mesa import DataCollector
from tqdm import tqdm

import config
from agent_type import AgentType
from data import generate_agent_params, read_agent_params
from metric import Metric
from parameter import AgentParameter
from social_agent import *


class SocialModel(mesa.Model):
    def __init__(
        self,
        G: nx.Graph,
        seed,
        exp,
        flooding_capacity,
        bot_follower_percentage,
        activation_delay,
        inoculation_rate,
        inoculation_range,
        from_scratch,
        param_index,
        collect_agent_data,
    ):
        self.G = G
        self.exp = exp

        self.l_bot_ids = [
            i
            for i in range(
                G.number_of_nodes(), G.number_of_nodes() + config.left_bot_number
            )
        ]
        self.r_bot_ids = [
            j
            for j in range(
                G.number_of_nodes() + config.left_bot_number,
                G.number_of_nodes() + config.left_bot_number + config.right_bot_number,
            )
        ]

        self.collect_agent_data = collect_agent_data
        # agents_params = agents_params
        self.schedule = mesa.time.RandomActivation(self)
        self.flooding_capacity = flooding_capacity
        self.bot_follower_percentage = bot_follower_percentage
        self.banned_count = 0
        self.activation_delay = activation_delay
        self.inoculation_rate = inoculation_rate
        self.inoculation_range = inoculation_range
        if from_scratch:
            self.init_agents_from_scratch(seed)
        else:
            self.read_agents_from_file(param_index)

        self.model_data_collector = DataCollector(
            model_reporters={
                "polarization": lambda m: Metric.polarization(m),
                "misinformation": lambda m: Metric.misinformation(m),
                "average_opinion_all": lambda m: Metric.average_opinion_all(m),
                "average_opinion_left": lambda m: Metric.average_opinion_left(m),
                "average_opinion_right": lambda m: Metric.average_opinion_right(m),
                "average_opinion_reg": lambda m: Metric.average_opinion_reg(m),
                #  'opinion_l_bot': lambda m: m.get_agent(self.l_bot_id).calculate_opinion(),
                #  'opinion_r_bot': lambda m: m.get_agent(self.r_bot_id).calculate_opinion(),
                "accounts_banned": lambda m: m.get_banned_count(),
                "num_bot_followers": lambda m: m.num_bot_followers,
                "num_l_followers": lambda m: m.num_l_followers,
                "num_r_followers": lambda m: m.num_r_followers,
                "num_regular_agents": lambda m: m.num_regular_agents,
            }
        )

        self.agent_data_collector = DataCollector(
            agent_reporters={
                "step": lambda a: self.schedule.steps,
                "opinion": lambda a: a.calculate_opinion(),
                # 'opinion_l_bf': lambda a: a.calculate_opinion() if a.bot_id == self.l_bot_id else None,
                # 'opinion_r_bf': lambda a: a.calculate_opinion() if a.bot_id == self.r_bot_id else None,
            }
        )

    def step(self):
        self.model_data_collector.collect(self)

        if self.collect_agent_data:
            if self.schedule.steps % 100 == 0:
                self.agent_data_collector.collect(self)

        self.schedule.step()

    def init_agents_from_scratch(self, seed):
        random.seed(seed)

        agents_params = generate_agent_params(
            seed,
            self.G.number_of_nodes(),
            self.exp,
            self.flooding_capacity,
            self.inoculation_range,
        )

        num_bot_followers = round(
            self.G.number_of_nodes() * self.bot_follower_percentage
        )

        num_l_followers = round(num_bot_followers * random.random())

        num_r_followers = num_bot_followers - num_l_followers

        num_regular_agents = self.G.number_of_nodes() - num_bot_followers

        self.num_bot_followers = num_bot_followers
        self.num_l_followers = num_l_followers
        self.num_r_followers = num_r_followers
        self.num_regular_agents = num_regular_agents

        for _ in range(num_l_followers):
            a = BotFollower(
                self.get_id(), self, agents_params.pop(), self.l_bot_ids, "L"
            )
            self.schedule.add(a)

        for _ in range(num_r_followers):
            a = BotFollower(
                self.get_id(), self, agents_params.pop(), self.r_bot_ids, "R"
            )
            self.schedule.add(a)

        for _ in range(num_regular_agents):
            a = RegularAgent(self.get_id(), self, agents_params.pop())
            self.schedule.add(a)

        if self.exp == 3:
            for a in random.sample(
                self.get_agents(),
                round(self.schedule.get_agent_count() * self.inoculation_rate),
            ):
                a.inoculated = True

        for l_bot_id in self.l_bot_ids:
            l_bot = LeftWingBot(l_bot_id, self, agents_params.pop())
            self.schedule.add(l_bot)

        for r_bot_id in self.r_bot_ids:
            r_bot = RightWingBot(r_bot_id, self, agents_params.pop())
            self.schedule.add(r_bot)

    def read_agents_from_file(self, param_index):
        agent_abs, lfs, rfs = read_agent_params(param_index)

        self.num_bot_followers = 0
        self.num_l_followers = 0
        self.num_r_followers = 0
        self.num_regular_agents = 0

        for i, ab in enumerate(agent_abs):
            if i in lfs:
                a = BotFollower(i, self, ab, self.l_bot_id)
                self.num_l_followers += 1
                self.num_bot_followers += 1
            elif i in rfs:
                a = BotFollower(i, self, ab, self.r_bot_id)
                self.num_r_followers += 1
                self.num_bot_followers += 1

            elif i == self.l_bot_id:
                a = LeftWingBot(i, self, ab)

            elif i == self.r_bot_id:
                a = RightWingBot(i, self, ab)

            else:
                a = RegularAgent(i, self, ab)
                self.num_regular_agents += 1

            self.schedule.add(a)

    def get_agent(self, unique_id):
        """Helper function for retrieving specific agent

        Args:
            unique_id (int)

        Returns:
            Agent: agent with that unique_id
        """
        try:
            agent = [a for a in self.schedule.agents if a.unique_id == unique_id][0]
            return agent
        except:
            return None

    def get_agents(self):
        return self.schedule.agents

    def get_id(self):
        return self.schedule.get_agent_count()

    def get_banned_count(self):
        temp = self.banned_count
        self.banned_count = 0
        return temp
