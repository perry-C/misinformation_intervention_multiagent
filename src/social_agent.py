import math
import random

import mesa
import networkx as nx
from numpy.random import binomial
from scipy.stats import beta

import config
from agent_type import AgentType
from parameter import AgentParameter
from utils import coin_flip


class SocialAgent(mesa.Agent):

    def __init__(self, unique_id, model, agent_parameter: AgentParameter):
        super().__init__(unique_id, model)
        self.model = model
        self.G = model.G
        # self.adjacency_matrix = nx.adjacency_matrix(model.G)
        self.agent_type = None

        # An agent pays attention to two things: information shared by the unbiased source
        # and those shared by its friends

        self.influence_of_friends = 0.5
        self.bot_id = None
        # Each agent starts with a prior belief θi, 0 assumed to follow a Beta distribution, for both a,b > 0
        self.a = agent_parameter.a
        self.b = agent_parameter.b

        # self.world_view = beta.stats(self.a, self.b)

    def calculate_opinion(self):
        opinion = beta.mean(self.a, self.b)
        if math.isnan(opinion):
            opinion = 0
        return opinion

    def get_neighbors(self):
        '''
        Returns:
            list[SocialAgents]
        '''

        # Filtering out neighbors based on result from a bernoulli draw
        # models the fact that a human is not likely to pay attention to every single piece of information
        neighbors = [self.model.get_agent(
            n) for n in nx.neighbors(self.G, self.unique_id) if coin_flip()]

        return neighbors

    def update_belief(self):
        '''
        The base class provides the update rules for regular agents and bot followers
        The bot class does not share the same update rule (which will override the base method)
        '''
        as_from_neighbors = sum(n.a for n in self.get_neighbors())
        bs_from_neighbors = sum(n.b for n in self.get_neighbors())

        self.a = (1 - self.influence_of_friends) * \
            (self.a + self.learn_truth())
        + self.influence_of_friends * as_from_neighbors

        self.b = (1 - self.influence_of_friends) * \
            (self.b + self.learn_truth())
        + self.influence_of_friends * bs_from_neighbors

    def learn_truth(self):
        '''
        "We formalize the information obtained from unbiased sources as a draw
        si,t from a Bernoulli distribution centered around the true state of the world θ"

        People have a possible chance of learning the truth

        As a possible future extension, we might model the prob. as a tunable parameters.
        More well-educated people have better possible chance of learning the truth
        '''

        if binomial(1, 0.5, 1):
            return config.truth
        else:
            return 0

    def step(self):
        '''
       "Note that this rule assumes that agents exchange information (i.e.
        αj,t and βj,t ) before processing new signals si,t+1." (w24462)

        Update agent as follows:
        1. Exchange information among interested neighbor nodes
        2. Receive information from unbiased source
        '''
        self.update_belief()
        '''
        Every agent in this model receives info from unbias sources,
        however, some might ignore it like the bot
        '''


class RegularAgent(SocialAgent):
    def __init__(self, unique_id, model, agent_parameter: AgentParameter):
        super().__init__(unique_id, model, agent_parameter)
        self.agent_type = AgentType.RA

    def get_neighbors(self):
        '''
        There exists one essential difference between regular agents and bot followers:
        Regular agent can distinguish thus filtering out the bots, hence it follows the 
        implementation of the base method
        '''
        return super().get_neighbors()


class BotFollower(SocialAgent):
    '''
    'while regular agents only pay
    attention to other regular agents, bot followers devote some share of their attention to the
    information transmitted by bots, and hence are exposed to fake news.'
    '''

    def __init__(self, unique_id, model, agentParamter: AgentParameter, bot_id):
        super().__init__(unique_id, model, agentParamter)
        self.agent_type = AgentType.BF
        self.bot_id = bot_id

    def get_neighbors(self):
        '''
        Whereas bot followers can not extinguish between real people and bots,
        so no filtering is done, and additionally each bot follower will listen to
        the one of the bot-types, depending on their own views of that matter 
        (e.g. more left-oriented people will follow the L-bot)   
        '''

        neighbors = super().get_neighbors()
        # The user will pay attention to information spread by bot also by chance
        if coin_flip():
            bot_followed = self.model.get_agent(self.bot_id)
            neighbors.extend([bot_followed])
        return neighbors


class LeftWingBot(SocialAgent):
    '''
    "We assume that there are two bots, a left wing bot (or L-bot) and a right wing bot (or
    R-bot), both with biased views."
    '''

    def __init__(self, unique_id, model, agentParameter: AgentParameter):
        super().__init__(unique_id, model, agentParameter)
        self.flooding_capacity = config.flooding_capacity
        self.agent_type = AgentType.LWB

    def update_belief(self):
        '''
        "They ignore unbiased signals and those provided by other
        individuals in the network." (Acting as sinks)

        Where instead, the bots only update their believes based on the parameter k,
        the flooding capacity, which represents "how fast/slow they can
        produce fake news compared to the regular slow of informative signals received by agents"
        '''
        self.b = self.b + self.flooding_capacity


class RightWingBot(SocialAgent):

    def __init__(self, unique_id, model, agentParameter: AgentParameter):
        super().__init__(unique_id, model, agentParameter)
        self.flooding_capacity = config.flooding_capacity
        self.agent_type = AgentType.RWB

    def update_belief(self):
        self.a = self.a + self.flooding_capacity
