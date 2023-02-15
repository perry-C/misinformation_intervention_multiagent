import mesa
import random
from numpy.random import binomial
from scipy.stats import beta
import networkx as nx


class SocialAgent(mesa.Agent):
    def __init__(self, unique_id, model: mesa.Model, G: nx.Graph, seed):
        super().__init__(unique_id, model)
        random.seed(seed)

        self.model = model
        self.adjacency_matrix = nx.adjacency_matrix(G)
        self.agent_id = unique_id
        self.misinformation = random.random()
        self.neighbors = list(nx.neighbors(G, self.agent_id))

        # self.polarisation = random.random()

        self.influence_of_friends = random.random()
        '''
        An agent pays attention to two things:
        information shared by the unbiased source and those shared by its friends
        '''

        self.a, self.b = 0
        '''
        Each agent starts with a prior belief θi, 0 assumed to follow a Beta distribution, for both a,b > 0
        '''
        # self.world_view = beta.stats(self.a, self.b)
        self.opinion = round(self.a / (self.a + self.b), 6)

    def learn_truth(self):
        '''
        "We formalize the information obtained from unbiased sources as a draw
        si,t from a Bernoulli distribution centered around the true state of the world θ"

        People have a possible chance of learning the truth 

        As a possible future extension, we might model the prob. as a tunable parameters.
        More well-educated people have better possible chance of learning the truth
        '''

        if binomial(1, 0.5, 1):
            return self.model.truth
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

        as_from_friends = sum(self.model.get_agent(
            n).a for n in self.neighbors)

        bs_from_friends = sum(self.model.get_agent(
            n).b for n in self.neighbors)

        self.a = (1 - self.influence_of_friends) * \
            (self.a + self.learn_truth())
        + self.influence_of_friends * as_from_friends

        self.b = (1 - self.influence_of_friends) * \
            (self.b + self.learn_truth())
        + self.influence_of_friends * bs_from_friends

        '''
        Every agent in this model receives info from unbias sources,
        however, some might ignore it like the bot
        '''

        print(f"agent {self.user_id} has \
              {self.misinformation} of misinformation and \
              {self.polarisation} of polarisation")


class RegularAgent(SocialAgent):
    def __init__(self, unique_id, model, seed):
        super.__init__(self, unique_id, model, seed)

    def step(self):
        '''
        Regular agents and bot followers share the same update rule.
        '''
        super.step()


class BotFollower(SocialAgent):
    def __init__(self, unique_id, model, seed):
        super.__init__(self, unique_id, model, seed)

    def step(self):
        '''
        Regular agents and bot followers share the same update rule.
        '''
        super.step()


class Bot(SocialAgent):

    '''
    while regular agents only pay
    attention to other regular agents, bot followers devote some share of their attention to the
    information transmitted by bots, and hence are exposed to fake news.
    '''

    def __init__(self, unique_id, model, seed):
        super.__init__(self, unique_id, model, seed)
