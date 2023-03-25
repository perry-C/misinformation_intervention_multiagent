# %%

import csv
import pickle
from os.path import exists

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from ipysigma import Sigma
from scipy.stats import beta
from tqdm import trange

from network import get_network_stats
from parameter import AgentParameter
from social_model import SocialModel
from utils import *

# from d3blocks import D3Blocks

# get_network_stats(G)


for i in trange(10):
    G = convert_csv_to_graph(
        f'data/networks/ego_net_test.csv', sep=',', whole=True)

    result_path = f'data/results/result_test_{i}.csv'

    # Initialise the agents based on the agent_parameters
    social_model = SocialModel(G, seed=i, from_scratch=True)

    for _ in trange(config.simulation_periods):
        model_df = social_model.data_collector.get_model_vars_dataframe()
        social_model.step()
        # agent_dfs = test_model.data_collector.get_agent_vars_dataframe()

    model_df['num_bot_followers'] = social_model.num_bot_followers
    model_df['num_l_bot_followers'] = social_model.num_l_bot_followers
    model_df['num_r_bot_followers'] = social_model.num_r_bot_followers

    model_df.to_csv(result_path, index=False)


# print(agent_dfs)
