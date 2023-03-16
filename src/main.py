# %%

from os.path import exists
import csv
from network import get_network_stats
from utils import *
from tqdm import trange
import networkx as nx
from parameter import AgentParameter
import numpy as np
import pickle
from ipysigma import Sigma
import pandas as pd
from social_model import SocialModel
from scipy.stats import beta
import matplotlib.pyplot as plt
from data import Data
# from d3blocks import D3Blocks

# %%
G = pickle.load(open('data/networks/MIM/ego_net.pkl', 'rb'))
# get_network_stats(G)

# %%
for i in trange(config.simulation_number, position=0):
    params_path = f'data/networks/MIM/agent_params/agent_params_{i}.pkl'
    result_path = f'data/networks/MIM/results/result_{i}.csv'
    with open(params_path, mode='rb') as fp:
        if exists(result_path):
            print(f'skipping {i}, already exist...')
            continue
        agent_params = pickle.load(fp)

    # Initialise the agents based on the agent_parameters
    test_model = SocialModel(G, agent_params)

    model_df = test_model.data_collector.get_model_vars_dataframe()
    for _ in trange(config.simulation_periods, position=1, leave=False):
        test_model.step()
        model_df = test_model.data_collector.get_model_vars_dataframe()
        # agent_dfs = test_model.data_collector.get_agent_vars_dataframe()

    model_df.to_csv(result_path)


# print(agent_dfs)
