import csv
import os
import random
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd

import config
from parameter import AgentParameter
from utils import compute_ab, get_intervals


def read_agent_params(index):

    aps = []
    lfs = []
    rfs = []

    with open(f'data/params/agent_abs_{index}.csv') as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            ap = AgentParameter(float(row[0]), float(row[1]))
            aps.extend([ap])

    with open(f'data/params/l_bot_followed_{index}.csv') as fp:
        reader = csv.reader(fp, delimiter='\n')
        for row in reader:
            lf = int(row[0])-1
            lfs.extend([lf])

    with open(f'data/params/r_bot_followed_{index}.csv') as fp:
        reader = csv.reader(fp, delimiter='\n')
        for row in reader:
            rf = int(row[0])-1
            rfs.extend([rf])

    return aps, lfs, rfs


def generate_agent_params(seed, agent_number):
    '''
    Generate the abs used for determining each agent's belief at the start of each simulation

    "This rule basically distributes our agents evenly over the belief spectrum [0, 1] such that each
    1/7 of the total mass of agents in the initial period."
    '''
    random.seed(seed)
    K = config.belief_distribution_groups

    nodes_intervals = np.linspace(
        0, agent_number + config.bot_number, K + 1)

    # Split the belief spectrum into K (7) evenly distributed intervals along [0,1]
    belief_intervals = get_intervals(0, 1, K)

    # Split the nodes into K even groups
    group_sizes = [round(nodes_intervals[i + 1]) - round(nodes_intervals[i])
                   for i, _ in enumerate(nodes_intervals) if i != K]

    agent_params = []

    for group_index in range(K):
        for _ in range(group_sizes[group_index]):
            lower, upper = belief_intervals[group_index]
            ex = random.uniform(lower, upper)
            a, b = compute_ab(ex)
            agent_params.extend([AgentParameter(a, b)])
    random.shuffle(agent_params)

    return agent_params


def pickle_network(network_path):
    fb_network = open(f"{network_path}.txt", 'r').read()
    lines = fb_network.split("\n")

    # df = pd.DataFrame()
    # temp_df = pd.DataFrame()

    processed_edges = []
    for line_index, line in enumerate(lines):

        # Chunk data processing by serialising already processed data every 10000 lines
        if line_index % 10000 == 0:
            temp_df = pd.DataFrame(processed_edges)
            processed_edges.clear()

            if os.path.exists(f"{network_path}.pkl"):
                df = pd.read_pickle(f"{network_path}.pkl")
            else:
                df = pd.DataFrame()
                df.to_pickle(f"{network_path}.pkl")

            df = pd.concat([temp_df, df], axis=0)
            df.to_pickle(f"{network_path}.pkl")

        edge = line.split(" ")
        processed_edges.extend([(int(edge[0]), int(edge[1]))])

    # return network

    # def creat_social_network(self, edges):

    #     G = nx.Graph()
    #     lines = edges.split("\n")
    #     for e in lines:
    #         nodes = e.split()
    #         if (len(nodes) > 1):
    #             if (nodes[0] not in G):
    #                 G.add_node(nodes[0])
    #             if (nodes[1] not in G):
    #                 G.add_node(nodes[1])

    #             G.add_edge(nodes[0], nodes[1])

    #     return G

    # def get_network(self, network_path):

    #     fb_network = open(network_path, 'r').read()
    #     G = self.__creat_social_network(fb_network)
    #     A = nx.adjacency_matrix(G).todense()
    #     A = np.array(A)
    #     n = A.shape[0]
    #     G = nx.from_numpy_matrix(A)

    #     # node_attr = df.set_index('id').to_dict('index')
    #     # nx.set_node_attributes(G, node_attr)

    #     return G
