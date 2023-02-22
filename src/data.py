import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

import config


def solve_ab(ab, ex):
    a, b = ab
    eq1 = ex * (a + b) - a
    eq2 = (1 - ex) * (a + b) - b
    return [eq1, eq2]


class Data:
    def generate_agent_beliefs():
        '''
        Generate the abs used for determining each agent's belief at the start of each simulation

        "This rule basically distributes our agents evenly over the belief spectrum [0, 1] such that each
        1/7 of the total mass of agents in the initial period."
        '''

        k = config.belief_distribution_groups

        belief_intervals = np.linspace(0, 1, k + 1)

        nodes_intervals = np.linspace(0, config.number_of_nodes, k + 1)

        # Split the belief spectrum into 7 evenly distributed intervals along [0,1]
        group_ranges = [(belief_intervals[i], belief_intervals[i + 1])
                        for i, _ in enumerate(belief_intervals) if i != k]

        # Split the nodes into K even groups
        group_sizes = [round(nodes_intervals[i + 1]) - round(nodes_intervals[i])
                       for i, _ in enumerate(nodes_intervals) if i != k]

        agent_abs = []

        for i in range(k):
            for _ in range(group_sizes[i]):
                lower, upper = group_ranges[i]
                ex = random.uniform(lower, upper)
                ab = fsolve(solve_ab, (1, 1), args=(ex,))
                agent_abs.extend([ab])

        random.shuffle(agent_abs)

        return agent_abs

    def generate_bot_beliefs():
        '''
        We assume that L bot and R bot each follow the 
        initial belief of 0.4 and 0.6, 
        disguising as themselves as regular agent
        '''

        l_bot_ab = fsolve(solve_ab, (1, 1), args=(0.4))
        r_bot_ab = fsolve(solve_ab, (1, 1), args=(0.6))
        return l_bot_ab, r_bot_ab

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
