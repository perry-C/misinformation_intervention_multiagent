import random
import numpy as np
import networkx as nx
import pandas as pd
import os


class Data:
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
