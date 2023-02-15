from operator import itemgetter
from ipysigma import Sigma
import networkx as nx
import random
from tqdm import tqdm, trange
import math
import matplotlib.pyplot as plt
import pandas as pd

from itertools import combinations


def find_matching_pair(G: nx.Graph, nodes, v):
    # Find all x,w pairs which are alters(neighbors) of v, and not connected to each other
    neighbors_of_v = list(nx.all_neighbors(G, v))
    xw_pairs = [(x, w) for (x, w) in combinations(
        neighbors_of_v, 2) if G.has_edge(x, w) == False]

    # Find all y,z pairs which are not alters of v, and not connected to each other
    # also they can not be connected to v
    not_neighbors_of_v = list(set(nodes) - set(neighbors_of_v))
    yz_pairs = [(y, z) for (y, z) in combinations(
        not_neighbors_of_v, 2) if G.has_edge(y, z) == False]

    # Find a random matching so that one node in xw pair connects to yz in another pair
    matching_pair = []

    for xw_pair in xw_pairs:
        for yz_pair in yz_pairs:
            x, w = xw_pair
            y, z = yz_pair
            # find all matching pairs such that there each node
            # in one pair is connected to at least one node in the other pair
            if G.has_edge(x, z) & G.has_edge(w, y) | G.has_edge(x, y) & G.has_edge(w, z):
                matching_pair = [x, w, y, z]
                break
            else:
                continue
    return matching_pair


def pick_five_nodes(G: nx.Graph):
    '''
    "five nodes in the random graph are randomly selected ..."

    "...in the second step the five nodes randomly picked from the random
    graph should meet the following conditions:
    1) x and w are alters of v;
    2) y and z are not alters of v;
    3) ewy and exz do exist.
    4) ewx and eyz do NOT exist."     
    '''

    # first pick v randomly, with at least 2 neighbors
    nodes = list(G.nodes())

    nodes_with_two_neighbors = [n for n in nodes if len(
        list(nx.all_neighbors(G, n))) >= 2]

    random.shuffle(nodes_with_two_neighbors)

    while True:
        try:
            v = nodes_with_two_neighbors.pop()
            matching_pair = find_matching_pair(G, nodes, v)
            x, w, y, z = matching_pair
            break
        except ValueError:
            continue

    # check algo validity

    # assert ego_network.has_edge(v, x)
    # assert ego_network.has_edge(v, w)
    # assert ego_network.has_edge(v, y) == False
    # assert ego_network.has_edge(v, z) == False
    # assert ego_network.has_edge(x, z)
    # assert ego_network.has_edge(y, w)

    return x, w, y, z, v


def rewire_edges(G: nx.Graph, x, w, y, z):
    '''
    "...two edges among them are partly rewired to add one triangle"
    '''
    if G.has_edge(x, y) & G.has_edge(w, z):
        G.remove_edge(x, y)
        G.remove_edge(w, z)
    elif G.has_edge(x, z) & G.has_edge(w, y):
        G.remove_edge(x, z)
        G.remove_edge(w, y)

    G.add_edge(x, w)
    G.add_edge(y, z)


# Algorithm for increasing the clustering coefficient of a graph (Guo and Kraines, 2009)

def increase_clustering_coefficient(G: nx.Graph, tcc):
    '''_summary_

    Args:
        G (graph): graph generated following power-law distribution, i.e. a BA graph 
        cc(float): target clustering coefficient
    '''

    acc = nx.average_clustering(G)

    '''
    the process will be repeated until the average clustering coefficient 
    of the rewired graph is greater than or equal to the target average clustering coefficient C(G)
    '''

    while acc < tcc:
        # Recalculate every loop
        acc = nx.average_clustering(G)
        # pick five node comforming to the 5 conditions
        x, w, y, z, v = pick_five_nodes(G)
        rewire_edges(G, x, w, y, z)
        print(f"acc: {acc}")

# Taken from https://stackoverflow.com/a/64787324


def have_bidirectional_relationship(G: nx.Graph, node1, node2):
    return G.has_edge(node1, node2) and G.has_edge(node2, node1)


def get_bidirectional_edges(G):
    biconnections = set()
    for u, v in G.edges():
        if u > v:  # Avoid duplicates, such as (1, 2) and (2, 1)
            v, u = u, v
        if have_bidirectional_relationship(G, u, v):
            biconnections.add((u, v))
    return biconnections


def modify_reciprocity(G):
    # Manipulate reciprocity around 0.32 by randomly turning bidirectional edges into single-direction ones
    bidirectional_edges = list(get_bidirectional_edges(G))
    random.shuffle(bidirectional_edges)
    n_edges_to_remove = math.floor(len(bidirectional_edges) * (1 - 0.2))

    for _ in trange(n_edges_to_remove):
        u, v = bidirectional_edges.pop()
        # Balancing things out, so that not all edges in one direction got removed
        if random.choice([0, 1]) == 0:
            G.remove_edge(u, v)
        else:
            G.remove_edge(v, u)


def generate_ego_network(n: int, m: int, seed):
    '''_summary_
    Args:
        n (_type_): number of nodes
        m (_type_): the number of new edge links formed with each new node 
        seed (_type_): seed for genererate random BA model graph
    '''

    # https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.powerlaw_cluster_graph.html
    plc_graph = nx.powerlaw_cluster_graph(n, m, 1, seed=seed)
    # ba_graph = nx.barabasi_albert_graph(n, m, seed=seed)

    # find node with largest degree
    node_and_degree = plc_graph.degree()

    (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]

    # Create ego graph
    ego_network = nx.ego_graph(plc_graph, largest_hub)

    ego_network = ego_network.to_directed()

    # # Set attributes of nodes for clarity
    # attributes = {n: {'identity': (
    #     'ego' if n == largest_hub else 'alter')} for n in all_nodes}

    # nx.set_node_attributes(ego_network, attributes)

    return ego_network


def get_network_stats(G: nx.Graph):

    average_clustering = nx.average_clustering(G.to_undirected())
    print()
    average_in_degree = sum(
        [y for (x, y) in G.in_degree]) / G.number_of_nodes()
    reciprocity = nx.overall_reciprocity(G)
    return average_clustering, average_in_degree, reciprocity
