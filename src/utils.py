import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import binomial
from scipy.stats import beta

import config


def convert_csv_to_graph(file: str, sep=" ", whole=False):
    with open(file, "r") as fp:
        edge_df = pd.read_csv(fp, header=None, sep=sep)
        edge_list = list(zip(edge_df[0].to_list(), edge_df[1].to_list()))
        ego_net = nx.DiGraph(edge_list)

        if not whole:
            file_name, _ = os.path.splitext(file)
            ego_node_name = file_name.split("/")[2]
            ego_net.add_node(int(ego_node_name))
            for node in list(ego_net.nodes):
                ego_net.add_edge(node, int(ego_node_name))

        return ego_net


def get_intervals(lower, upper, n):
    """
    Args:
        lower (float): lower bound
        upper (float): upper bound
        n (int): number of chunks

    Returns:
        list of pairs
    """
    end_points = np.linspace(lower, upper, n + 1)

    # Split the belief spectrum into 7 evenly distributed intervals along [0,1]
    intervals = [
        (end_points[i], end_points[i + 1]) for i, _ in enumerate(end_points) if i != n
    ]
    return intervals


def compute_ab(mu):
    """
    Args:
        mu (float): expected value of the beta distribution given by some a and b

    Returns:
        a and b from that specific distribution
    """
    var = config.opinion_variance

    a = -(mu * (var + mu**2 - mu) / var)
    b = (var + mu**2 - mu) * (mu - 1) / var

    return a, b


def coin_flip(prob: float = config.communication_speed) -> bool:
    """
    A single sample from the Bernoulli distribution
    """
    result = bool(binomial(1, prob, 1)[0])

    return result


def smooth(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        # Anchor the last smoothed value
        last = smoothed_val

    return smoothed
