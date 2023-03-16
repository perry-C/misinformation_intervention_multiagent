import random

import config
import numpy as np
from numpy.random import binomial
from scipy.stats import beta


def get_intervals(lower, upper, n):
    '''
    Args:
        lower (float): lower bound
        upper (float): upper bound
        n (int): number of chunks

    Returns:
        list of pairs
    '''
    end_points = np.linspace(lower, upper, n + 1)

    # Split the belief spectrum into 7 evenly distributed intervals along [0,1]
    intervals = [(end_points[i], end_points[i + 1])
                 for i, _ in enumerate(end_points) if i != n]
    return intervals


def compute_ab(mu):
    '''
        Args:
            mu (float): expected value of the beta distribution given by some a and b

        Returns:
            a and b from that specific distribution
    '''
    var = config.opinion_variance

    a = -(mu * (var + mu**2 - mu) / var)
    b = (var + mu**2 - mu) * (mu - 1) / var

    return a, b


def coin_flip(prob: float = config.communication_speed) -> bool:
    '''
    A single sample from the Bernoulli distribution
    '''
    result = bool(binomial(
        1, prob, 1)[0])

    return result
