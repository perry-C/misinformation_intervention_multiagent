from numpy.random import binomial

import config


def bernoulli_draw() -> bool:
    '''
    A single sample from the Bernoulli distribution
    '''
    result = bool(binomial(
        1, config.speed_of_communication, 1)[0])

    return result
