from statistics import mean

import numpy as np

import config
from utils import get_intervals


class Metric:
    def misinformation(model):
        misinfo = mean(
            [(a.calculate_opinion() - config.truth)**2 for a in model.get_agents()])
        return misinfo

    # todo: change this to the definition of the paper

    def polarization(model):
        '''
        Original definition from Esteban94
        '''

        # Initialise K groups
        K = config.belief_distribution_groups
        alpha = 0.5
        groups = [np.array([])
                  for _ in range(K)]

        # Distribute all agents into the corresponding group based on their opinion
        opinions = [a.calculate_opinion()
                    for a in model.get_agents()]

        lower = min(opinions)
        upper = max(opinions)

        opinion_intervals = get_intervals(lower, upper, K)

        for op in opinions:
            for i, interval in enumerate(opinion_intervals):
                low, high = interval
                if low <= op <= high:
                    groups[i] = np.append(groups[i], op)
                    break
                if i == K - 1:
                    print(op)

        group_wise_pols = []

        # Implement the formula
        for i in range(K):
            for j in range(K):
                k_ratio = len(groups[i]) / \
                    model.schedule.get_agent_count()

                l_ratio = len(groups[j]) / \
                    model.schedule.get_agent_count()
                k_average_opinion = np.average(groups[i])
                l_average_opinion = np.average(groups[j])

                '''
                from azzimonti2022social:
                "The mutiplication by 2 is a normalization such that Pol = 1
                when a=0, two groups equally sized, group opinions are 0 and 1."
                '''
                K = 1 / (2 * (0.5) ** (2 + alpha))

                group_wise_pol = k_ratio ** (1+alpha) * l_ratio * \
                    abs(k_average_opinion - l_average_opinion) * norm
                group_wise_pols.extend([group_wise_pol])

        polarization = sum(group_wise_pols)
        return polarization

    def average_opinion(model):
        average_opinion = mean(
            [a.calculate_opinion() for a in model.get_agents()])
        return average_opinion
