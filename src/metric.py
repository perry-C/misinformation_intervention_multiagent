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

        # Initialise 7 groups
        alpha = 0.5
        groups = [np.array([])
                  for _ in range(config.belief_distribution_groups)]

        # Distribute all agents into the corresponding group based on their opinion
        opinions = [a.calculate_opinion()
                    for a in model.get_agents()]

        lower = min(opinions)
        upper = max(opinions)

        opinion_intervals = get_intervals(lower, upper, len(groups))

        for op in opinions:
            for i, interval in enumerate(opinion_intervals):
                low, high = interval
                if low <= op <= high:
                    groups[i] = np.append(groups[i], op)
                    break

        group_wise_pols = []

        # Implement the formula
        for i in range(config.belief_distribution_groups):
            for j in range(config.belief_distribution_groups):
                i_ratio = len(groups[i]) / \
                    model.schedule.get_agent_count()

                j_ratio = len(groups[j]) / \
                    model.schedule.get_agent_count()
                i_average_opinion = np.average(groups[i])
                j_average_opinion = np.average(groups[j])

                '''
                from azzimonti2022social:
                "The mutiplication by 2 is a normalization such that Pol = 1
                when a=0, two groups equally sized, group opinions are 0 and 1."
                '''
                K = 1 / (2 * (0.5) ** (2 + alpha))

                group_wise_pol = K * i_ratio ** (1+alpha) * j_ratio * \
                    abs(i_average_opinion - j_average_opinion)
                group_wise_pols.extend([group_wise_pol])

        polarization = sum(group_wise_pols)
        return polarization

    def average_opinion_all(model):
        average_opinion = mean(
            [a.calculate_opinion() for a in model.get_agents()])
        return average_opinion

    def average_opinion_left(model):
        average_opinion = mean(
            [a.calculate_opinion() for a in model.get_agents() if a.bot_id == model.l_bot_id])
        return average_opinion

    def average_opinion_right(model):
        average_opinion = mean(
            [a.calculate_opinion() for a in model.get_agents() if a.bot_id == model.r_bot_id])
        return average_opinion
