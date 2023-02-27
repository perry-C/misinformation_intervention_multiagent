from statistics import mean

import numpy as np

import config


class Metric:
    def misinformation(model):
        misinfo = mean(
            [(a.calculate_opinion() - config.truth)**2 for a in model.get_agents()])
        return misinfo

    # todo: change this to the definition of the paper

    def polarization(income, group):
        # calculate total income
        total_income = np.sum(income)

        # calculate shares of income and population for each group
        group_income_share = []
        group_pop_share = []
        for i in np.unique(group):
            group_income_share.append(
                np.sum(income[group == i]) / total_income)
            group_pop_share.append(np.sum(group == i) / len(group))

        # calculate the Gini index of income inequality
        gini = 1 - 2 * np.sum(np.multiply(group_income_share,
                                          1 - np.array(group_income_share)))

        # calculate the entropy index of segregation
        entropy = -np.sum(np.multiply(group_pop_share,
                          np.log(group_pop_share)))

        # calculate polarization
        polarization = gini * entropy

        return polarization

    def average_opinion(model):
        average_opinion = mean(
            [a.calculate_opinion() for a in model.get_agents()])
        return average_opinion
