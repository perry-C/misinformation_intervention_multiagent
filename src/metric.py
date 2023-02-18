from statistics import mean

import config


class Metric:
    def misinformation(model):
        misinfo = mean(
            [a.calculate_opinion() - config.truth for a in model.get_agents()])
        return misinfo
