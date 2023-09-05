# %%

from os.path import exists

from tqdm import trange

from social_model import SocialModel
from utils import *

# from d3blocks import D3Blocks

# get_network_stats(G)

# Specifiy which experiment you want to do
exps = config.exp
ws = config.flooding_capacity
bfps = config.bot_follower_percentage
nbrs = config.not_ban_range
ads = config.activation_delays
iranges = config.inoculation_ranges

for exp in exps:
    for bfp in bfps:
        for i in trange(config.simulation_number):
            G = convert_csv_to_graph(f"data/networks/ego_net.csv", sep=",", whole=True)

            # model_df_path = f"data/results/MIM/exp_{exp}_w={w}/result_{i}.csv"
            # agent_df_path = f"data/results/MIM/exp_{exp}_w={w}/agent_result_{i}.csv"

            model_df_path = f"data/results/MIM/exp_{exp}_bfp={bfp}/result_{i}.csv"
            agent_df_path = f"data/results/MIM/exp_{exp}_bfp={bfp}/agent_result_{i}.csv"

            # model_df_path = f'data/results/MIM/exp_{exp}_b=10/result_{i}.csv'
            # agent_df_path = f'data/results/MIM/exp_{exp}_b=10/agent_result_{i}.csv'

            # model_df_path = f'data/results/MIM/exp_{exp}_br=0.02/result_{i}.csv'

            # model_df_path = f'data/results/MIM/exp_{exp}_ad={ad}/result_{i}.csv'

            # model_df_path = f"data/results/MIM/exp_{exp}_irate={irate}/result_{i}.csv"

            # model_df_path = f"data/results/MIM/exp_{exp}_irange={irange[0]}-{irange[1]}/result_{i}.csv"

            # model_df_path = f"data/results/MIM/exp_{exp}_w=25/result_{i}.csv"
            # agent_df_path = f"data/results/MIM/exp_{exp}_w=25/agent_result_{i}.csv"

            if exists(model_df_path):
                print(f"{model_df_path} exists, passing")
                continue

            if exists(agent_df_path):
                print(f"{agent_df_path} exists, passing")
                continue

            # Initialise the agents based on the agent_parameters
            social_model = SocialModel(
                G,
                seed=i,
                exp=exp,
                flooding_capacity=25,
                bot_follower_percentage=bfp,
                activation_delay=0,
                inoculation_rate=0.2,
                inoculation_range=[0.2, 0.8],
                from_scratch=True,
                param_index=i,
                collect_agent_data=True,
            )

            for step in trange(config.simulation_periods):
                social_model.step()

            model_df = social_model.model_data_collector.get_model_vars_dataframe()
            model_df.to_csv(model_df_path, index=False)

            agent_df = social_model.agent_data_collector.get_agent_vars_dataframe()
            agent_df.to_csv(agent_df_path, index=False)

        # print(agent_dfs)
