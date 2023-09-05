import csv
import os
import random
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd

import config
from parameter import AgentParameter
from utils import compute_ab, get_intervals


def read_agent_params(index):
    aps = []
    lfs = []
    rfs = []

    with open(f"data/params/agent_abs_{index}.csv") as fp:
        reader = csv.reader(fp, delimiter=",")
        for row in reader:
            ap = AgentParameter(float(row[0]), float(row[1]))
            aps.extend([ap])

    with open(f"data/params/l_bot_followed_{index}.csv") as fp:
        reader = csv.reader(fp, delimiter="\n")
        for row in reader:
            lf = int(row[0]) - 1
            lfs.extend([lf])

    with open(f"data/params/r_bot_followed_{index}.csv") as fp:
        reader = csv.reader(fp, delimiter="\n")
        for row in reader:
            rf = int(row[0]) - 1
            rfs.extend([rf])

    return aps, lfs, rfs


def generate_agent_params(seed, agent_number, exp, flooding_capacity, innoculate_range):
    """
    Generate the abs used for determining each agent's belief at the start of each simulation

    "This rule basically distributes our agents evenly over the belief spectrum [0, 1] such that each
    1/7 of the total mass of agents in the initial period."
    """
    random.seed(seed)
    K = config.belief_distribution_groups

    nodes_intervals = np.linspace(
        0, agent_number + config.left_bot_number + config.right_bot_number, K + 1
    )

    # Split the belief spectrum into K (7) evenly distributed intervals along [0,1]
    belief_intervals = get_intervals(0, 1, K)

    # Split the nodes into K even groups
    group_sizes = [
        round(nodes_intervals[i + 1]) - round(nodes_intervals[i])
        for i, _ in enumerate(nodes_intervals)
        if i != K
    ]

    agent_params = []

    for group_index in range(K):
        for _ in range(group_sizes[group_index]):
            lower, upper = belief_intervals[group_index]
            ex = random.uniform(lower, upper)
            a, b = compute_ab(ex)
            agent_params.extend(
                [AgentParameter(a, b, exp, flooding_capacity, innoculate_range)]
            )
    random.shuffle(agent_params)

    return agent_params


def pickle_network(network_path):
    fb_network = open(f"{network_path}.txt", "r").read()
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


def get_SFP_df():
    df1 = pd.read_csv("data/results/SFP/DF-op.csv", header=None)
    df2 = pd.read_csv("data/results/SFP/DF-opaud0.csv", header=None)
    df3 = pd.read_csv("data/results/SFP/DF-opaud1.csv", header=None)
    df4 = pd.read_csv("data/results/SFP/DFpol.csv", header=None)
    df5 = pd.read_csv("data/results/SFP/DFmisinf.csv", header=None)

    df1.drop(columns=[0], inplace=True)
    df2.drop(columns=[0], inplace=True)
    df3.drop(columns=[0], inplace=True)
    df4.drop(columns=[0], inplace=True)
    df5.drop(columns=[0], inplace=True)

    SFP_y1 = df1.mean(axis=0)
    SFP_y2 = df2.mean(axis=0)
    SFP_y3 = df3.mean(axis=0)
    SFP_y4 = df4.mean(axis=0)
    SFP_y5 = df5.mean(axis=0)

    SFP_df = pd.concat([SFP_y1, SFP_y2, SFP_y3, SFP_y4, SFP_y5], axis=1)
    SFP_df.rename(
        columns={
            0: "average_opinion_all",
            1: "average_opinion_left",
            2: "average_opinion_right",
            3: "polarization",
            4: "misinformation",
        },
        inplace=True,
    )
    SFP_df.reset_index(inplace=True)

    return SFP_df


def get_MIM_df(exp: str):
    """_summary_

    Args:
        exp (str): exp_w=

    Returns:
        modeldf: an average of model-wise measurements across 10 experiments
        agentdf: an average of agnent-wise opinion across 10 experiments
    """

    folder_path = f"data/results/MIM/exp_{exp}/"

    model_results = glob(folder_path + "result_*.csv")
    agent_results = glob(folder_path + "agent_result_*.csv")

    model_dfs = []
    agent_dfs = []

    for file in model_results:
        df = pd.read_csv(file, sep=",")
        model_dfs.append(df)

    for file in agent_results:
        df = pd.read_csv(file, sep=",")

        # ===================================================
        # df_mod = df.drop(
        #     columns=['opinion_l_bf', 'opinion_r_bf', 'opinion_l_bot', 'opinion_r_bot'])

        # steps = sum([[s] * (df_mod.shape[0]//11)
        #             for s in range(0, 1100, 100)], [])

        # df_mod['step'] = steps
        # df_mod.to_csv(file, index=False)
        # ===================================================

        agent_dfs.append(df)
    if model_dfs:
        merged_df = pd.concat(model_dfs)
        by_row_index = merged_df.groupby(merged_df.index)
        MIM_model_df = by_row_index.mean()
    else:
        MIM_model_df = None

    if agent_dfs:
        merged_df = pd.concat(agent_dfs)
        by_row_index = merged_df.groupby(merged_df.index)
        MIM_agent_df = by_row_index.mean()
    else:
        MIM_agent_df = None
    return MIM_model_df, MIM_agent_df


def get_MIM_df_no_merge(exp: str):
    """_summary_

    Args:
        exp (str): exp_w=

    Returns:
        modeldf: an average of model-wise measurements across 10 experiments
        agentdf: an average of agnent-wise opinion across 10 experiments
    """

    folder_path = f"data/results/MIM/exp_{exp}/"

    model_results = glob(folder_path + "result_*.csv")
    agent_results = glob(folder_path + "agent_result_*.csv")

    model_dfs = []
    agent_dfs = []

    for file in model_results:
        df = pd.read_csv(file, sep=",")
        model_dfs.append(df)

    for file in agent_results:
        df = pd.read_csv(file, sep=",")

        # ===================================================
        # df_mod = df.drop(
        #     columns=['opinion_l_bf', 'opinion_r_bf', 'opinion_l_bot', 'opinion_r_bot'])

        # steps = sum([[s] * (df_mod.shape[0]//11)
        #             for s in range(0, 1100, 100)], [])

        # df_mod['step'] = steps
        # df_mod.to_csv(file, index=False)
        # ===================================================

        agent_dfs.append(df)
    if model_dfs:
        MIM_model_df = model_dfs
        # by_row_index = merged_df.groupby(merged_df.index)
        # MIM_model_df = by_row_index.mean()
    else:
        MIM_model_df = None

    if agent_dfs:
        MIM_agent_df = agent_dfs
        # merged_df = pd.concat(agent_dfs)
        # by_row_index = merged_df.groupby(merged_df.index)
    else:
        MIM_agent_df = None
    return MIM_model_df, MIM_agent_df
