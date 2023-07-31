import json
import os
import numpy as np
from typing import List, Dict
import networkx as nx

import dpro

from expert import DynamicGraph
import algo_entry
import parse_trace

import core

dpro.init(".workspace/", "test")

def gen_distribution(worker_num, expert_num, token_num, moe_layer_info, method=0):
    all_worker2token2expert = {}
    for op_name in moe_layer_info:
        expert_list = list(range(expert_num))
        if method == 0:
            prob = np.array(expert_list) / sum(expert_list)
        elif method == 1:
            # Even distribution
            prob = np.ones_like(expert_list, dtype=float) / expert_num
        elif method == 2:
            ### one hot expert with 80 % probability
            prob = np.ones_like(expert_list, dtype=float)
            prob[0] = 0.8
            prob[1:] = (1 - 0.8) / (expert_num - 1)
        elif method == 3:
            ### one hot expert with 80 % probability
            assert expert_num > 3
            prob = np.ones_like(expert_list, dtype=float)
            prob[0] = prob[1] = prob[2] = 0.3
            prob[3:] = (1 - 0.3 * 3) / (expert_num - 3)
        else:
            raise
        
        worker2token2expert = [np.random.choice(expert_list, token_num, p=prob) 
                                for worker_id in range(worker_num)]
        all_worker2token2expert[op_name] = worker2token2expert
    return all_worker2token2expert
        
def run_test(moe_layer_info, dynamic_graph, workspace, distribution_id=0):
    ########################################################################
    ### Test

    METHODS = ["OEM", "FastMoE", "FasterMoE", "Random"]

    rst_path = os.path.join(workspace, "rst.json")
    try:
        with open(rst_path, 'r') as fp:
            rst = json.load(fp)
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        rst = {}
    
    # df.set_index(["TokenDist", "host_num", "local_rank_num", "local_ep_num"])

    ### number of tokens per worker
    token_num = 32
    for host_num, local_rank_num, local_ep_num in [
        [1, 4, 1],
        [1, 4, 2],
        [1, 4, 4],
        [2, 2, 2],
        [2, 2, 4],
        [2, 4, 4],
        [2, 4, 8],
        [2, 4, 16]
    ]:
        rst_key = f"{distribution_id},{host_num},{local_rank_num},{local_ep_num}"
        if rst_key not in rst:
            rst[rst_key] =  [(-1, -1)] * len(METHODS)

        core.HOST_NUM = host_num
        core.LOCAL_RANK_NUM = local_rank_num

        worker_num = host_num  * local_rank_num
        # number of total experts
        expert_num = worker_num * local_ep_num
    
        all_worker2token2expert = gen_distribution(
            worker_num, expert_num, token_num, moe_layer_info, method=distribution_id)

        for method_name, test_func in [
            # ("OEM", algo_entry.test_oem),
            ("FastMoE", algo_entry.test_fast_moe),
            ("FasterMoE", algo_entry.test_faster_moe),
            ("Random", algo_entry.test_random)
        ]:
            print("\n")
            dynamic_graph.reset()
            all_moe_solutions = test_func(moe_layer_info, all_worker2token2expert, worker_num, expert_num)
            balance_ratio = algo_entry.measure_solution_balance_ratio(all_moe_solutions, 
                                        all_worker2token2expert, worker_num)
            G = dynamic_graph.finalize_graph(all_moe_solutions, all_worker2token2expert, worker_num)
            iter_time = DynamicGraph.replay(G, workspace)
            print(method_name, iter_time)
            rst[rst_key][METHODS.index(method_name)] = [iter_time, balance_ratio]

    with open(rst_path, 'w') as fp:
        json.dump(rst, fp, indent=4)

if __name__ == '__main__':
    workspace = ".workspace/traces/toy_example"

    moe_layer_info, dynamic_graph = parse_trace.parse_workload_trace(workspace)

    for distribution_id in [
        0, 
        1, 
        2, 
        3
    ]:
        run_test(moe_layer_info, dynamic_graph, workspace, distribution_id=distribution_id)

