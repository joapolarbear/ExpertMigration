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

def run_test(moe_layer_info, dynamic_graph):
    ########################################################################
    ### Test
    ### number of tokens per worker
    token_num = 32
    for host_num, local_rank_num, local_ep_num in [
        # [1, 4, 1],
        # [1, 4, 2],
        # [1, 4, 4],
        # [2, 2, 2],
        # [2, 2, 4],
        # [2, 4, 4],
        # [2, 4, 8],
        [2, 4, 16]
    ]:
        core.HOST_NUM = host_num
        core.LOCAL_RANK_NUM = local_rank_num

        worker_num = host_num  * local_rank_num
        # number of total experts
        expert_num = worker_num * local_ep_num
    
        all_worker2token2expert = {}
        for op_name in moe_layer_info:
            expert_list = list(range(expert_num))
            prob = np.array(expert_list) / sum(expert_list)
            worker2token2expert = [np.random.choice(expert_list, token_num, p=prob) 
                                for worker_id in range(worker_num)]
            all_worker2token2expert[op_name] = worker2token2expert

        # dynamic_graph.reset()
        # all_moe_solutions = algo_entry.test_oem(moe_layer_info, all_worker2token2expert, worker_num, expert_num)
        # G = dynamic_graph.finalize_graph(all_moe_solutions, all_worker2token2expert, worker_num)
        # DynamicGraph.replay(G)

        dynamic_graph.reset()
        all_moe_solutions = algo_entry.test_fast_moe(moe_layer_info, all_worker2token2expert, worker_num, expert_num)
        G = dynamic_graph.finalize_graph(all_moe_solutions, all_worker2token2expert, worker_num)
        DynamicGraph.replay(G)

        dynamic_graph.reset()
        all_moe_solutions = algo_entry.test_faster_moe(moe_layer_info, all_worker2token2expert, worker_num, expert_num)
        G = dynamic_graph.finalize_graph(all_moe_solutions, all_worker2token2expert, worker_num)
        DynamicGraph.replay(G)

        dynamic_graph.reset()
        all_moe_solutions = algo_entry.test_random(moe_layer_info, all_worker2token2expert, worker_num, expert_num)
        G = dynamic_graph.finalize_graph(all_moe_solutions, all_worker2token2expert, worker_num)
        DynamicGraph.replay(G)

if __name__ == '__main__':  
    moe_layer_info, dynamic_graph = parse_trace.parse_workload_trace(".workspace/traces/toy_example")

    run_test(moe_layer_info, dynamic_graph)

