import json
import os
import numpy as np
from typing import List, Dict
import networkx as nx

import dpro
from dpro.trace_utils import gen_long_name

from expert import Expert, MoELayer, DynamicGraph, MoESolution
import core


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "3rdparty", "OEM-MoE"))
from algorithm import opt, shadow_policy

def migrate_expert(expert2worker, source_worker, target_worker, ep_id):
    assert expert2worker[ep_id] == source_worker
    expert2worker[ep_id] = target_worker

def _workload_metric(all_token_to_expert, expert2worker, worker_num):
    load_per_worker = np.zeros(worker_num)
    for ep_id in all_token_to_expert:
        load_per_worker[expert2worker[ep_id]] += 1
    load_per_worker /= sum(load_per_worker)
    return np.linalg.norm(load_per_worker)

def measure_solution_balance_ratio(all_moe_solutions, all_worker2token2expert, worker_num):
    all_metrics = []
    for op_name in all_moe_solutions:
        worker2token2expert = all_worker2token2expert[op_name]
        all_token_to_expert = np.array(worker2token2expert).flatten()
        metric = _workload_metric(all_token_to_expert, 
                                all_moe_solutions[op_name].expert2worker, worker_num)
        all_metrics.append(metric)
    return np.mean(all_metrics)

def test_oem(moe_layer_info, all_worker2token2expert, worker_num, expert_num):
    all_moe_solutions = {}

    for op_name in moe_layer_info:
        # capacity for each worker
        # TODO (huhanpeng): use the real value
        memory_cap_of_each_worker = [1000 for _ in range(worker_num)]
        param_size_of_expert = np.ones(expert_num, dtype=float) * 3

        worker2token2expert = all_worker2token2expert[op_name]
        token_target_expert = np.zeros((worker_num, expert_num))
        for worker_id in range(worker_num):
            for expert_id in worker2token2expert[worker_id]:
                token_target_expert[worker_id][expert_id] += 1
        del worker_id, expert_id
        
        rst = opt.oem_solver(worker_num, expert_num,
                memory_cap_of_each_worker,
                token_target_expert,
                param_size_of_expert,
                num_time_slots = 30,
                cache_dir = ".workspace/OEM"
            )

        expert2worker = [ep_id // (expert_num // worker_num) for ep_id in range(expert_num)]
        for ep_id in range(expert_num):
            for source_worker_id in range(worker_num):
                for target_worker_id in range(worker_num):
                    if rst.s[ep_id, source_worker_id, target_worker_id] == 1:
                        migrate_expert(expert2worker,
                            source_worker_id, target_worker_id, ep_id)
        print(f"[OEM Policy] use mapping {expert2worker} for moe layer {op_name}")
        all_moe_solutions[op_name] = MoESolution(expert2worker)

    _workload_metric(all_token_to_expert, expert2worker, worker_num)

    return all_moe_solutions

def test_fast_moe(moe_layer_info, all_worker2token2expert, worker_num, expert_num):
    all_moe_solutions = {}
    for op_name in moe_layer_info:
        expert2worker = [ep_id // (expert_num // worker_num) for ep_id in range(expert_num)]
        all_moe_solutions[op_name] = MoESolution(expert2worker)

    return all_moe_solutions

def test_faster_moe(moe_layer_info, all_worker2token2expert, worker_num, expert_num):
    all_moe_solutions = {}
    for op_name in moe_layer_info:
        moe_layer = moe_layer_info[op_name]
        N, E, C, M = moe_layer.NECM
        expert2worker = [ep_id // (expert_num // worker_num) for ep_id in range(expert_num)]

        worker2token2expert = all_worker2token2expert[op_name]
        all_global_expert_count = np.zeros(expert_num)
        for worker_id in range(worker_num):
            for expert_id in worker2token2expert[worker_id]:
                all_global_expert_count[expert_id] += 1
        expert_shadow = shadow_policy.shadow_policy(
            all_global_expert_count, M, expert_num)
        
        print(f"Shadow decision ", np.where(np.array(expert_shadow) == True))

        all_moe_solutions[op_name] = MoESolution(expert2worker, expert_shadow)

    return all_moe_solutions

import random

def _random_change_mapping(expert2worker, num_worker):
    assert num_worker > 1
    rand_idx = random.randint(0, len(expert2worker)-1)
    while True:
        new_worker_id = random.randint(0, num_worker-1)
        if new_worker_id != expert2worker[rand_idx]:
            ret = np.copy(expert2worker)
            ret[rand_idx] = new_worker_id
            return ret

def test_random(moe_layer_info, all_worker2token2expert, worker_num, expert_num):
    all_moe_solutions = {}
    for op_name in moe_layer_info:
        expert2worker = [ep_id // (expert_num // worker_num) for ep_id in range(expert_num)]
        worker2token2expert = all_worker2token2expert[op_name]
        all_token_to_expert = np.array(worker2token2expert).flatten()
        metric = _workload_metric(all_token_to_expert, expert2worker, worker_num)
        for _ in range(15):
            tmp_mapping = _random_change_mapping(expert2worker, worker_num)
            tmp_metric = _workload_metric(all_token_to_expert, tmp_mapping, worker_num)
            # print(metric, tmp_metric)
            if tmp_metric < metric:
                # print(f" - [Random policy] change mapping to {tmp_mapping} for {op_name}")
                metric = tmp_metric
                expert2worker = tmp_mapping

        # print(f"[Random policy] use {expert2worker} for moe layer {op_name}")

        all_moe_solutions[op_name] = MoESolution(expert2worker)

    return all_moe_solutions