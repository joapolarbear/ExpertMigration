import json
import os
import numpy as np
from typing import List, Dict
import networkx as nx

import dpro
from dpro.trace_utils import gen_long_name

from expert import Expert, MoELayer, DynamicGraph, MoESolution
import core

dpro.init(".workspace/", "test")

def is_expert_event(event):
    return event["args"]["op_name"].startswith("MoE_Experts")

def gen_rawname(phase, op_name):
    return f"{phase}.{op_name}"

def parse_rawtrace(path=".workspace/rawtrace.json"):
    with open(path, 'r') as fp:
        rst = json.load(fp)

    RANK = rst["rank"]
    EP_WORLD_SIZE = rst["ep_world_size"]
    START_TIME = rst["start_time"]

    dynamic_graph = DynamicGraph(EP_WORLD_SIZE)

    events = []
    op_stat = {}
    moe_layer_info = {}
    unsure_event_ids = set()
    G = nx.DiGraph()

    prev_end_t = START_TIME * 1e6
    prev_trace_name = None
    for phase, name, end_t, attr in rst["timestamps"]:
        end_t *= 1e6
        if phase == "FW_PRE":
            prev_end_t = end_t
            continue
        
        dur = end_t - prev_end_t
        trace_name = gen_rawname(phase, name)
        event = {
            "ph": "X",
            "name": trace_name,
            "pid": "computation",
            "ts": prev_end_t,
            "dur": dur,
            "args": {
                "phase": phase,
                "op_name": name,
                "end_t": end_t,
            }
        }
        event_id = len(events)
        events.append(event)

        if name not in op_stat:
            op_stat[name] = {}
        if phase == "FW":
            op_stat[name]["fw_event_id"] = event_id
        elif phase == "BW":
            op_stat[name]["bw_event_id"] = event_id
        elif phase == "UPDATE_":
            op_stat[name]["update_event_id"] = event_id
        else:
            raise

        print(f"Parse event {trace_name}")
        if is_expert_event(event):
            if phase == "FW":
                # of shape [N_worker, N_expert/worker, capacity, d_model] or [N, E/N, C, M]
                N, E_div_N, C, M = attr
                E = N * E_div_N
                event["args"]["NECM"] = (N, E, C, M)

                experts = [Expert(expert_id, (N, E, C, M), moe_layer_id=name) for expert_id in range(E)]
                moe_layer_info[name] = MoELayer(experts)
            else:
                unsure_event_ids.add(event_id)
            ### Collect info for graph construction
            dynamic_graph.met_expert_for_all_worker(moe_layer_info[name], phase)
            prev_trace_name = None
        else:
            ### Collect info for graph construction
            if prev_trace_name:
                dynamic_graph.add_comp_edge_for_all_worker(prev_trace_name, trace_name)
            prev_trace_name = trace_name
            
            if phase == "BW" and len(events) > 1 and is_expert_event(events[-2]):
                unsure_event_ids.add(event_id)

            if name.startswith("MoE_Gate") and phase == "FW":
                mask_wo_drop, mask_w_drop = attr
                event["args"]["mask_wo_drop"] = mask_wo_drop
                event["args"]["mask_w_drop"] = mask_w_drop
            else:
                pass

        prev_end_t = end_t
    
    return op_stat, events, unsure_event_ids, moe_layer_info, dynamic_graph

def cal_bw_to_fw_ratio(op_stat: Dict, events: List, unsure_event_ids: set):
    ### Estimate the ratio between bw and fw
    non_expert_fw_accum = []
    non_expert_bw_accum = []
    for op_name in op_stat.keys():
        if "update_event_id" in op_stat[op_name]:
            continue
        if op_stat[op_name]["fw_event_id"] in unsure_event_ids or \
            op_stat[op_name]["bw_event_id"] in unsure_event_ids:
            pass
        non_expert_fw_accum.append(events[op_stat[op_name]["fw_event_id"]]["dur"])
        non_expert_bw_accum.append(events[op_stat[op_name]["bw_event_id"]]["dur"])
    ratio_list = np.array([bw_t / fw_t for bw_t, fw_t in 
                zip(non_expert_bw_accum, non_expert_fw_accum)])
    bw_to_fw_ratio = np.median(ratio_list)
    # bw_to_fw_ratio = np.mean(ratio_list)
    return bw_to_fw_ratio
    

def correct_unsure_event_ts(op_stat: Dict, events: List, unsure_event_ids: set, 
         moe_layer_info: Dict[str, MoELayer], bw_to_fw_ratio: float):
    # Correct the duration of Expert ops,
    # Calcudate the execution time of AllToAll and add corrresponding events to the traces
    for expert_name in moe_layer_info:
        ### Given
        expert_fw_event_id = op_stat[expert_name]["fw_event_id"]
        expert_bw_event_id = op_stat[expert_name]["bw_event_id"]

        a_fw_ed = events[expert_fw_event_id-1]["args"]["end_t"]
        c_fw_ed = events[expert_fw_event_id]["args"]["end_t"]
        e_fw_ed = events[expert_fw_event_id+1]["args"]["end_t"]

        e_bw_ed = events[expert_bw_event_id-1]["args"]["end_t"]
        c_bw_ed = events[expert_bw_event_id]["args"]["end_t"]
        a_bw_ed = events[expert_bw_event_id+1]["args"]["end_t"]

        a_fw_dur = events[expert_fw_event_id-1]["dur"]

        c_fw_st = events[expert_fw_event_id]["ts"]
        c_fw_dur = events[expert_fw_event_id]["dur"]
        e_fw_st = events[expert_fw_event_id+1]["ts"]

        ### Solution for the communication time
        x_list = []
        def _check(rst):
            if rst > 0:
                x_list.append(rst)
            return rst
        x1 = _check(c_fw_st - a_fw_ed)
        x2 = _check(c_bw_ed - e_bw_ed - bw_to_fw_ratio * c_fw_dur)
        x3 = _check(e_fw_st - c_fw_ed)
        x4 = _check(a_bw_ed - c_bw_ed - bw_to_fw_ratio * a_fw_dur)
        # print(x_list)
        t_all_to_all = min(x_list)

        ### FW timestamps
        b_fw_st = a_fw_ed; b_fw_dur = t_all_to_all
        d_fw_st = c_fw_ed; d_fw_dur = t_all_to_all

        ### BW timestamps
        d_bw_st = e_bw_ed; d_bw_dur = t_all_to_all
        c_bw_dur = bw_to_fw_ratio * c_fw_dur; c_bw_st = c_bw_ed - c_bw_dur,   
        b_bw_st = c_bw_ed; b_bw_dur = t_all_to_all
        a_bw_dur = bw_to_fw_ratio * a_fw_dur; a_bw_st = a_bw_ed - a_bw_dur

        unsure_event_ids.remove(expert_bw_event_id)
        unsure_event_ids.remove(expert_bw_event_id+1)

        ### Modify traces
        ####################### FW
        #### trace b, fw pre-calc
        op_stat[expert_name]["pre-calc_FW_event_id"] = len(events)
        events.append({
            "ph": "X",
            "name": gen_long_name(None, f"Comm.{expert_name}", "pre-calc_FW"),
            "pid": "communication",
            "ts": b_fw_st,
            "dur": b_fw_dur
        })
        #### trace d, fw post-calc
        op_stat[expert_name]["post-calc_FW_event_id"] = len(events)
        events.append({
            "ph": "X",
            "name": gen_long_name(None, f"Comm.{expert_name}", "post-calc_FW"),
            "pid": "communication",
            "ts": d_fw_st,
            "dur": d_fw_dur
        })
        ####################### BW
        #### trace d, bw pre-calc
        op_stat[expert_name]["pre-calc_BW_event_id"] = len(events)
        events.append({
            "ph": "X",
            "name": gen_long_name(None, f"Comm.{expert_name}", "pre-calc_BW"),
            "pid": "communication",
            "ts": d_bw_st,
            "dur": d_bw_dur
        })
        #### trace c, expert BW
        events[expert_bw_event_id]["ts"] = c_bw_st
        events[expert_bw_event_id]["dur"] = c_bw_dur
        #### trace b, bw post-calc
        op_stat[expert_name]["post-calc_BW_event_id"] = len(events)
        events.append({
            "ph": "X",
            "name": gen_long_name(None, f"Comm.{expert_name}", "post-calc_BW"),
            "pid": "communication",
            "ts": b_bw_st,
            "dur": b_bw_dur
        })
        #### trace a, BW
        events[expert_bw_event_id+1]["ts"] = a_bw_st
        events[expert_bw_event_id+1]["dur"] = a_bw_dur
    assert len(unsure_event_ids) == 0, unsure_event_ids

def dump_traces(events, workspace=".workspace"):
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, "trace.json"), 'w') as fp:
        json.dump({
            "traceEvents": events
        }, fp, indent=4)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "3rdparty", "OEM-MoE"))
import algorithm as moe_scheduler

def migrate_expert(expert2worker, source_worker, target_worker, ep_id):
    assert expert2worker[ep_id] == source_worker
    expert2worker[ep_id] = target_worker


def test_oem(moe_layer_info, all_worker2token2expert):
    all_moe_solutions = {}

    for op_name in moe_layer_info:
        moe_layer = moe_layer_info[op_name]
        N, E, C, M = moe_layer.NECM

        # capacity for each worker
        # TODO (huhanpeng): use the real value
        memory_cap_of_each_worker = [1000 for _ in range(N)]
        param_size_of_expert = np.ones(E, dtype=float) * 3

        worker2token2expert = all_worker2token2expert[op_name]
        token_target_expert = np.zeros((N, E))
        for worker_id in range(N):
            for expert_id in worker2token2expert[worker_id]:
                token_target_expert[worker_id][expert_id] += 1
        del worker_id, expert_id
        
        rst = moe_scheduler.opt.oem_solver(N, E,
                memory_cap_of_each_worker,
                token_target_expert,
                param_size_of_expert,
                num_time_slots = 30,
                cache_dir = ".workspace/OEM"
            )

        expert2worker = [ep_id // (E // N) for ep_id in range(E)]
        for ep_id in range(E):
            for source_worker_id in range(M):
                for target_worker_id in range(M):
                    if rst.s[ep_id, source_worker_id, target_worker_id] == 1:
                        migrate_expert(expert2worker,
                            source_worker_id, target_worker_id, ep_id)

        all_moe_solutions[op_name] = MoESolution(expert2worker, worker2token2expert)

    return all_moe_solutions

def test_fast_moe(moe_layer_info, all_worker2token2expert):
    all_moe_solutions = {}
    for op_name in moe_layer_info:
        moe_layer = moe_layer_info[op_name]
        N, E, C, M = moe_layer.NECM
        expert2worker = [ep_id // (E // N) for ep_id in range(E)]
        all_moe_solutions[op_name] = MoESolution(expert2worker)

    return all_moe_solutions

def test_faster_moe(moe_layer_info, all_worker2token2expert):
    all_moe_solutions = {}
    for op_name in moe_layer_info:
        moe_layer = moe_layer_info[op_name]
        N, E, C, M = moe_layer.NECM
        expert2worker = [ep_id // (E // N) for ep_id in range(E)]

        worker2token2expert = all_worker2token2expert[op_name]
        all_global_expert_count = np.zeros(E)
        for worker_id in range(N):
            for expert_id in worker2token2expert[worker_id]:
                all_global_expert_count[expert_id] += 1
        expert_shadow = moe_scheduler.shadow_policy.shadow_policy(
            all_global_expert_count, M, E)

        all_moe_solutions[op_name] = MoESolution(expert2worker, expert_shadow)

    return all_moe_solutions

def test_random(moe_layer_info, all_worker2token2expert):
    return all_moe_solutions

if __name__ == '__main__':      
    op_stat, events, unsure_event_ids, moe_layer_info, dynamic_graph = parse_rawtrace()

    bw_to_fw_ratio = cal_bw_to_fw_ratio(op_stat, events, unsure_event_ids)
    print("bw_to_fw_ratio ", bw_to_fw_ratio)

    correct_unsure_event_ts(op_stat, events, unsure_event_ids, 
                            moe_layer_info, bw_to_fw_ratio)
        
    dump_traces(events)

    ### Assign node avg
    for op_name in op_stat:
        if op_name in moe_layer_info:
            moe_layer = moe_layer_info[op_name]
            moe_layer.set_comm_time(op_stat, events, bw_to_fw_ratio)
        elif "update_event_id" in op_stat[op_name]:
            ### Update ops
            rawname = gen_rawname("UPDATE_", op_name)
            avg = events[op_stat[op_name]["update_event_id"]]["dur"] / 1e3
            dynamic_graph.set_avg_for_all_worker(rawname, avg)
        else:
            rawname = gen_rawname("FW", op_name)
            avg = events[op_stat[op_name]["fw_event_id"]]["dur"] / 1e3
            dynamic_graph.set_avg_for_all_worker(rawname, avg)

            rawname = gen_rawname("BW", op_name)
            avg = events[op_stat[op_name]["bw_event_id"]]["dur"] / 1e3
            dynamic_graph.set_avg_for_all_worker(rawname, avg)

    token_num = 5
    all_worker2token2expert = {}
    for op_name in moe_layer_info:
        moe_layer = moe_layer_info[op_name]
        N, E, C, M = moe_layer.NECM
        expert_list = list(range(E))
        worker2token2expert = [np.random.choice(expert_list, token_num) 
                               for worker_id in range(N)]

    all_moe_solutions = test_oem(moe_layer_info, all_worker2token2expert)
    G = dynamic_graph.finalize_graph(all_moe_solutions, all_worker2token2expert)
    DynamicGraph.replay(G)
