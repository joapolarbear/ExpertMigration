import json
import os
import numpy as np

from dpro.trace_utils import gen_long_name

with open("rawtrace.json", 'r') as fp:
    rst = json.load(fp)

def is_expert_event(event):
    return event["args"]["name"].startswith("MoE_Experts")

RANK = rst["rank"]
EP_WORLD_SIZE = rst["ep_world_size"]
START_TIME = rst["start_time"]

prev_end_t = START_TIME * 1e6
prev_name = None
events = []
op_stat = {}
### Map event ID to moe layer info
expert_op_set = set()
non_expert_fw_accum = []
non_expert_bw_accum = []
unsure_event_ids = set()
for phase, name, end_t, attr in rst["timestamps"]:
    end_t *= 1e6
    if phase == "FW_PRE":
        prev_end_t = end_t
        prev_name = name
        continue
    elif phase == "FW":
        assert name == prev_name, (name, prev_name)
        prev_name = None
    dur = end_t - prev_end_t
    trace_name = f"{phase}.{name}"
    event = {
        "ph": "X",
        "name": trace_name,
        "pid": "computation",
        "ts": prev_end_t,
        "dur": dur,
        "args": {
            "phase": phase,
            "name": name,
            "end_t": end_t,
        }
    }
    event_id = len(events)
    events.append(event)

    if name not in op_stat:
        op_stat[name] = {}
    if phase == "FW":
        op_stat[name]["fw_event_id"] = event_id
    else:
        op_stat[name]["bw_event_id"] = event_id

    if is_expert_event(event):
        if phase == "FW":
            expert_op_set.add(name)
            # of shape [N_worker, N_expert/worker, capacity, d_model] or [N, E/N, C, M]
            N, E_div_N, C, M = attr
            E = N * E_div_N
            event["args"]["NECM"] = (N, E, C, M)
        else:
            unsure_event_ids.add(event_id)
    else:
        if phase == "BW" and len(events) > 1 and is_expert_event(events[-2]):
            unsure_event_ids.add(event_id)

        if name.startswith("MoE_Gate"):
            if phase == "FW":
                mask_wo_drop, mask_w_drop = attr
                event["args"]["mask_wo_drop"] = mask_wo_drop
                event["args"]["mask_w_drop"] = mask_w_drop
        else:
            pass

    prev_end_t = end_t

for op_name in op_stat.keys():
    if op_stat[op_name]["fw_event_id"] in unsure_event_ids or \
        op_stat[op_name]["bw_event_id"] in unsure_event_ids:
        pass
    non_expert_fw_accum.append(events[op_stat[op_name]["fw_event_id"]]["dur"])
    non_expert_bw_accum.append(events[op_stat[op_name]["bw_event_id"]]["dur"])
    
ratio_list = np.array([bw_t / fw_t for bw_t, fw_t in 
            zip(non_expert_bw_accum, non_expert_fw_accum)])
bw_to_fw_ratio = np.median(ratio_list)
# bw_to_fw_ratio = np.mean(ratio_list)

print("bw_to_fw_ratio ", bw_to_fw_ratio)
correct = True
if correct:
    # Correct the duration of Expert ops,
    # Calcudate the execution time of AllToAll and add corrresponding events to the traces
    for expert_name in expert_op_set:
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
        x1 = _check(c_fw_st - a_fw_ed)
        x2 = _check(c_bw_ed - e_bw_ed - bw_to_fw_ratio * c_fw_dur)
        x3 = _check(e_fw_st - c_fw_ed)
        x4 = _check(a_bw_ed - c_bw_ed - bw_to_fw_ratio * a_fw_dur)
        print(x_list)
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

workspace = ".workspace"
os.makedirs(workspace, exist_ok=True)
with open(os.path.join(workspace, "trace.json"), 'w') as fp:
    json.dump({
        "traceEvents": events
    }, fp, indent=4)


        
    