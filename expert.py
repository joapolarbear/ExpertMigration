import networkx as nx
from typing import List, Dict
import copy
import numpy as np

from dpro.trace_utils import gen_long_name, parse_rawname
from dpro.replay import Replayer
from dpro.dag_utils import cal_edge_cost, dag_longest_path

import core

END_NODE = "UPDATE_.END"

class Expert:
    def __init__(self, expert_id, moe_layer):
        '''
        expert_id: int
            The unique id for this expert in this MoE layer

        ### Deperecated arguments
        worker_id: int
            The it of the worker where the expert are located
        dur_fw: float
            the duration to perform forward for this expert, in ms
        dur_bw: float
            The duration to perform backpropagation for this expert, in ms
        '''
        self.expert_id = expert_id
        self.moe_layer = moe_layer
        self.moe_layer_id = moe_layer.moe_layer_id
        self.NECM = moe_layer.NECM
        
        self.pipeline_degree = 1

        self.worker_id = None
        ### Used for shadow policy
        self.is_shadow = False
    
    def reset(self):
        self.worker_id = None
        self.is_shadow = False

    def assign_worker(self, worker_id):
        ### FW and BW should share the same worker assignment
        self.worker_id = worker_id
        self.pid = core.worker_id_to_pid(worker_id)

    def add_prefix(self, name, _prefix=None):
        if _prefix is None:
            return gen_long_name(self.pid, name)
        else:
            return gen_long_name(_prefix, name)

    def expert_graph_for_one_worker(self, token_ids: list, 
            predecessor_worker_id, phase="FW", output_graph=None, virtual=False):
        ''' Build the corresponding expert graph corresponds to tokens transmitted from 
        `predecessor_worker_id` to thie expert

        Pararmeters
        -----------
        token_ids: list
            The IDs of tokens that are routed to this expert from worker `predecessor_worker_pid`
        phase: str:
            "FW" denotes the forward pass, "BW" denotes the back-propagation pass

        Return
        ------
        graph: nx.Digraph
            The subgraph corresponding to the pre-calc, expert training and post-calc phases
        source_nodes: list[str]
            The starting points for any path that goes through this graph
        end_nodes: list[str]
            The end points for any path that goes through this graph
        '''
        assert self.worker_id is not None
        
        predecessor_worker_pid = core.worker_id_to_pid(predecessor_worker_id)
        edges_to_add = []
        node_attrs = {}

        expert_boundary_in = set()
        expert_boundary_out = set()

        ### decide the execution time
        if virtual:
            expert_comp_time = per_token_comm_time = 0
        else:
            expert_comp_time = self.moe_layer.get_comp_time_for_one_expert(self.pipeline_degree, phase)
            per_token_comm_time = self.moe_layer.get_p2p_comm_time(self.pipeline_degree, predecessor_worker_id, self.worker_id)

        for token_id in token_ids:
            if predecessor_worker_pid == self.pid:
                ### Expert training
                comp_op = gen_long_name(self.pid, f"{phase}.{self.moe_layer_id}.E{self.expert_id}", f"{token_id}")
                expert_boundary_in.add(comp_op)
                expert_boundary_out.add(comp_op)
                node_attrs[comp_op] = expert_comp_time
            elif self.is_shadow:
                comp_op = gen_long_name(predecessor_worker_pid, f"{phase}.{self.moe_layer_id}.E{self.expert_id}_shadow", f"{token_id}")
                expert_boundary_in.add(comp_op)
                expert_boundary_out.add(comp_op)
                node_attrs[comp_op] = expert_comp_time
            else:
                ### Token pre-calc
                pre_comm_op_send = gen_long_name(predecessor_worker_pid, f"Comm.{self.moe_layer_id}.SEND", f"pre-calc_{phase}_E{self.expert_id}_{token_id}")
                pre_comm_op_recv = gen_long_name(self.pid, f"Comm.{self.moe_layer_id}.RECV", f"pre-calc_{phase}_E{self.expert_id}_{token_id}")
                expert_boundary_in.add(pre_comm_op_send)
                edges_to_add.append((pre_comm_op_send, pre_comm_op_recv))

                ### Expert training
                comp_op = gen_long_name(self.pid, f"{phase}.{self.moe_layer_id}.E{self.expert_id}", f"{token_id}")
                edges_to_add.append((pre_comm_op_recv, comp_op))

                ### Token post-calc
                post_comm_op_send = gen_long_name(self.pid, f"Comm.{self.moe_layer_id}.SEND", f"post-calc_{phase}_E{self.expert_id}_{token_id}")
                post_comm_op_recv = gen_long_name(predecessor_worker_pid, f"Comm.{self.moe_layer_id}.RECV", f"post-calc_{phase}_E{self.expert_id}_{token_id}")
                edges_to_add.append((comp_op, post_comm_op_send))
                edges_to_add.append((post_comm_op_send, post_comm_op_recv))
                expert_boundary_out.add(post_comm_op_recv)

                ### Decide the avg of each nodes
                node_attrs[pre_comm_op_send] = per_token_comm_time
                node_attrs[pre_comm_op_recv] = per_token_comm_time
                node_attrs[comp_op] = expert_comp_time
                node_attrs[post_comm_op_send] = per_token_comm_time
                node_attrs[post_comm_op_recv] = per_token_comm_time

        ### Connect the non-expert graph to the expert sub graph
        boundary_in_op = gen_long_name(predecessor_worker_pid, self.moe_layer.non_expert_boundary["fw_in"] 
                                       if phase == "FW" else self.moe_layer.non_expert_boundary["bw_in"])
        boundary_out_op = gen_long_name(predecessor_worker_pid, self.moe_layer.non_expert_boundary["fw_out"] 
                                       if phase == "FW" else self.moe_layer.non_expert_boundary["bw_out"])
        for source_node in expert_boundary_in:
            edges_to_add.append((boundary_in_op, source_node))
        for end_node in expert_boundary_out:
            edges_to_add.append((end_node, boundary_out_op))
        
        graph = output_graph or nx.DiGraph()
        graph.add_edges_from(edges_to_add)
        nx.set_node_attributes(graph, node_attrs, name="avg")
        return graph

class MoELayer:
    def __init__(self, moe_layer_id, NECM):
        '''
        moe_layer_id: int or str:
            The unique id for the moe layer that contains this expert
        '''
        self.moe_layer_id = moe_layer_id
        self.NECM = NECM
        self.experts = None
    
        self.non_expert_boundary = {
            "fw_in": None,
            "fw_out": None,
            "bw_in": None,
            "bw_out": None
        }
        
    ### Set boundaries
    def register_boudary_in(self, phase, non_expert_in):
        if phase == "FW":
            self.non_expert_boundary["fw_in"] = non_expert_in
        elif phase == "BW":
            self.non_expert_boundary["bw_in"] = non_expert_in
        else:
            raise
    
    def register_boundary_out(self, phase, non_expert_out):
        if phase == "FW":
            print(f" - One time hook adds dependency {phase}.{self.moe_layer_id} --> {non_expert_out}")
            self.non_expert_boundary["fw_out"] = non_expert_out
        elif phase == "BW":
            print(f" - One time hook adds dependency {phase}.{self.moe_layer_id} --> {non_expert_out}")
            self.non_expert_boundary["bw_out"] = non_expert_out
        else:
            raise

    ### Time related
    def set_comp_time_arg(self, fw_dur, bw_to_fw_ratio):
        assert self.experts is None
        N, E, C, M = self.NECM
        self.k_expert = fw_dur / (C * M * E)
        self.bw_to_fw_ratio = bw_to_fw_ratio

        one_token_fw_time = self.get_comp_time_for_one_expert(1, "FW") / 1e3
        core.COMP_THROUGHPUT = 1 / one_token_fw_time

    def set_comm_time_arg(self, t_all_to_all, bandwidth):
        assert self.experts is None
        N, E, C, M = self.NECM
        self.k_comm = (t_all_to_all * bandwidth * N) / (E * C * M * (N - 1))
    
    def get_comp_time_for_one_expert(self, token_num, phase):
        N, E, C, M = self.NECM
        if phase == "FW":
            return self.k_expert * token_num * M
        elif phase == "BW":
            return self.k_expert * token_num * M * self.bw_to_fw_ratio
        else:
            raise
    
    def get_p2p_comm_time(self, token_num, source_worker, target_worker):
        N, E, C, M = self.NECM
        bandwidth = core.get_bandwidth(source_worker, target_worker)
        return self.k_comm * token_num * M / bandwidth 
        # return 1e-3 * token_num * M / bandwidth
    
    def set_comm_comp_time(self, op_stat, events, bw_to_fw_ratio):
        assert self.experts is None
        fw_dur = events[op_stat[self.moe_layer_id]["fw_event_id"]]["dur"] / 1e3
        self.set_comp_time_arg(fw_dur, bw_to_fw_ratio)

        t_all_to_all = events[op_stat[self.moe_layer_id]["pre-calc_FW_event_id"]]["dur"] / 1e3
        self.set_comm_time_arg(t_all_to_all, core.INTRA_MACHINE_BW)

    ### Init
    def update_setting(self, worker_num, ep_num):
        assert self.experts is None
        self.NECM[0] = worker_num
        self.NECM[1] = ep_num

    def init(self):
        N, E, C, M = self.NECM
        self.experts = [Expert(expert_id, self) for expert_id in range(E)]
    
    def reset(self):
        self.experts = None
        self.coarse_ep_time = None

    def apply_moe_solution(self, moe_solution):
        assert self.experts is not None
        if moe_solution.allocation_per_timeslot is not None and \
            moe_solution.time_slot_span is not None:
            # rst.z[k, 3, i, j, t]
            z = moe_solution.allocation_per_timeslot[:, [1, 3, 4], :, :, :]
            non_zero_inds = z.nonzero() # [inds_k_array, inds_a_array, inds_i_array, ... ]
            last_time_slot = max(non_zero_inds[-1])
            # Select inds with largest inds_t
            rst = [l[np.argwhere(non_zero_inds[-1] == np.amax(non_zero_inds[-1])).flatten()].tolist() for l in non_zero_inds]
            residual = 0
            # import pdb; pdb.set_trace()
            for _k, _a, _i, _j, _t in zip(*rst):
                if _a == 1:
                    continue
                _bandwidth = core.get_bandwidth(_i, _j)
                residual = max(residual, 1e3 * z[_k, _a, _i, _j, _t] / _bandwidth)
            self.coarse_ep_time = last_time_slot * moe_solution.time_slot_span + residual
            return
        elif moe_solution.ideal:
            self.coarse_ep_time = -1
            return
    
        ### Map experts to workers and shadow decision
        for ep_id, expert in enumerate(self.experts):
                worker_id = moe_solution.expert2worker[ep_id]
                expert.assign_worker(worker_id)
                if moe_solution.expert_shadow is not None:
                    expert.is_shadow = moe_solution.expert_shadow[ep_id]
    
    def apply_token_distribution(self, worker2token2expert, graph, moe_layer_only):
        assert self.experts is not None
        N, E, C, M = self.NECM
        
        if self.coarse_ep_time is not None:
            if self.coarse_ep_time < 0:
                self.coarse_ep_time = 1e3 * len(worker2token2expert[0]) / core.COMP_THROUGHPUT
            # print(len(worker2token2expert[0]))
            print(f"Use coarse grained expert time {self.coarse_ep_time}")
            for worker_id in range(N):
                _pid = core.worker_id_to_pid(worker_id)
                
                edges_to_add = []
                node_attrs = {}

                def _add_edge_f(phase):
                    boundary_in_op = gen_long_name(_pid, self.non_expert_boundary["fw_in"] 
                                        if phase == "FW" else self.non_expert_boundary["bw_in"])
                    boundary_out_op = gen_long_name(_pid, self.non_expert_boundary["fw_out"] 
                                                if phase == "FW" else self.non_expert_boundary["bw_out"])
                    
                    coase_op = gen_long_name(_pid, f"{phase}.{self.moe_layer_id}.Coarse")
                    edges_to_add.append((boundary_in_op, coase_op))
                    edges_to_add.append((coase_op, boundary_out_op))
                    if phase == "FW":
                        _ratio = 1
                    elif moe_layer_only:
                        _ratio = 0
                    else:
                        _ratio = self.bw_to_fw_ratio
                    node_attrs[coase_op] = self.coarse_ep_time * _ratio
                
                _add_edge_f("FW")
                _add_edge_f("BW")
                graph.add_edges_from(edges_to_add)
                nx.set_node_attributes(graph, node_attrs, name="avg")
                return 

        worker2expert2tokens = [[[] for _ in range(len(self.experts))] for _ in range(N)]
        for worker_id in range(N):
            for token_id, expert_id in enumerate(worker2token2expert[worker_id]):
                worker2expert2tokens[worker_id][expert_id].append(f"W{worker_id}S{token_id}")
        
        for worker_id in range(N):
            for expert in self.experts:
                expert.expert_graph_for_one_worker(worker2expert2tokens[worker_id][expert.expert_id], 
                    worker_id, phase="FW", output_graph=graph)

                expert.expert_graph_for_one_worker(worker2expert2tokens[worker_id][expert.expert_id],
                    worker_id, phase="BW", output_graph=graph, virtual=moe_layer_only)


class MoESolution:
    ''' The deployment solution for one MoE layer
    '''
    def __init__(self, expert2worker: List[int],
                 expert_shadow: List[bool] = None,
                 allocation_per_timeslot = None,
                 time_slot_span = None,
                 ideal = False):
        '''
        Parameters
        ----------
        expert2worker: shape = [N_expert], each value denots the worker id 
        expert_shadow: shape = [N_expert], expert_shadow[k] = True if the k-th expert is shadowed
        Example:
        expert2worker = [0, 1, 0]
        worker2token2expert = [
            [0, 1, 0, 1, 2, 2, 0, 1],
            [0, 1, 0, 1, 2, 2, 0, 1],
        ]
        '''
        self.expert2worker = expert2worker
        self.expert_shadow = expert_shadow
        self.allocation_per_timeslot = allocation_per_timeslot
        self.time_slot_span = time_slot_span # in ms
        self.ideal = ideal


class DynamicGraph:
    def __init__(self, ep_world_size):
        self.ep_world_size = ep_world_size

        self.graph_non_expert = nx.DiGraph()
        self.rawname2avg = {}
        self.moe_layers = {}

        self.edges_to_add = []
        self.edges_to_add_buffer = []
        self.one_time_add_edge_hook = None
    
    def add_comp_edge_for_all_worker(self, source, target):
        print(f" - Add edge {source} --> {target}")
        ### Used for homogeneous systems
        self.edges_to_add_buffer.append((source, target))

        if "UPDATE_" in target:
            self.edges_to_add_buffer.append((target, END_NODE))

        if self.one_time_add_edge_hook is not None:
            self.one_time_add_edge_hook(source)
            self.one_time_add_edge_hook = None
        
    def set_avg_for_all_worker(self, rawname, avg):
        self.rawname2avg[rawname] = avg
            
    def met_expert_for_all_worker(self, moe_layer: MoELayer, phase):
        ### Step 1, finalize current cached edges
        # self.graph_non_expert.add_edges_from(self.edges_to_add_buffer)
        # non_expert_in = parse_rawname(self.edges_to_add_buffer[-1][1])

        self.edges_to_add += self.edges_to_add_buffer
        non_expert_in = self.edges_to_add_buffer[-1][1]
        moe_layer.register_boudary_in(phase, non_expert_in)

        def _hook(non_expert_out):
            moe_layer.register_boundary_out(phase, non_expert_out)
        self.one_time_add_edge_hook = _hook

        self.moe_layers[moe_layer.moe_layer_id] = moe_layer
        self.edges_to_add_buffer = []

    def reset(self):
        for moe_layer in self.moe_layers.values():
            moe_layer.reset()

    def finalize_graph(self, all_moe_solution: Dict[str, MoESolution],
                    all_worker2token2expert: Dict, 
                    worker_num=None,
                    moe_layer_only=True) -> nx.DiGraph:
        '''
        all_worker2token2expert: a dict, where the key is the moe layer id, 
            and the value is worker2token2expert of shape = [N_worker, N_token], 
            whose each value denotes the expert id in the MoE layer
        '''
        if len(self.edges_to_add_buffer) > 0:
            # self.graph_non_expert.add_edges_from(self.edges_to_add_buffer)
            self.edges_to_add += self.edges_to_add_buffer
            self.edges_to_add_buffer = []

        _worker_num = worker_num or self.ep_world_size
        G = nx.DiGraph()

        _edges_to_add = []
        for worker_id in range(_worker_num):
            _pid = core.worker_id_to_pid(worker_id)
            for source, target in self.edges_to_add:
                op1 = gen_long_name(_pid, source)
                op2 = gen_long_name(_pid, target) if target != END_NODE else target
                _edges_to_add.append((op1, op2))
        G.add_edges_from(_edges_to_add)

        _node2avg = {END_NODE: 0}
        for worker_id in range(_worker_num):
            _pid = core.worker_id_to_pid(worker_id)
            for rawname, _avg in self.rawname2avg.items():
                op_name = gen_long_name(_pid, rawname) 
                _node2avg[op_name] = _avg if not moe_layer_only else 0
        nx.set_node_attributes(G, _node2avg, name="avg")
    
        for moe_layer_id, moe_solution in all_moe_solution.items():
            required_expert_num = len(moe_solution.expert2worker)

            moe_layer = self.moe_layers[moe_layer_id]
            moe_layer.update_setting(_worker_num, required_expert_num)
            
            worker2token2expert = all_worker2token2expert[moe_layer_id]
            assert len(worker2token2expert) == _worker_num, (len(worker2token2expert), _worker_num) # worker_num

            moe_layer.init()

            ### Apply solution
            moe_layer.apply_moe_solution(moe_solution)

            ### Apply token distribution
            moe_layer.apply_token_distribution(worker2token2expert, G, moe_layer_only)

        return G

    @staticmethod
    def replay(G, workspace):
        replayer = Replayer(dag=G, _step_num=1, leaf_dirs=None, 
                        dump_path=workspace,
                        comm_backend="default",
                        byteps_graph=None)

        replayer.replay(verbose=False)
        cal_edge_cost(replayer.exct_dag)
        critical_path_with_len = dag_longest_path(replayer.exct_dag, None, 
                            weight="cost", default_weight=0, _debug_level=0)
        crt_path, len_list = zip(*critical_path_with_len)
        length = sum(len_list)
        return length


def gen_full_graph(expert2worker: List[int], worker2token2expert, worker_num):
    '''
    Parameters
    ----------
    expert2worker: List[int]
        A list where the e'th element denote the ID of the worker where this expert 
        is located
    worker2token2expert: List[List[int]]
        The 2D list of shape |W| x |S|, worker2token2expert[w][s] denotes the expert that 
        token s on worker w will be routed to
    '''

    EXPERT_NUM = len(expert2worker)

    experts = [Expert(expert_id, expert2worker[expert_id], dur_fw=1, dur_bw=2) 
            for expert_id in range(EXPERT_NUM)]
    
    assert len(worker2token2expert) == worker_num

    worker2expert2tokens = [[[] for _ in range(len(experts))] for _ in range(worker_num)]
    for worker_id in range(worker_num):
        for token_id, expert_id in enumerate(worker2token2expert[worker_id]):
            worker2expert2tokens[worker_id][expert_id].append(f"W{worker_id}S{token_id}")

    G = nx.DiGraph()

    edges_to_add = []
    node_attrs = {}
    for worker_id in range(worker_num):
        _pid = core.worker_id_to_pid(worker_id)
        fw_op1 = gen_long_name(_pid, "FW.1")
        fw_op2 = gen_long_name(_pid, "FW.2")

        bw_op1 = gen_long_name(_pid, "BW.1")
        bw_op2 = gen_long_name(_pid, "BW.2")

        edges_to_add.append((fw_op2, bw_op2))

        node_attrs.update({
            fw_op1: 3,
            fw_op2: 5,
            bw_op1: 6,
            bw_op2: 10
        })

        ### Connect the main part to the expert part
        for expert in experts:
            
            expert.non_expert_boundary["fw_in"] = "FW.1"
            expert.non_expert_boundary["fw_out"] = "FW.2"

            expert.non_expert_boundary["bw_in"] = "BW.2"
            expert.non_expert_boundary["bw_out"] = "BW.1"

            ### Forward phase
            expert.expert_graph_for_one_worker(worker2expert2tokens[worker_id][expert.expert_id],
                worker_id, phase="FW", output_graph=G)

            expert.expert_graph_for_one_worker(worker2expert2tokens[worker_id][expert.expert_id],
                worker_id, phase="BW", output_graph=G)
    
    G.add_edges_from(edges_to_add)
    nx.set_node_attributes(G, node_attrs, name="avg")

    return G
    