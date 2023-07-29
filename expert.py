import networkx as nx
from typing import List, Dict
import copy

from dpro.trace_utils import gen_long_name
from dpro.replay import Replayer
from dpro.dag_utils import cal_edge_cost, dag_longest_path

from utils import gen_pid
import core


class Expert:
    def __init__(self, expert_id, NECM, moe_layer_id=0):
        '''
        expert_id: int
            The unique id for this expert in this MoE layer
        moe_layer_id: int or str:
            The unique id for the moe layer that contains this expert

        ### Deperecated arguments
        worker_id: int
            The it of the worker where the expert are located
        dur_fw: float
            the duration to perform forward for this expert, in ms
        dur_bw: float
            The duration to perform backpropagation for this expert, in ms
        '''
        self.moe_layer_id = moe_layer_id
        self.expert_id = expert_id
        self.NECM = NECM

        self.pipeline_degree = 1

        self.non_expert_boundary = {
            "fw_in": set(),
            "fw_out": set(),
            "bw_in": set(),
            "bw_out": set()
        }
    
    def set_comp_time_arg(self, fw_dur, bw_to_fw_ratio):
        N, E, C, M = self.NECM
        self.k_expert = fw_dur / (C * M * E)
        self.bw_to_fw_ratio = bw_to_fw_ratio
    
    def get_comp_time_for_one_expert(self, token_num, phase):
        N, E, C, M = self.NECM
        if phase == "FW":
            return self.k_expert * token_num * M
        elif phase == "BW":
            return self.k_expert * token_num * M * self.bw_to_fw_ratio
        else:
            raise
    
    def set_comm_time_arg(self, t_all_to_all, bandwidth):
        N, E, C, M = self.NECM
        self.k_comm = (t_all_to_all * bandwidth * N) / (E * C * M * (N - 1))
    
    def get_p2p_comm_time(self, token_num, source_worker, target_worker):
        N, E, C, M = self.NECM
        bandwidth = core.get_bandwidth(source_worker, target_worker)
        return self.k_comm * token_num * M / bandwidth  

    def assign_worker(self, worker_id):
        ### FW and BW should share the same worker assignment
        self.worker_id = worker_id
        self.pid = gen_pid(0, self.worker_id)

    def add_prefix(self, name, _prefix=None):
        if _prefix is None:
            return gen_long_name(self.pid, name)
        else:
            return gen_long_name(_prefix, name)

    def expert_graph_for_one_worker(self, token_ids: list, predecessor_worker_id, phase="FW"):
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
        edges_to_add = []
        graph = nx.DiGraph()

        predecessor_worker_pid = worker_id_to_pid(predecessor_worker_id)
        virtual_start_op = gen_long_name(predecessor_worker_pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}.start")
        virtual_end_op = gen_long_name(predecessor_worker_pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}.end")

        node_attrs = {
            virtual_start_op: 0,
            virtual_end_op: 0
        }

        ### decide the execution time
        expert_comp_time = self.get_comp_time_for_one_expert(self.pipeline_degree, phase)
        per_token_comm_time = self.get_p2p_comm_time(self.pipeline_degree, predecessor_worker_id, self.worker_id)

        for token_id in token_ids:
            if predecessor_worker_pid == self.pid:
                ### Expert training
                comp_op = gen_long_name(self.pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}", f"{token_id}")
                edges_to_add.append((virtual_start_op, comp_op))
                edges_to_add.append((comp_op, virtual_end_op))
                node_attrs[comp_op] = expert_comp_time
            else:
        
                ### Token pre-calc
                pre_comm_op_send = gen_long_name(predecessor_worker_pid, f"Comm.MoE_L{self.moe_layer_id}E{self.expert_id}.SEND", f"pre-calc_{phase}_{token_id}")
                pre_comm_op_recv = gen_long_name(self.pid, f"Comm.MoE_L{self.moe_layer_id}E{self.expert_id}.RECV", f"pre-calc_{phase}_{token_id}")
                edges_to_add.append((virtual_start_op, pre_comm_op_send))
                edges_to_add.append((pre_comm_op_send, pre_comm_op_recv))

                ### Expert training
                comp_op = gen_long_name(self.pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}", f"{token_id}")
                edges_to_add.append((pre_comm_op_recv, comp_op))

                ### Token post-calc
                post_comm_op_send = gen_long_name(self.pid, f"Comm.MoE_L{self.moe_layer_id}E{self.expert_id}.SEND", f"post-calc_{phase}_{token_id}")
                post_comm_op_recv = gen_long_name(predecessor_worker_pid, f"Comm.MoE_L{self.moe_layer_id}E{self.expert_id}.RECV", f"post-calc_{phase}_{token_id}")
                edges_to_add.append((comp_op, post_comm_op_send))
                edges_to_add.append((post_comm_op_send, post_comm_op_recv))
                edges_to_add.append((post_comm_op_recv, virtual_end_op))

                ### Decide the avg of each nodes
                node_attrs[pre_comm_op_send] = per_token_comm_time
                node_attrs[pre_comm_op_recv] = per_token_comm_time
                node_attrs[comp_op] = expert_comp_time
                node_attrs[post_comm_op_send] = per_token_comm_time
                node_attrs[post_comm_op_recv] = per_token_comm_time


        ### Connect the non-expert graph to the expert sub graph
        start_nodes = [virtual_start_op]
        end_nodes = [virtual_end_op]
        boundary_in_op = gen_long_name(predecessor_worker_pid, self.non_expert_boundary["fw_in"] 
                                       if phase == "FW" else self.non_expert_boundary["bw_in"])
        boundary_out_op = gen_long_name(predecessor_worker_pid, self.non_expert_boundary["fw_out"] 
                                       if phase == "FW" else self.non_expert_boundary["bw_out"])
        for source_node in start_nodes:
            edges_to_add.append((boundary_in_op, source_node))
        for end_node in end_nodes:
            edges_to_add.append((end_node, boundary_out_op))
            
        graph.add_edges_from(edges_to_add)
        nx.set_node_attributes(graph, node_attrs, name="avg")
        return graph


def worker_id_to_pid(worker_id):
    return gen_pid(0, worker_id)


class MoEAssign:
    ''' The deployment decision for one MoE layer
    '''
    def __init__(self, expert2worker: List[int], worker2token2expert: List[List[int]]):
        '''
        Parameters
        ----------
        expert2worker: shape = [N_expert], each value denots the worker id 
        worker2token2expert: shape = [N_worker, N_token], each value denotes the expert id in the MoE layer
        Example:
        expert2worker = [0, 1, 0]
        worker2token2expert = [
            [0, 1, 0, 1, 2, 2, 0, 1],
            [0, 1, 0, 1, 2, 2, 0, 1],
        ]
        '''
        self.expert2worker = expert2worker
        self.worker2token2expert = worker2token2expert


class DynamicGraph:
    def __init__(self, ep_world_size):
        self.ep_world_size = ep_world_size

        self.graph_non_expert = nx.DiGraph()
        self.node2avg = {}
        self.moe_layers = {}

        self.edges_to_add_buffer = []
        self.one_time_add_edge_hook = None
    
    def add_comp_edge_for_all_worker(self, source, target):
        ### Used for homogeneous systems
        for worker_id in range(self.ep_world_size):
            _pid = worker_id_to_pid(worker_id)
            op1 = gen_long_name(_pid, source)
            op2 = gen_long_name(_pid, target)
            self.edges_to_add_buffer.append((op1, op2))

        if self.one_time_add_edge_hook:
            self.one_time_add_edge_hook(source)
            self.one_time_add_edge_hook = None
        
    def set_avg_for_all_worker(self, rawname, avg):
        for worker_id in range(self.ep_world_size):
            _pid = worker_id_to_pid(worker_id)
            op = gen_long_name(_pid, rawname)
            self.node2avg[op] = avg
    
    def met_expert_for_all_worker(self, experts: List[Expert], phase):
        ### Step 1, finalize current cached edges
        self.graph_non_expert.add_edges_from(self.edges_to_add_buffer)
        non_expert_in = self.edges_to_add_buffer[-1][1]
        if phase == "FW":
            for expert in experts:
                expert.non_expert_boundary["fw_in"].add(non_expert_in)
        elif phase == "BW":
            for expert in experts:
                expert.non_expert_boundary["bw_in"].add(non_expert_in)
        else:
            raise
        
        def _hook(non_expert_out):
            if phase == "FW":
                for expert in experts:
                    print(f"One time hook for register {phase} {expert.moe_layer_id}, add bound {non_expert_out}")
                    expert.non_expert_boundary["fw_out"].add(non_expert_out)
            elif phase == "BW":
                for expert in experts:
                    print(f"One time hook for register {phase} {expert.moe_layer_id}, add bound {non_expert_out}")
                    expert.non_expert_boundary["bw_out"].add(non_expert_out)
            else:
                raise
        self.one_time_add_edge_hook = _hook

        self.moe_layers[experts[0].moe_layer_id] = experts
        self.edges_to_add_buffer = []

    def finalize_graph(self, moe_assignments: Dict[str, MoEAssign]) -> nx.DiGraph:

        G = copy.deepcopy(self.graph_non_expert)
        nx.set_node_attributes(G, self.node2avg, name="avg")

        for moe_layer_id, moe_assign in moe_assignments.items():
            EXPERT_NUM = len(moe_assign.expert2worker)
            experts = self.moe_layers[moe_layer_id]

            N, E, C, M = experts[0].NECM
            
            assert len(moe_assign.worker2token2expert) == N # worker_num
            assert E == len(experts) and EXPERT_NUM == E

            ### Map experts to workers
            for ep_id, expert in enumerate(experts):
                worker_id = moe_assign.expert2worker[ep_id]
                expert.assign_worker(worker_id)

            worker2expert2tokens = [[[] for _ in range(len(experts))] for _ in range(N)]
            for worker_id in range(N):
                for token_id, expert_id in enumerate(moe_assign.worker2token2expert[worker_id]):
                    worker2expert2tokens[worker_id][expert_id].append(f"W{worker_id}S{token_id}")
            
            for worker_id in range(N):
                for expert in experts:
                    graph = expert.expert_graph_for_one_worker(
                        worker2expert2tokens[worker_id][expert.expert_id], worker_id, phase="FW")
                    G = nx.union(G, graph)

                    graph = expert.expert_graph_for_one_worker(
                        worker2expert2tokens[worker_id][expert.expert_id], worker_id, phase="BW")
                    G = nx.union(G, graph)
        
        return G
            

    @staticmethod
    def replay(G):
        replayer = Replayer(dag=G, _step_num=1, leaf_dirs=None, 
                        dump_path=".workspace/",
                        comm_backend="default",
                        byteps_graph=None)

        # import pdb; pdb.set_trace()
        replayer.replay(verbose=True)
        cal_edge_cost(replayer.exct_dag)
        critical_path = dag_longest_path(replayer.exct_dag, None, 
                            weight="weight", default_weight=0, _debug_level=1)


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
        _pid = worker_id_to_pid(worker_id)
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
            
            expert.non_expert_boundary["fw_in"].add("FW.1")
            expert.non_expert_boundary["fw_out"].add("FW.2")

            expert.non_expert_boundary["bw_in"].add("BW.2")
            expert.non_expert_boundary["bw_out"].add("BW.1")

            ### Forward phase
            graph = expert.expert_graph_for_one_worker(
                worker2expert2tokens[worker_id][expert.expert_id], worker_id, phase="FW")
            G = nx.union(G, graph)

            graph = expert.expert_graph_for_one_worker(
                worker2expert2tokens[worker_id][expert.expert_id], worker_id, phase="BW")
            G = nx.union(G, graph)
    
    G.add_edges_from(edges_to_add)
    nx.set_node_attributes(G, node_attrs, name="avg")

    return G
    
def migrate_expert(expert2worker, source_worker, target_worker, expert):
    assert expert2worker[expert] == source_worker
    expert2worker[expert] = target_worker