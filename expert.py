import networkx as nx
from typing import List

from dpro.trace_utils import gen_long_name

from utils import gen_pid

per_token_comm_time = 1

class Expert:
    def __init__(self, expert_id, worker_id, dur_fw, dur_bw, moe_layer_id=0):
        '''
        expert_id: int
            The unique id for this expert
        worker_id: int
            The it of the worker where the expert are located
        dur_fw: float
            the duration to perform forward for this expert, in ms
        dur_bw: float
            The duration to perform backpropagation for this expert, in ms
        moe_layer_id: int:
            The unique id for the moe layer that contains this expert
        '''
        self.expert_id = expert_id
        self.worker_id = worker_id
        self.moe_layer_id = moe_layer_id
        self.dur_fw = dur_fw
        self.dur_bw = dur_bw

        self.pipeline_degree = None

        self.pid = gen_pid(0, self.worker_id)

    def add_prefix(self, name, _prefix=None):
        if _prefix is None:
            return gen_long_name(self.pid, name)
        else:
            return gen_long_name(_prefix, name)

    def gen_graph(self, token_ids: list, predecessor_worker_pid, phase="FW"):
        '''
        Pararmeters
        -----------
        token_ids: list
            The IDs of tokens that are routed to this expert
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

        virtual_start_op = gen_long_name(predecessor_worker_pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}.start")
        virtual_end_op = gen_long_name(predecessor_worker_pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}.end")

        node_attrs = {
            virtual_start_op: 0,
            virtual_end_op: 0
        }

        for token_id in token_ids:
            if predecessor_worker_pid == self.pid:
                ### Expert training
                comp_op = gen_long_name(self.pid, f"{phase}.MoE_L{self.moe_layer_id}E{self.expert_id}", f"{token_id}")
                edges_to_add.append((virtual_start_op, comp_op))
                edges_to_add.append((comp_op, virtual_end_op))
                node_attrs[comp_op] = self.dur_fw if phase == "FW" else self.dur_bw
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
                node_attrs[comp_op] = self.dur_fw if phase == "FW" else self.dur_bw
                node_attrs[post_comm_op_send] = per_token_comm_time
                node_attrs[post_comm_op_recv] = per_token_comm_time

        graph.add_edges_from(edges_to_add)
        nx.set_node_attributes(graph, node_attrs, name="avg")
        return graph, [virtual_start_op], [virtual_end_op]


def gen_full_graph(expert2worker: List[int], token2expert, worker_num):
    '''
    Parameters
    ----------
    expert2worker: List[int]
        A list where the e'th element denote the ID of the worker where this expert 
        is located
    token2expert: List[List[int]]
        The 2D list of shape |W| x |S|, token2expert[w][s] denotes the expert that 
        token s on worker w will be routed to
    '''

    EXPERT_NUM = len(expert2worker)

    experts = [Expert(expert_id, expert2worker[expert_id], dur_fw=1, dur_bw=2) 
            for expert_id in range(EXPERT_NUM)]
    
    assert len(token2expert) == worker_num

    expert2tokens = [[[] for _ in range(len(experts))] for _ in range(worker_num)]
    for worker_id in range(worker_num):
        for token_id, expert_id in enumerate(token2expert[worker_id]):
            expert2tokens[worker_id][expert_id].append(f"W{worker_id}S{token_id}")

    G = nx.DiGraph()

    edges_to_add = []
    node_attrs = {}
    for worker_id in range(worker_num):
        _pid = gen_pid(0, worker_id)
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

            ### Forward phase
            graph, source_nodes, end_nodes = expert.gen_graph(
                expert2tokens[worker_id][expert.expert_id], _pid, phase="FW")
            G = nx.union(G, graph)
            for source_node in source_nodes:
                edges_to_add.append((fw_op1, source_node))
            for end_node in end_nodes:
                edges_to_add.append((end_node, fw_op2))

            graph, source_nodes, end_nodes = expert.gen_graph(
                expert2tokens[worker_id][expert.expert_id], _pid, phase="BW")
            G = nx.union(G, graph)
            for source_node in source_nodes:
                edges_to_add.append((bw_op2, source_node))
            for end_node in end_nodes:
                edges_to_add.append((end_node, bw_op1))
    
    G.add_edges_from(edges_to_add)
    nx.set_node_attributes(G, node_attrs, name="avg")

    return G
    
def migrate_expert(expert2worker, source_worker, target_worker, expert):
    assert expert2worker[expert] == source_worker
    expert2worker[expert] = target_worker