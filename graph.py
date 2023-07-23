import networkx as nx
from typing import List

from dpro.trace_utils import gen_long_name

from expert import Expert

def gen_full_graph(experts: List[Expert]):
    token2expert = [0, 1, 0, 1, 2, 2, 0, 1]

    expert2tokens = [[] for _ in range(len(experts))]
    for token_id, expert_id in enumerate(token2expert):
        expert2tokens[expert_id].append(token_id)

    G = nx.DiGraph()

    edges_to_add = []
    default_worker_id = 0

    default_pid = f"Worker{default_worker_id}"
    fw_op1 = gen_long_name(default_pid, "FW.1")
    fw_op2 = gen_long_name(default_pid, "FW.2")

    bw_op1 = gen_long_name(default_pid, "BW.1")
    bw_op2 = gen_long_name(default_pid, "BW.2")

    node_attrs = {
        fw_op1: 3,
        fw_op2: 5,
        bw_op1: 6,
        bw_op2: 10
    }

    ### Connect the main part to the expert part
    for expert in experts:

        ### Forward phase
        graph, source_nodes, end_nodes = expert.gen_graph(
            expert2tokens[expert.expert_id], phase="FW")
        G = nx.union(G, graph)
        for source_node in source_nodes:
            edges_to_add.append((fw_op1, source_node))
        for end_node in end_nodes:
            edges_to_add.append((end_node, fw_op2))

        graph, source_nodes, end_nodes = expert.gen_graph(
            expert2tokens[expert.expert_id], phase="BW")
        G = nx.union(G, graph)
        for source_node in source_nodes:
            edges_to_add.append((bw_op2, source_node))
        for end_node in end_nodes:
            edges_to_add.append((end_node, bw_op1))
    
    G.add_edges_from(edges_to_add)
    nx.set_node_attributes(G, node_attrs, name="avg")

    return G
    
