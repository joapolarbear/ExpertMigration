import networkx as nx

from dpro.trace_utils import gen_long_name

per_token_comm_time = 1

class Expert:
    def __init__(self, expert_id, worker_id, dur_fw, dur_bw):
        self.expert_id = expert_id
        self.worker_id = worker_id
        self.dur_fw = dur_fw
        self.dur_bw = dur_bw

        self.pipeline_degree = None

        self.prefix = f"Worker{self.worker_id}"

    def add_prefix(self, name, _prefix=None):
        if _prefix is None:
            return gen_long_name(self.prefix, name)
        else:
            return gen_long_name(_prefix, name)

    def gen_graph(self, token_ids: list, phase="FW"):
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

        virtual_start_op = self.add_prefix(f"{phase}.Expert{self.expert_id}.start")
        virtual_end_op = self.add_prefix(f"{phase}.Expert{self.expert_id}.end")

        node_attrs = {
            virtual_start_op: 0,
            virtual_end_op: 0
        }

        for token_id in token_ids:
            ### Token pre-calc
            pre_comm_op = gen_long_name(self.prefix, f"Comm.Expert{self.expert_id}.RECV", f"pre-calc_{phase}_{token_id}")
            edges_to_add.append((virtual_start_op, pre_comm_op))

            ### Expert training
            comp_op = gen_long_name(self.prefix, f"{phase}.Expert{self.expert_id}", f"{token_id}")
            edges_to_add.append((pre_comm_op, comp_op))

            ### Token post-calc
            post_comm_op = gen_long_name(self.prefix, f"Comm.Expert{self.expert_id}.SEND", f"post-calc_{phase}_{token_id}")
            edges_to_add.append((comp_op, post_comm_op))
            edges_to_add.append((post_comm_op, virtual_end_op))

            ### Decide the avg of each nodes
            node_attrs[pre_comm_op] = per_token_comm_time
            node_attrs[comp_op] = self.dur_fw if phase == "FW" else self.dur_bw
            node_attrs[post_comm_op] = per_token_comm_time

        graph.add_edges_from(edges_to_add)
        nx.set_node_attributes(graph, node_attrs, name="avg")
        return graph, [virtual_start_op], [virtual_end_op]

