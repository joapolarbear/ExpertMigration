
import dpro
from dpro.replay import Replayer
from dpro.dag_utils import cal_edge_cost, dag_longest_path

from expert import Expert
from graph import gen_full_graph

dpro.init(".workspace/", "test")

WORKER_NUM = 2
EXPERT_NUM = 3
expert2worker = [0, 1, 0]

experts = [Expert(expert_id, expert2worker[expert_id], dur_fw=1, dur_bw=2) 
           for expert_id in range(EXPERT_NUM)]

G = gen_full_graph(experts)

replayer = Replayer(dag=G, _step_num=1, leaf_dirs=None, 
                dump_path=".workspace/",
                comm_backend="default",
                byteps_graph=None)

# import pdb; pdb.set_trace()
replayer.replay(verbose=True)
cal_edge_cost(replayer.exct_dag)
critical_path = dag_longest_path(replayer.exct_dag, None, 
                    weight="weight", default_weight=0, _debug_level=1)




