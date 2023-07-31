''' Physical settings
'''
from utils import gen_pid

# intra-machine bandwidth for A100 GPUs, in GB/s
INTRA_MACHINE_BW = 25
# inter-machine bandwidth, in GB/s
INTER_MACHINE_BW = 1000 / 8 / 1024

# set according to the profiling result, # of token / s
COMP_THROUGHPUT = None

HOST_NUM = None
LOCAL_RANK_NUM = None

def get_bandwidth(source_worker, target_worker):
    source_host = source_worker // LOCAL_RANK_NUM
    target_host = target_worker // LOCAL_RANK_NUM
    if source_host != target_host:
        return INTER_MACHINE_BW
    else:
        return INTRA_MACHINE_BW

def worker_id_to_pid(worker_id):
    host_id = worker_id // LOCAL_RANK_NUM
    return gen_pid(host_id, worker_id)
