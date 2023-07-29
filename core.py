''' Physical settings
'''
from utils import gen_pid

# intra-machine bandwidth for A100 GPUs, in GB/s
INTRA_MACHINE_BW = 25

def get_bandwidth(source_worker, target_worker):
    return INTRA_MACHINE_BW

def worker_id_to_pid(worker_id):
    return gen_pid(0, worker_id)
