''' Physical settings
'''


# intra-machine bandwidth for A100 GPUs, in GB/s
INTRA_MACHINE_BW = 25

def get_bandwidth(source_worker, target_worker):
    return INTRA_MACHINE_BW