import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt

def reduce_tick_num(num, fontsize, axis="y", low=1, high=1, type=None):
    if axis == "y":
        locs, labels = plt.yticks()
    elif axis == "x":
        locs, labels = plt.xticks()
    else:
        raise ValueError(axis)
    _min = min(locs)
    _max = max(locs)
    _mid = (_min + _max) / 2
    _range = (_max - _min)
    low *= (_mid - 1.1 * _range / 2)
    high *= _max
    new_locs = np.arange(low, high, step=(high-low)/float(num), dtype=type)
    # new_ticks = (new_locs / 1e4).astype(int)
    if axis == "y":
        plt.yticks(new_locs, fontsize=fontsize)
    else:
        plt.xticks(new_locs, fontsize=fontsize)
  

workspace = ".workspace/traces/d_model_1024"
fig_dir = os.path.join(workspace, "fig")
os.makedirs(fig_dir, exist_ok=True)

rst_path = os.path.join(workspace, "rst.json")
with open(rst_path, 'r') as fp:
    rst = json.load(fp)

rst_dict = {}
for rst_key in rst.keys():
    distribution_id, host_num, local_rank_num, local_ep_num = list(map(int, rst_key.split(",")))
    if distribution_id not in rst_dict:
        rst_dict[distribution_id] = {}
    if (host_num, local_rank_num) not in rst_dict[distribution_id]:
        rst_dict[distribution_id][(host_num, local_rank_num)] = []
    rst_dict[distribution_id][(host_num, local_rank_num)].append(
        (local_ep_num, np.array(rst[rst_key])))

fontsize = 36
barwidth = 0.2
marks = ["/", "-", "\\", "x", "+", "."]
METHODS = ["OEM", "FastMoE", "FasterMoE", "Random"]

def plot_legend(ax, ncol, path):     
    label_params = ax.get_legend_handles_labels()
    figl, axl = plt.subplots(figsize=(25, 2))
    axl.axis(False)

    axl.legend(*label_params,
        ncol=ncol, 
        loc="center", 
        bbox_to_anchor=(0.5, 0.5), 
        frameon=False,
        fontsize=fontsize,
        # prop={"size":50}
        )
    figl.savefig(path)

legend_done = False
for distribution_id in [
        # 0, 
        1, 
        2, 
        3
    ]:
    _rst_dict = rst_dict[distribution_id]
    iter_speedup = []
    for (host_num, local_rank_num), _data in _rst_dict.items():
        local_ep_num_list, _data_list = zip(*sorted(_data, key=lambda x: x[0]))
        # shape = N_local_expert, N_methods, N_metric
        _data_array = np.array(_data_list)
        iter_speedup.append((1/_data_array[:, 0, 0][:, None] - 1/_data_array[:, :, 0]) / (1/_data_array[:, :, 0]))

        x = np.arange(len(local_ep_num_list))

        for sub_idx, yaxis_name in enumerate([
            "Iteration Time(ms)",
            "Imbalanced Degree"
            ]):
            fig = plt.figure(figsize=(8, 5))
            ax = plt.subplot(111)
            ax.grid(axis="y")
            for idx in range(len(METHODS)):
                bars = ax.bar(
                    x + idx*barwidth, _data_array[:, idx, sub_idx], width=barwidth, label=METHODS[idx])
                for bar in bars:
                    bar.set_hatch(marks[idx])
            plt.ylabel(yaxis_name, fontsize=fontsize-6)
            plt.xlabel("Original # of experts/worker", fontsize=fontsize-6)
            plt.ylim(0, 1.4*np.max(_data_array[:, :, sub_idx]))
            # plt.legend(fontsize=fontsize, ncol=2)
            plt.xticks(x + (len(METHODS)/2)*barwidth,
                    local_ep_num_list, fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize)
            reduce_tick_num(5, fontsize, axis="y", type=int if sub_idx == 0 else float, low=0, high=1.3)
            # plt.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.95,
                                # wspace=0.2, hspace=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 
                    f"D{distribution_id}_{host_num}x{local_rank_num}_{sub_idx}.pdf"), bbox_inches='tight')
            if not legend_done:
                plot_legend(ax, ncol=4, path=os.path.join(fig_dir, "legend.pdf"))
            plt.close()

    
    iter_speedup = np.concatenate(iter_speedup, axis=0)
    assert len(iter_speedup.shape) == 2
    print(METHODS)
    print(f"max speedup {np.max(iter_speedup, axis=0)}")
    print(f"mean speedup {np.mean(iter_speedup, axis=0)}")