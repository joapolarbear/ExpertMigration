import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt

workspace = ".workspace/traces/toy_example"
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
    for (host_num, local_rank_num), _data in _rst_dict.items():
        local_ep_num_list, _data_list = zip(*sorted(_data, key=lambda x: x[0]))
        # shape = N_local_expert, N_methods, N_metric
        _data_array = np.array(_data_list)

        x = np.arange(len(local_ep_num_list))

        for sub_idx, yaxis_name in enumerate([
            "Iteration Time(ms)",
            "Imbalanced Degree"
            ]):
            fig = plt.figure(figsize=(8, 5))
            ax = plt.subplot(111)
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
            plt.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.95,
                                wspace=0.2, hspace=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 
                    f"D{distribution_id}_{host_num}x{local_rank_num}_{sub_idx}.pdf"), bbox_inches='tight')
            if not legend_done:
                plot_legend(ax, ncol=4, path=os.path.join(fig_dir, "legend.pdf"))
            plt.close()
