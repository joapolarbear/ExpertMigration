import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rcParams['pdf.fonttype'] = 42

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
METHODS = ["OEM", "FastMoE", "FasterMoE", "REM"]

def plot_legend(ax, ncol, path, column=1):     
    label_params = ax.get_legend_handles_labels()
    figl, axl = plt.subplots(figsize=(25*column, 2))
    axl.axis(False)

    axl.legend(*label_params,
        ncol=ncol, 
        loc="center", 
        bbox_to_anchor=(0.5, 0.5), 
        frameon=False,
        fontsize=fontsize,
        # mode = "expand"
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
    throughput_speedup = []
    for (host_num, local_rank_num), _data in _rst_dict.items():
        local_ep_num_list, _data_list = zip(*sorted(_data, key=lambda x: x[0]))
        # shape = N_local_expert, N_methods, N_metric
        _data_array = np.array(_data_list)
        throughput_speedup.append((1/_data_array[:, 0, 0][:, None]) / (1/_data_array[:, :, 0]))
        iter_speedup.append((_data_array[:, :, 0] - _data_array[:, 0, 0][:, None]) / (_data_array[:, :, 0]))

        x = np.arange(len(local_ep_num_list))

        xticks = np.array(local_ep_num_list) * host_num * local_rank_num

        for metric_idx, yaxis_name in enumerate([
            "Iteration Time(ms)",
            "Imbalanced Degree",
            "Resource Util. (%)"
            ]):
            if metric_idx == 1 or metric_idx == 2:
                fig = plt.figure(figsize=(8, 5))
            else:
                fig = plt.figure(figsize=(10, 5))
            ax = plt.subplot(111)
            ax.grid(axis="y")

            if metric_idx == 2:
                _data = 100 * _data_array[:, -1, 0][:, None] / _data_array[:, :-1, 0]
            else:
                _data = _data_array[:, :-1, metric_idx]
                
            for idx in range(len(METHODS)):
                bars = ax.bar(x + idx*barwidth, _data[:, idx], width=barwidth, label=METHODS[idx])
                for bar in bars:
                    bar.set_hatch(marks[idx])
            plt.ylabel(yaxis_name, fontsize=fontsize-6)
            plt.xlabel("# of experts", fontsize=fontsize-6)
            # plt.ylim(0, 1.2*np.max(_data))
            # plt.legend(fontsize=fontsize, ncol=2)
            plt.xticks(x + (len(METHODS)/2)*barwidth,
                    xticks, fontsize=fontsize-6, rotation=0)
            plt.yticks(fontsize=fontsize-6)
            if (distribution_id == 1 and metric_idx == 0) or (distribution_id == 2 and metric_idx == 1):
                diff = np.max(_data) - np.min(_data)
                plt.ylim(np.min(_data) - 2 * diff, np.max(_data) + 0.1* diff)
                reduce_tick_num(6, fontsize, axis="y", type=int if metric_idx == 0 else float, low=1, high=1)
            else:
                reduce_tick_num(5, fontsize, axis="y", type=int if metric_idx == 0 else float, low=0, high=1.2)
            # plt.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.95,
                                # wspace=0.2, hspace=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 
                    f"D{distribution_id}_{host_num}x{local_rank_num}_{metric_idx}.pdf"), bbox_inches='tight')
            if not legend_done:
                plot_legend(ax, ncol=4, path=os.path.join(fig_dir, "legend.pdf"))
                plot_legend(ax, ncol=4, path=os.path.join(fig_dir, "legend_2column.pdf"), column=2)
            plt.close(fig)
    
    throughput_speedup = np.concatenate(throughput_speedup, axis=0)
    iter_speedup = np.concatenate(iter_speedup, axis=0)
    # assert len(iter_speedup.shape) == 2
    print(METHODS)
    print(f"max/mean iteration speedup {np.max(iter_speedup, axis=0)} / {np.mean(iter_speedup, axis=0)}")
    print(f"max/mean throughput speedup {np.max(throughput_speedup, axis=0)} / {np.mean(throughput_speedup, axis=0)}")