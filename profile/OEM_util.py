import os
import torch
import time
import json

import deepspeed

print("Hi, OEM")

def get_timestamp():
    torch.cuda.synchronize()
    return time.time()

class Profiler:
    def __init__(self, workspace=".workspace"):
        
        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)

        self.rank = torch.distributed.get_rank()
        self.ep_world_size = torch.distributed.get_world_size()

        self.step_num = 0

        self.name_stat = {}
        self.start_time = get_timestamp()
        self.timestamps = []

    def dump(self, path=None):
        _path = path or os.path.join(self.workspace, f"rawtrace_r{self.rank}_i{self.step_num}.txt")
        with open(_path, 'w') as fp:
            json.dump({
                "timestamps": self.timestamps,
                "rank": self.rank,
                "ep_world_size": self.ep_world_size,
                "start_time": self.start_time
                }, fp, indent=4)

    def get_hook_fn(self, _name, phase="FW"):
        if phase == "FW_PRE":
            def hook_fn(module, input):
                ### forward pre hook
                t = get_timestamp()
                unique_name = self.unique_name(_name, phase=phase)
                attr = []
                self.timestamps.append(("FW_PRE", unique_name, t, attr))
        elif phase == "FW":
            def hook_fn(module, input, output):
                ### forward hook
                t = get_timestamp()
                unique_name = self.unique_name(_name, phase=phase)
                attr = []
                if isinstance(module, deepspeed.moe.sharded_moe.TopKGate):
                    _, _, dispatch_mask, exp_counts, mask_wo_drop, mask_w_drop = output
                    # print(mask_wo_drop, mask_w_drop)
                    attr = [mask_wo_drop.detach().tolist(), mask_w_drop.detach().tolist()]
                elif isinstance(module, deepspeed.moe.experts.Experts):
                    ### input[0] is of shape [N_worker, N_expert/worker, capacity, d_model]
                    # print(len(input), input[0].shape)
                    attr = list(input[0].shape)
                self.timestamps.append(("FW", unique_name, t, attr))
        elif phase == "BW":
            def hook_fn(module, grad_input, grad_output):
                ### backward hook
                t = get_timestamp()
                unique_name = self.unique_name(_name, phase=phase)
                attr = []
                if isinstance(module, deepspeed.moe.experts.Experts):
                    ### input[0] is of shape [N_worker, N_expert/worker, capacity, d_model]
                    # print(len(input), input[0].shape)
                    attr = list(grad_input[0].shape)
                self.timestamps.append(("BW", unique_name, t, attr))
        else:
            raise
        return hook_fn
    
    def register(self, model):
        for name, module in model.named_children():
            # print(name, type(module))
            self.register_implt(name, module)

    def register_implt(self, name, module):
            if isinstance(module, torch.nn.ModuleList):
                for sub_module in module:
                    self.register_implt(name, sub_module)
            elif isinstance(module, deepspeed.moe.layer.MoE):
                self._register_implt(module.deepspeed_moe.gate, "MoE_Gate")
                self._register_implt(module.deepspeed_moe.experts, "MoE_Experts")
            else:
                self._register_implt(module, name)

    def _register_implt(self, module, _name):
        module.register_forward_pre_hook(self.get_hook_fn(_name, phase="FW_PRE"))
        module.register_forward_hook(self.get_hook_fn(_name))
        module.register_full_backward_hook(self.get_hook_fn(_name, phase="BW"))

    def unique_name(self, name, phase="FW"):
        if phase == "FW_PRE":
            if name in self.name_stat:
                self.name_stat[name]["cnt_in_step"] += 1
            else:
                self.name_stat[name] = {"cnt_in_step":  0}
            _unique_name = f'{name}_{self.name_stat[name]["cnt_in_step"]}'
        elif phase == "FW":
            _unique_name = f'{name}_{self.name_stat[name]["cnt_in_step"]}'
        else:
            assert name in self.name_stat
            _unique_name = f'{name}_{self.name_stat[name]["cnt_in_step"]}'
            self.name_stat[name]["cnt_in_step"] -= 1
        return _unique_name

    def step_start(self):
        self.name_stat = {}
        self.timestamps = []
        self.start_time = get_timestamp()

    def step_end(self, dump=False):
        t = get_timestamp()
        self.timestamps.append(("UPDATE_", "UPDATE", t, []))
        if dump:
            self.dump()
        self.step_num += 1

