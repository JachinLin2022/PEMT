#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class MoeGating(nn.Module):
    def __init__(self, emb_size, hid_size, num_models, num_layers, dropout):
        super().__init__()
        linear_models = []
        linear_models.append(nn.Dropout(dropout))
        linear_models.append(nn.Linear(emb_size, hid_size))
        for _ in range(0, num_layers - 1):
            linear_models.append(nn.ReLU())
            linear_models.append(nn.Dropout(dropout))
            linear_models.append(nn.Linear(hid_size, hid_size))
        linear_models.append(nn.ReLU())
        linear_models.append(nn.Dropout(dropout))
        linear_models.append(nn.Linear(hid_size, num_models))
        self.linear = nn.Sequential(*linear_models)
        self.output = nn.Softmax(dim=-1)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1]
            x = x.unsqueeze(1)
        else:
            x = x[-1]
            x = x.unsqueeze(0)
        hid_states = self.linear(x)
        output = self.output(hid_states)
        return output
        
class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        lora_num:int = 1,
        task_prefix_len:int = 1,
        is_decoder:int = 0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.lora_num = lora_num
        self.fan_in_fan_out = fan_in_fan_out
        self.task_prefix_len = 1 if task_prefix_len is None else task_prefix_len
        # Actual trainable parameters
        if r > 0:
            if lora_num == 1:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            else:
                self.lora_A = nn.ParameterList()
                self.lora_B = nn.ParameterList()
                for i in range(lora_num):
                    self.lora_A.append(nn.Parameter(self.weight.new_zeros((r, in_features))))
                    self.lora_B.append(nn.Parameter(self.weight.new_zeros((out_features, r))))

                # gate network
                # self.k = lora_num
                # self.num_experts = lora_num
                # self.noisy_gating = True
                self.w_gate = nn.Parameter(torch.zeros(out_features if is_decoder else in_features, lora_num), requires_grad=True)
                # self.temperature = (out_features * torch.exp(torch.clamp(nn.Parameter(torch.Tensor([1]), requires_grad=True), min=0.005, max=5))).cuda()
                self.temperature = 1
                nn.init.zeros_(self.w_gate)
                print('using zeros_ gate')
                # nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

                # self.softplus = nn.Softplus()
                self.softmax = nn.Softmax(dim=-1)
                # self.register_buffer("mean", torch.tensor([0.0]))
                # self.register_buffer("std", torch.tensor([1.0]))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            if self.lora_num == 1:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                # nn.init.zeros_(self.lora_B)
                nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            else:
                for i in range(self.lora_num):
                    nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[i])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

   

    def forward(self, x: torch.Tensor, moe_output=None, encoder_hidden_states=None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            gating_weights = None
            all_results=None
            if self.lora_num == 1:         
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
                return result
            else:
                # gates, load = self.noisy_top_k_gating(x, self.training)
                # importance = gates.sum(0)

                # using token 0 as task vector
                # print(self.task_prefix_len)

                # if x.shape[1] > self.task_prefix_len:
                #     gate_x = x[:,:self.task_prefix_len,:].mean(1).unsqueeze(1)
                # else:
                #     gate_x = x[:,0,:].unsqueeze(1)
                if encoder_hidden_states is not None:
                    gate_x = encoder_hidden_states[:,:self.task_prefix_len,:].mean(1).unsqueeze(1)
                else:
                    gate_x = x[:,:self.task_prefix_len,:].mean(1).unsqueeze(1)


                gete_out = gate_x @ self.w_gate

                # gete_out = (x @ self.w_gate).sum(dim=1).unsqueeze(1)



                # gate = gate_x@self.w_gate
                # gate = torch.nn.functional.softmax(gate.sum(0))
                # for i in range(self.lora_num):
                #     result += (self.lora_dropout(x) @ self.lora_A[i].transpose(0, 1) @ self.lora_B[i].transpose(0, 1)) * self.scaling * gate[i].item()
                

                if 1 or self.training:
                    gating_weights = self.softmax(gete_out/self.temperature)
                else:
                    print('argmax')
                    max_index = torch.argmax(gete_out,dim=2)
                    gating_weights = torch.zeros_like(gete_out)
                    gating_weights.scatter_(2, max_index.unsqueeze(2), 1)

                # gete_out[:,:,0] = gete_out[:,:,0] + 999999999
                # gete_out[:,:,1:] = 0
                
                
                if moe_output:
                    pass
                    # moe_output.append(gating_weights.cpu().numpy())
                
                # tmp_x = self.lora_dropout(x).expand(self.lora_num, -1, -1, -1).view(self.lora_num, -1, 3072)
                # # 将参数堆叠成3D张量
                # A_stacked = torch.stack([a.transpose(0, 1) for a in self.lora_A], dim=0)
                # B_stacked = torch.stack([b.transpose(0, 1) for b in self.lora_B], dim=0)

                # # 执行批量矩阵乘法
                # intermediate_results = torch.bmm(tmp_x, A_stacked)
                # all_results = torch.bmm(intermediate_results, B_stacked).view(self.lora_num, 128, -1, 768)
                
                # # 应用缩放并堆叠
                # final_output = all_results * self.scaling
                # final_output = torch.stack([final_output[i] for i in range(self.lora_num)], dim=3) @ gating_weights.unsqueeze(3)
                
                all_results = []
                x = self.lora_dropout(x)
                for i in range(self.lora_num):
                    all_results.append((x @ self.lora_A[i].transpose(0, 1) @ self.lora_B[i].transpose(0, 1)) * self.scaling)
                final_output = torch.stack(all_results, dim=3) @ gating_weights.unsqueeze(3)
                final_output = final_output.squeeze()
                if len(final_output.shape) == 2:
                    final_output = final_output.unsqueeze(1)
                result += final_output
            return result, gating_weights, all_results
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

from typing import Dict
def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def task_embedding_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'task_shared' in k}
    else:
        raise NotImplementedError

def adapter_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'adapters' in k}
    else:
        raise NotImplementedError
