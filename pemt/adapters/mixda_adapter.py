import torch
from torch import nn
import math

# Implement example reweighting
class WeightTensor(nn.Module):
    def __init__(self, example_num, batch_size):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(batch_size)) for _ in range((example_num - example_num % batch_size) // batch_size)])
        if example_num % batch_size != 0:
            self.weights.append(nn.Parameter(torch.ones(example_num % batch_size)))
    def forward(self, batch_id):
        return self.weights[batch_id].softmax(dim=0)

# Main class, Mixture-of-Domain adapters
class MixtureOfDomainAdapter(nn.Module):
    def __init__(self, config, down_scale=None, input_size=None, in_feature=None, mid_feature=None, out_feature=None):
        super().__init__()
        self.config = config
        adapter_down_scale = down_scale if down_scale is not None else config.adapter_down_scale
        self.down_sample = int(config.d_ff // adapter_down_scale)
        self.input_size = input_size if input_size else config.d_ff
        self.output_size = config.d_model
        if in_feature is not None and mid_feature is not None and out_feature is not None:
            self.input_size, self.down_sample, self.output_size = in_feature, mid_feature, out_feature
        self.adapter_down = nn.Sequential(
            nn.Linear(self.input_size, self.down_sample),
            nn.GELU(),
        )
        self.adapter_up = nn.Linear(self.down_sample, self.output_size)
        
        # initialize weights
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.adapter_down[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down[0].bias)
            nn.init.zeros_(self.adapter_up.bias)
            nn.init.zeros_(self.adapter_up.weight)
    def forward(self, x):
        down = self.adapter_down(x)
        up = self.adapter_up(down)
        output = up
        return output

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