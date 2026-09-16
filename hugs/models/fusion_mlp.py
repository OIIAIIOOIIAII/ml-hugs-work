import torch
from torch import nn

from hugs.utils.general import get_expon_lr_func


class ResidualMLP(nn.Module):
    def __init__(self, prop_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(prop_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, prop_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(x)))


class HumanSceneFuseDecoder(nn.Module):
    """Direct port of STM HumanSceneFuseDecoder.

    Concatenates human + scene GS along the point dimension, runs a per-property
    ResidualMLP (fc2 zero-initialized → identity at init), returns fused_props
    containing all N_human + N_scene points together.

    shs is handled as (N, 48) flat vectors (reshape from (N, 16, 3)) matching
    both human (AppearanceDecoder outputs 16*3) and scene (sh_degree=3) dims.
    """

    def __init__(self, property_dims, hidden_dim=64):
        super().__init__()
        self.fusion_layers = nn.ModuleDict({
            prop_name: ResidualMLP(prop_dim, hidden_dim)
            for prop_name, prop_dim in property_dims.items()
        })

    def forward(self, human_props, scene_props):
        fused_props = {}
        for prop_name in human_props.keys():
            concatenated = torch.cat([human_props[prop_name], scene_props[prop_name]], dim=0)
            fused_props[prop_name] = self.fusion_layers[prop_name](concatenated)
        return fused_props

    def setup_optimizer(self, lr_init, lr_final, lr_delay_mult, max_steps):
        params = [{'params': self.parameters(), 'lr': lr_init, 'name': 'fusion_mlp_params'}]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.scheduler_args = get_expon_lr_func(
            lr_init=lr_init,
            lr_final=lr_final,
            lr_delay_mult=lr_delay_mult,
            max_steps=max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "fusion_mlp_params":
                lr = self.scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
