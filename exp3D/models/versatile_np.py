import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import models
from models import register
from utils import rendering


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


@register('versatile_np')
class Versatile_NP(nn.Module):

    def __init__(self, tokenizer, self_attender, cross_attender, hierarchical_model, pe_dim=128):
        super().__init__()
        dim_y = hierarchical_model['args']['dim_y']
        self.pe_dim = pe_dim # dimension of positional embedding

        self.tokenizer = models.make(tokenizer, args={'dim': dim_y})
        self.self_attn = models.make(self_attender)
        self.cross_attn = models.make(cross_attender)
        self.hier_model = models.make(hierarchical_model)

        self.embed_input = build_mlp(pe_dim * 3, dim_y, dim_y, 2)
        self.embed_last = build_mlp(dim_y, dim_y, 4, 2)
        
    def forward(self, data, x_tgt, rays_o, z_vals, y_tgt, is_train=True):
        data_tokens = self.tokenizer(data)
        context_tokens = self.self_attn(data_tokens)
        x_tgt = self.coord_embedding(x_tgt)
        y_middle = self.cross_attn(x_tgt, context_tokens)

        y_middle, kls = self.hier_model(y_middle, x_tgt, y_tgt, rays_o, z_vals, training=is_train)
        y_pred = rendering(self.embed_last(y_middle), rays_o = rays_o, z_vals = z_vals)
        return y_pred, kls

    def coord_embedding(self, x):
        w = 2 ** torch.linspace(0, 8, self.pe_dim // 2, device=x.device)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        x = self.embed_input(x)
        return x