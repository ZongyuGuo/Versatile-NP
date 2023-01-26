import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
import einops

from models import register
from utils import rendering

from .blocks import StyledConv, ModulatedConv2d

def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


@register('hier_model')
class Hierarchical_Model(nn.Module):
    def __init__(self, depth, dim_y, dim_hid, dim_lat):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(ModBlock(dim_y=dim_y, dim_hid=dim_hid, dim_lat=dim_lat))

    def forward(self, y, x_tgt, y_tgt, rays_o, z_vals, training=True):
        kls = 0
        for layer in self.layers:
            y, kld = layer(y, x_tgt, y_tgt, rays_o, z_vals, training=training)
            kls += kld
        return y, kls


class ModBlock(nn.Module):
    def __init__(self, dim_y, dim_hid, dim_lat):
        super().__init__()
        self.input_mlp = build_mlp(dim_y, dim_hid, 4, depth=3)

        self.merge_p = build_mlp(3, dim_hid, dim_hid, depth=2)
        self.merge_q = build_mlp(6, dim_hid, dim_hid, depth=2)
        self.latent_encoder_p = build_mlp(dim_hid, dim_hid, dim_lat * 2, depth=2)
        self.latent_encoder_q = build_mlp(dim_hid, dim_hid, dim_lat * 2, depth=2)
        self.latent_decoder = build_mlp(dim_lat, dim_hid, dim_hid, depth=2)
    
        self.unmod_mlp = build_mlp(dim_y * 2, dim_hid, dim_y, depth=2)
        self.mod_conv1 = StyledConv(dim_y, dim_hid, 1, dim_hid, demodulate=True)# with built-in activation function
        self.mod_conv2 = ModulatedConv2d(dim_hid, dim_y, 1, dim_hid, demodulate=True) # without built-in activation function
        self.mod_conv2.weight.data *= np.sqrt(1 / 6)

    def forward(self, y, x_tgt, y_tgt, rays_o, z_vals, training=True):
        # 3D points -> 2D image pixels
        y_rgb = rendering(self.input_mlp(y), rays_o=rays_o, z_vals=z_vals)
        
        # variational inference in Neural Process (with an average pooling across all points)
        # if not training, y_tgt is not used
        z, kld = self.forward_latent(y_rgb, y_tgt, training=training)
        
        # unmodulated and modulated layers leveraging latent variables
        y = self.forward_mlps(y, x_tgt, latent=z)
        return y, kld

    def forward_latent(self, y_rgb, y_tgt, training):
        z_prior = self.merge_p(y_rgb).mean(dim=-2, keepdim=True)
        dist_prior = self.normal_distribution(self.latent_encoder_p(z_prior))

        if training:
            z_posterior = self.merge_q(torch.cat([y_rgb, y_tgt], dim=-1)).mean(dim=-2, keepdim=True)
            dist_posterior = self.normal_distribution(self.latent_encoder_q(z_posterior))
            z = dist_posterior.rsample()
            kld = kl_divergence(dist_posterior, dist_prior).sum(-1)
        else:
            z = dist_prior.rsample()
            kld = torch.zeros_like(z)

        z = self.latent_decoder(z)
        return z, kld

    def forward_mlps(self, y, x_tgt, latent):
        b, num_tgt, c = y.shape
        
        # Two unmodulated MLPs (residual structure)
        # We enhance feature y by combining x_tgt, which can actually be omitted.
        y = y + self.unmod_mlp(torch.cat([x_tgt, y], dim=-1))

        # Two modulated MLPs (residual structure)
        # Modulated MLP is implemented in the 2D form.        
        y_res = einops.rearrange(y, 'b n c -> b c n 1').contiguous()
        y_res = self.mod_conv1(y_res, latent)
        y_res = self.mod_conv2(y_res, latent)
        y_res = einops.rearrange(y_res, 'b c n 1 -> b n c').contiguous()
        
        y = y + y_res
        return y

    @staticmethod
    def normal_distribution(input):
        mean, var = input.chunk(2, dim=-1)
        var = 0.1 + 0.9 * torch.sigmoid(var)
        dist = Normal(mean, var)
        return dist

