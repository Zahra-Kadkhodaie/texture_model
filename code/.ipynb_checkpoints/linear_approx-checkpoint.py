import numpy as np
import torch
import time

#########


def traj_projections(inter_ys, denoiser, x_cond=None, skip=True): 
    with torch.no_grad():
        if x_cond is None:
            out = denoiser(inter_ys).detach()
        else: 
            out = denoiser(inter_ys, x_cond).detach()       
            
        if skip: 
            out = inter_ys - out 
            
    return out