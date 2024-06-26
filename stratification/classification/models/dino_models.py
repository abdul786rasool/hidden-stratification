import torch
import torch.nn as nn


def Dino_Model(**kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    model.activation_layer_name = 'head'
    return model