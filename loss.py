import logging
import os
import torch
import torch.nn as nn


# Implement your loss here
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        pass


def build_criterion(opt):
    return Loss()