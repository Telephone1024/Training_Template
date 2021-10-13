import os
import torch
import torch.nn as nn


# Implement your loss here
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        # In your loss, it return a list of various loss
        pass


def build_criterion(opt):
    return Loss()


if __name__=='__main__':
    pass