import logging
import os
import torch
import torch.nn as nn


# Implement your loss here
# In your loss, it return a list of various loss
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        pass


def build_criterion(opt):
    return Loss()


if __name__=='__main__':
    pass