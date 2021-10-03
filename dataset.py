import torch
import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader


class Dataset(data.Dataset):
    def __init__(self):
        super().__init__()


    def __len__(self):
        pass


    def __getitem__(self, index):
        pass


def build_loader(opt):
    # set parameters of your Dataset according to config
    train_set = Dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, 
                            drop_last=False, num_workers=opt.num_workers, 
                            sampler=train_sampler, pin_memory=opt.pin_memory)

    eval_set = Dataset()
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set)
    eval_loader = DataLoader(eval_set, batch_size=opt.batch_size*4, shuffle=False, 
                            drop_last=False, num_workers=opt.num_workers,
                            sampler=eval_sampler, pin_memory=opt.pin_memory)
    
    test_loader = None
    if opt.val_test:
        test_set = Dataset()
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        test_loader = DataLoader(test_set, batch_size=opt.batch_size*4, shuffle=False, 
                                drop_last=False, num_workers=opt.num_workers,
                                sampler=test_sampler, pin_memory=opt.pin_memory)
    
    return train_loader, train_sampler, eval_loader, test_loader


if __name__ == '__main__':
    dataset = Dataset()
    for i in range(32):
        ret = dataset.__getitem__(i)
