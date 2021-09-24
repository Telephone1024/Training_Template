import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import accuracy, AverageMeter


# Implement your model here
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Trainer(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = model.device


    def _decode_data(self, data, opt):
        # Decode the data from loader, data is always a tuple of X, y
        pass        


    def step(self, data, opt):
        self.model.train()

        inputs, targets = self._decode_data(data, opt)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(opt.cur_step) # modify this line if plateau scheduler is adopted

        return loss.item(), targets.shape[0]
        

    @torch.no_grad()
    def pred(self, data, opt):
        self.model.eval()
        
        inputs, targets = self._decode_data(data, opt)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        acc = accuracy(outputs, targets)[0]

        return acc, loss.item(), targets.shape[0]


def build_model(opt):
    return Model()


def build_trainer(model, criterion, optimizer, scheduler, opt):
    return Trainer(model, criterion, optimizer, scheduler)


if __name__=='__main__':
    inputs = torch.randn(2, 3, 224, 224)
    net = Model()
    outputs = net(inputs)
    from torchsummary import summary
    summary(net, (3, 224, 224), device='cpu')