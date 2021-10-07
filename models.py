import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter, reduce_tensor


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
        if opt.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip_grad)
        self.optimizer.step()
        self.scheduler.step_update(opt.cur_step) # modify this line if plateau scheduler is adopted

        return loss.item(), targets.shape[0]
        

    @torch.no_grad()
    def pred(self, data, opt):
        self.model.eval()
        
        inputs, targets = self._decode_data(data, opt)
        outputs = self.model(inputs)

        loss = self.criterion(outputs, targets)
        acc = accuracy(outputs, targets)[0]

        acc = reduce_tensor(acc, dist.get_world_size())
        loss = reduce_tensor(loss, dist.get_world_size())

        return acc, loss.item(), targets.shape[0]


def build_model(opt):
    net = Model()
    # net can be any other models you want
    if 0 == opt.local_rank:
        opt.logger.info('Model is built!')
        if opt.use_tb:
            opt.writer.add_graph(net, torch.randn(2, 3, opt.img_size, opt.img_size))
    if opt.resume:
        net.load_state_dict(torch.load(opt.resume, map_location='cpu'), strict=False)
        # dist.barrier()
        if 0 == opt.local_rank:
            opt.logger.info('Load pretrained weight from %s'%(opt.resume))

    return net


def build_trainer(model, criterion, optimizer, scheduler, opt):
    return Trainer(model, criterion, optimizer, scheduler)


if __name__=='__main__':
    inputs = torch.randn(2, 3, 224, 224)
    net = Model()
    outputs = net(inputs)