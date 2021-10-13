from typing import OrderedDict
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

        self.acc_meter = AverageMeter()
        # define your loss here, the first one is sum of loss, others are details
        self.loss_meter = [AverageMeter(), AverageMeter(), AverageMeter()]

    def _decode_data(self, data, opt):
        # Decode the data from loader, data is always a tuple of X, y
        pass

    def _decode_loss(self, all_loss, n):
        assert isinstance(all_loss, tuple)
        for i in len(self.loss_meter):
            loss = reduce_tensor(all_loss[i].item(), dist.get_world_size())
            self.loss_meter.update(loss, n)

    def step(self, data, opt):
        self.model.train()

        inputs, targets = self._decode_data(data, opt)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        loss[0].backward()
        if opt.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip_grad)
        self.optimizer.step()
        if not 'plateau' == opt.sche_type:
            self.scheduler.step_update(opt.cur_step)

        return self.get_meter()

    @torch.no_grad()
    def val(self, data, opt, stage):
        self.model.eval()
        
        inputs, targets = self._decode_data(data, opt)
        outputs = self.model(inputs)

        loss = self.criterion(outputs, targets)

        self._decode_loss(loss, targets.shape[0])

        acc = accuracy(outputs, targets)[0]
        acc = reduce_tensor(acc, dist.get_world_size())
        if 'plateau' == opt.sche_type and 'Val' == stage:
            self.scheduler.step(opt.cur_epoch, acc)
        self.acc_meter.update(acc, targets.shape[0])
        
        return self.get_meter()

    def reset_meter(self):
        self.acc_meter.reset()
        for i in len(self.loss_meter):
            self.loss_meter.reset()

    def get_meter(self):
        # define your loss names here
        loss_name = ['total_loss', 'loss_1', 'loss_2']
        loss = OrderedDict()
        # ret.update({'acc':(self.acc_meter.val, self.acc_meter.avg)})
        for i in len(self.loss_meter):
            loss.update({loss_name[i]:(self.loss_meter[i].val, self.loss_meter[i].avg)})
        
        return (self.acc_meter.val, self.acc_meter.avg), loss
        

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