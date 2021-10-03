'''
Written by Telephone1024
2021/09/24
# This part needs no modification if not necessary #
'''


import logging
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter, reduce_tensor

# Implement your training settinga and model(s) in these files
from config import get_config
from models import build_model, build_trainer
from loss import build_criterion
from dataset import build_loader
from optimizer import build_optimizer
from scheduler import build_scheduler
from utils import lr_adjust, save_checkpoint


def train(opt):
    model = build_model(opt)
    criterion = build_criterion(opt)
    train_loader, train_sampler, eval_loader, test_loader = build_loader(opt)
    optimizer = build_optimizer(opt, model)
    scheduler = build_scheduler(opt, optimizer, len(train_loader))

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    trainer = build_trainer(model, criterion, optimizer, scheduler, opt)
    
    if 0 == opt.local_rank:
        logging.info('Training Starts!')
        opt.best_eval_acc = 0.
        opt.best_eval_acc_epoch = 0
        opt.best_eval_acc_loss = 1e5
        if opt.val_test:
            opt.best_test_acc = 0.
            opt.best_test_acc_epoch = 0
            opt.best_test_acc_loss = 1e5

    for epoch in range(opt.num_epochs):
        # through set_epoch function , sampler can shuffle the data on each process
        train_sampler.set_epoch(epoch)
        train_one_epoch(opt, epoch, trainer, train_loader)
        if 0 == opt.local_rank and 0 == (epoch+1)%opt.save_interval:
            save_checkpoint(trainer.model, opt, epoch+1)

        if 0 == opt.local_rank and 0 == (epoch+1)%opt.val_interval:        
            eval(opt, epoch, trainer, eval_loader)
            if opt.val_test:
                eval(opt, epoch, trainer, test_loader, stage='Test')
    
    if 0 == opt.local_rank:
        logging.info('Training Over!')
        logging.info('Best Eval Acc is %.5f, Corresponding Epoch is %03d'%(opt.best_eval_acc, opt.best_eval_acc_epoch))
        if opt.use_tb:
            opt.writer.close()


def train_one_epoch(opt, epoch, trainer, data_loader):

    loss_meter = AverageMeter()

    for iter, data in enumerate(data_loader):
        opt.cur_step = iter + epoch*len(data_loader)

        loss, n = trainer.step(data, opt)

        torch.cuda.synchronize()

        loss_meter.update(loss, n)

        if 0 == opt.local_rank and 0 == (iter+1)%opt.train_print_fre:
            curr_lr = trainer.optimizer.param_groups[0]['lr']
            logging.info(
                f'Train: [{epoch+1:03d}/{opt.num_epochs:03d}][{iter+1:03d}/{len(data_loader):03d}]\t'
                f'lr {curr_lr:.4e}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
            if opt.use_tb:
                opt.writer.add_scalar('lr', curr_lr, opt.cur_step)
                opt.writer.add_scalar('Train_loss', loss_meter.val, opt.cur_step)


@torch.no_grad()
def validate(opt, epoch, trainer, data_loader, stage='Eval'):
    torch.cuda.empty_cache()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    

    for iter, data in enumerate(data_loader):
        acc, loss, n = trainer.pred(data, opt)
        # acc = accuracy()
        acc = reduce_tensor(acc, dist.get_world_size())
        loss = reduce_tensor(loss, dist.get_world_size())

        acc_meter.update(acc, n)
        loss_meter.update(loss, n)

        if 0 == opt.local_rank and 0 == (iter+1)%opt.val_print_fre:
            logging.info(
                f'{stage}: [{iter+1:04d}/{len(data_loader):04d}]\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})')
            if opt.use_tb:
                opt.writer.add_scalar('%s_loss'%(stage), loss_meter.val, iter + epoch*len(data_loader))
                opt.writer.add_scalar('%s_acc'%(stage), acc_meter.val, iter + epoch*len(data_loader))

    return acc_meter.avg, loss_meter.avg


def eval(opt, epoch, trainer, data_loader, stage='Eval'):
    acc, loss = validate(opt, epoch, trainer, data_loader)
    logging.info('%s Average Acc: %.4f, Loss: %.4f'%(stage, acc, loss))
    if opt.use_tb:
        opt.writer.add_scalar('%s_acc_avg'%(stage), acc, epoch+1)
        opt.writer.add_scalar('%s_loss_avg'%(stage), loss, epoch+1)
    
    stage = stage.lower()
    
    if acc >= opt.__dict__['best_%s_acc'%(stage)]:
        if acc == opt.__dict__['best_%s_acc'%(stage)]:
            if loss < opt.__dict__['best_%s_acc_loss'%(stage)]:
                # if acc is equal, update epoch and loss value if loss is less
                opt.__dict__['best_%s_acc_epoch'%(stage)] = epoch
                opt.__dict__['best_%s_acc_loss'%(stage)] = loss
        else:
            opt.__dict__['best_%s_acc_epoch'%(stage)] = epoch
            opt.__dict__['best_%s_acc_loss'%(stage)] = loss
        
        opt.__dict__['best_%s_acc'%(stage)] = acc

        save_checkpoint(trainer.model, opt, epoch+1, True, stage)


if __name__=='__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    opt = get_config()
    opt.local_rank = local_rank

    dist.init_process_group(backend='nccl')
    # dist.barrier()
    torch.cuda.set_device(local_rank)

    cudnn.benchmark = True

    if opt.adjust_lr:
        opt = lr_adjust(opt)

    if 0 == local_rank:
        import time
        saved_path = os.path.join(opt.saved_path, time.strftime('%m-%d-%H:%M', time.localtime()))
        os.makedirs(saved_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(saved_path, 'run.log'),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO
        )
        os.system('cp config.py %s'%(saved_path))
        logging.info('Training config saved to %s'%(saved_path))
        opt.saved_path = saved_path
        if opt.use_tb:
            opt.writer = SummaryWriter(os.path.join(opt.saved_path, 'tensorboard'))         


    train(opt)