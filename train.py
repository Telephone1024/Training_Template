'''
Written by Telephone1024
2021/09/24
# This part needs no modification if not necessary #
'''
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

# Implement your training settinga and model(s) in these files
from config import get_config
from models import build_model, build_trainer
from loss import build_criterion
from dataset import build_loader
from optimizer import build_optimizer
from scheduler import build_scheduler
from utils import build_logger, lr_adjust, save_checkpoint


def train(opt):
    model = build_model(opt)
    criterion = build_criterion(opt)
    train_loader, train_sampler, val_loader, test_loader = build_loader(opt)
    optimizer = build_optimizer(opt, model)
    scheduler = build_scheduler(opt, optimizer, len(train_loader))

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    trainer = build_trainer(model, criterion, optimizer, scheduler, opt)
    
    if 0 == opt.local_rank:
        opt.logger.info('Training Starts!')
        opt.best_val_acc = 0.
        opt.best_val_acc_epoch = 0
        opt.best_val_acc_loss = 1e5
        if opt.val_test:
            opt.best_test_acc = 0.
            opt.best_test_acc_epoch = 0
            opt.best_test_acc_loss = 1e5

    for epoch in range(opt.num_epochs):
        # through set_epoch function , sampler can shuffle the data on each process
        train_sampler.set_epoch(epoch)
        opt.cur_epoch = epoch
        train_one_epoch(opt, epoch, trainer, train_loader)
        if 0 == opt.local_rank and 0 == (epoch+1)%opt.save_interval:
            save_checkpoint(trainer.model, opt, epoch+1)

        if 0 == (epoch+1)%opt.val_interval:        
            validate(opt, epoch, trainer, val_loader)
            if opt.val_test:
                validate(opt, epoch, trainer, test_loader, stage='Test')
    
    if 0 == opt.local_rank:
        opt.logger.info('Training Over!')
        opt.logger.info('Best Val Acc is %.5f, Corresponding Epoch is %03d'%(opt.best_val_acc, opt.best_val_acc_epoch+1))
        if opt.use_tb:
            opt.writer.close()


def train_one_epoch(opt, epoch, trainer, data_loader):
    torch.cuda.empty_cache()

    trainer.reset_meter()

    for iter, data in enumerate(data_loader):
        opt.cur_step = iter + epoch*len(data_loader)

        _, loss = trainer.step(data, opt)

        torch.cuda.synchronize()
        

        if 0 == opt.local_rank and 0 == (iter+1)%opt.train_print_fre:
            curr_lr = trainer.optimizer.param_groups[0]['lr']
            loss_msg = ''
            for loss_name, loss_val in loss.items():
                loss_msg += f'\t{loss_name} {loss_val[0]:.4f} ({loss_val[1]:.4f})'
                loss[loss_name] = loss_val[0] # For tensorboard
            opt.logger.info(
                f'Train: [{epoch+1:03d}/{opt.num_epochs:03d}][{iter+1:03d}/{len(data_loader):03d}]\t' + 
                f'lr {curr_lr:.4e}' +
                loss_msg)
            if opt.use_tb:
                opt.writer.add_scalar('lr', curr_lr, opt.cur_step)
                opt.writer.add_scalars('Train_loss', loss, opt.cur_step)


@torch.no_grad()
def validate(opt, epoch, trainer, data_loader, stage='Val'):
    torch.cuda.empty_cache()
    trainer.reset_meter()

    for iter, data in enumerate(data_loader):
        acc, loss = trainer.val(data, opt, stage)

        if 0 == opt.local_rank and 0 == (iter+1)%opt.val_print_fre:
            loss_msg = ''
            for loss_name, loss_val in loss.items():
                loss_msg += f'\t{loss_name} {loss_val[0]:.4f} ({loss_val[1]:.4f})'
                loss[loss_name] = loss_val[1] # For tensorboard
            
            opt.logger.info(
                f'{stage}: [{iter+1:04d}/{len(data_loader):04d}]\t' +
                f'Acc {acc[0]:.4f} ({acc[1]:.4f})' +
                loss_msg)

    acc = acc[1] # get average acc and total loss
    
    if 0 == opt.local_rank:
        opt.logger.info('%s Average Acc: %.4f, Loss: %.4f'%(stage, acc, loss))
        if opt.use_tb:
            opt.writer.add_scalar('%s_acc_avg'%(stage), acc, epoch+1)
            opt.writer.add_scalars('%s_loss_avg'%(stage), loss, epoch+1)
    
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
        opt.saved_path = saved_path
        # build logger export to console and file
        opt.logger = build_logger(saved_path)
        os.system('cp config.py %s'%(saved_path))
        opt.logger.info('Training config saved to %s'%(saved_path))
        if opt.use_tb:
            opt.writer = SummaryWriter(os.path.join(opt.saved_path, 'tensorboard'))         


    train(opt)