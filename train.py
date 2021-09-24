import logging
import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter

# Implement your training set and model in these files
from config import get_config
from models import build_model, build_trainer
from loss import build_criterion
from dataset import build_loader
from optimizer import build_optimizer
from scheduler import build_scheduler
from utils import save_checkpoint


def train(opt):
    model = build_model(opt)
    if opt.resume:
        model.load_state_dict(torch.load(opt.resume, map_location='cpu'), strict=False)
        if 0 == opt.local_rank:
            logging.info('Load pretrained weight from %s'%(opt.resume))

    criterion = build_criterion(opt)
    train_loader, eval_loader = build_loader(opt)
    optimizer = build_optimizer(opt, model)
    scheduler = build_scheduler(opt, optimizer, len(train_loader))

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    trainer = build_trainer(model, criterion, optimizer, scheduler, opt)
    
    if 0 == opt.local_rank:
        logging.info('Training Starts!')
        opt.best_acc = 0.
        opt.best_acc_epoch = 0
        opt.best_acc_loss = 1e5

    for epoch in range(opt.num_epochs):

        train_one_epoch(opt, epoch, trainer, train_loader)
        if 0 == opt.local_rank and 0 == (epoch+1)%opt.save_interval:
            save_checkpoint(trainer.model, opt, epoch+1)

        if 0 == opt.local_rank and 0 == (epoch+1)%opt.val_interval:        
            acc, loss = validate(opt, trainer, eval_loader)
            # logging.info('Validation Acc: %.5f'%(acc))

            if acc >= opt.best_acc:
                if acc == opt.best_acc:
                    if loss < opt.best_acc_loss:
                        opt.best_acc_epoch = epoch
                        opt.best_acc_loss = loss
                else:                        
                    opt.best_acc_epoch = epoch
                    opt.best_acc_loss = loss
                opt.best_acc = acc

                save_checkpoint(trainer.model, opt, epoch+1, True)
    
    if 0 == opt.local_rank:
        logging.info('Training Over!')


def train_one_epoch(opt, epoch, trainer, data_loader):

    loss_meter = AverageMeter()

    for iter, data in enumerate(data_loader):
        opt.cur_step = iter + epoch*len(data_loader)

        loss, n = trainer.step(data, opt)

        torch.cuda.synchronize()

        loss_meter.update(loss, n)

        if 0 == (iter+1)%opt.print_interval and 0 == opt.local_rank:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logging.info(
                f'Train: [{epoch+1}/{opt.num_epochs}][{iter+1}/{len(data_loader)}]\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')


@torch.no_grad()
def validate(opt, trainer, data_loader):
    torch.cuda.empty_cache()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    for iter, data in enumerate(data_loader):
        acc, loss, n = trainer.pred(data, opt)
        # acc = accuracy()
        acc_meter.update(acc, n)
        loss_meter.update(loss, n)

        if 0 == (iter+1)%opt.print_interval and 0 == opt.local_rank:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logging.info(
                f'Test: [{iter+1}/{len(data_loader)}]\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    return acc_meter.avg, loss_meter.avg


if __name__=='__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    opt = get_config()
    opt.local_rank = local_rank
    if 0 == local_rank:
        import time
        saved_path = os.path.join(opt.saved_path, time.strftime('%m-%d-%H:%M', time.localtime()))
        os.makedirs(saved_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(saved_path, 'running.log'),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO
        )
        os.system('cp config.py %s'%(saved_path))
        logging.info('Training config saved to %s'%(saved_path))
        opt.saved_path = saved_path

    dist.init_process_group(backend='nccl')
    # dist.barrier()
    torch.cuda.set_device(local_rank)

    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = opt.lr * opt.batch_size * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = opt.warmup_lr * opt.batch_size * dist.get_world_size() / 512.0
    linear_scaled_min_lr = opt.min_lr * opt.batch_size * dist.get_world_size() / 512.0

    opt.lr = linear_scaled_lr
    opt.warmup_lr = linear_scaled_warmup_lr
    opt.min_lr = linear_scaled_min_lr

    train(opt)
