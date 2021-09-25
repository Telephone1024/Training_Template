import logging
import os
import torch
import torch.distributed as dist
   

def save_checkpoint(model, opt, epoch, is_best=False):
    if hasattr(model, 'module'):
        params = model.module.state_dict()
    else:
        params = model.state_dict()
    logging.info('saving to %s'%(opt.saved_path))
    if is_best:
        torch.save(params, os.path.join(opt.saved_path, 'ckpt_best.pth'))
        logging.warning('BEST MODEL IS SAVED!!! CURRENT ACCURACY IS: %.5f'%(opt.best_acc))
    else:
        torch.save(params, os.path.join(opt.saved_path, 'ckpt_epoch_%03d.pth'%(epoch)))
        logging.info('model is saved!')


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def lr_adjust(opt):
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = opt.lr * opt.batch_size * dist.get_world_size() / opt.total_batch_size
    linear_scaled_warmup_lr = opt.warmup_lr * opt.batch_size * dist.get_world_size() / opt.total_batch_size
    linear_scaled_min_lr = opt.min_lr * opt.batch_size * dist.get_world_size() / opt.total_batch_size

    opt.lr = linear_scaled_lr
    opt.warmup_lr = linear_scaled_warmup_lr
    opt.min_lr = linear_scaled_min_lr

    return opt