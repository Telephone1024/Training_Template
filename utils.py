import logging
import os
import functools
import torch
import torch.distributed as dist

from termcolor import colored
   

def save_checkpoint(model, opt, epoch, is_best=False, stage='val'):
    if hasattr(model, 'module'):
        params = model.module.state_dict()
    else:
        params = model.state_dict()
    opt.logger.info('saving to %s'%(opt.saved_path))
    if is_best:
        torch.save(params, os.path.join(opt.saved_path, 'ckpt_%s_best.pth'%(stage)))
        opt.logger.warning('BEST MODEL IS SAVED!!! CURRENT %s ACCURACY IS: %.5f, EPOCH: %03d'%(            
            (stage.upper(), opt.best_val_acc, opt.best_val_acc_epoch+1) if 'val' == stage else (stage.upper(), opt.best_test_acc, opt.best_test_acc_epoch+1)
            ))
    else:
        torch.save(params, os.path.join(opt.saved_path, 'ckpt_epoch_%03d.pth'%(epoch)))
        opt.logger.info('model is saved!')


def lr_adjust(opt):
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = opt.lr * opt.batch_size * dist.get_world_size() / opt.total_batch_size
    linear_scaled_warmup_lr = opt.warmup_lr * opt.batch_size * dist.get_world_size() / opt.total_batch_size
    linear_scaled_min_lr = opt.min_lr * opt.batch_size * dist.get_world_size() / opt.total_batch_size

    opt.lr = linear_scaled_lr
    opt.warmup_lr = linear_scaled_warmup_lr
    opt.min_lr = linear_scaled_min_lr

    return opt


@functools.lru_cache()
def build_logger(saved_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    format = '%(asctime)s %(levelname)s [%(filename)s %(lineno)d]\t:\t%(message)s'
    colored_format = colored('%(asctime)s ', 'green') + '%(levelname)s' + \
             colored('[%(filename)s %(lineno)d]\t', 'yellow') +\
             ': \t%(message)s'

    con_handler = logging.StreamHandler()
    con_handler.setLevel(logging.INFO)
    con_handler.setFormatter(
        logging.Formatter(fmt=colored_format, datefmt='%Y-%m-%d %H:%M:%S')
    )

    file_handler = logging.FileHandler(os.path.join(saved_path, 'run.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(fmt=format, datefmt='%Y-%m-%d %H:%M:%S')
    )

    logger.addHandler(con_handler)
    logger.addHandler(file_handler)
    
    return logger