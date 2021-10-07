import logging
import os
import functools
import torch
import torch.distributed as dist

from termcolor import colored
   

def save_checkpoint(model, opt, epoch, is_best=False, stage='eval'):
    if hasattr(model, 'module'):
        params = model.module.state_dict()
    else:
        params = model.state_dict()
    logging.info('saving to %s'%(opt.saved_path))
    if is_best:
        torch.save(params, os.path.join(opt.saved_path, 'ckpt_%s_best.pth'%(stage)))
        logging.warning('BEST MODEL IS SAVED!!! CURRENT %s ACCURACY IS: %.5f, EPOCH: %03d'%(            
            (stage.upper(), opt.best_eval_acc, opt.best_eval_acc_epoch) if 'eval' == stage else (stage.upper(), opt.best_test_acc, opt.best_test_acc_epoch)
            ))
    else:
        torch.save(params, os.path.join(opt.saved_path, 'ckpt_epoch_%03d.pth'%(epoch)))
        logging.info('model is saved!')


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
def build_logger(opt):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    format = colored('%(asctime)s', 'green') + \
             colored('[%(filename)s %(lineno)d]', 'yellow') +\
             ': %(levelname)s %(message)s'

    con_handler = logging.StreamHandler()
    con_handler.setLevel(logging.DEBUG)
    con_handler.setFormatter(
        logging.Formatter(fmt=format, datefmt='%Y-%m-%d %H:%M:%S')
    )

    file_handler = logging.FileHandler(os.path.join(opt.saved_path, 'run.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=format, datefmt='%Y-%m-%d %H:%M:%S')
    )

    logger.addHandler(con_handler)
    logger.addHandler(file_handler)
    
    return logger