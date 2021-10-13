from timm import scheduler as sche

def build_scheduler(opt, optimizer, iters_per_epoch):
    scheduler = None
    if 'cosine' == opt.sche_type:
        scheduler = sche.CosineLRScheduler(
            optimizer=optimizer,
            t_initial=opt.num_epochs * iters_per_epoch,
            t_mul=1.,
            lr_min=opt.min_lr,
            warmup_lr_init=opt.warmup_lr,
            warmup_t=opt.num_warmup_epochs * iters_per_epoch,
            cycle_limit=1,
            t_in_epochs=False
        )
    elif 'step' == opt.sche_type:
        scheduler = sche.StepLRScheduler(
            optimizer=optimizer,
            decay_t=opt.decay_epochs * iters_per_epoch,
            decay_rate=opt.decay_rate,
            warmup_lr_init=opt.warmup_lr,
            warmup_t=opt.num_warmup_epochs * iters_per_epoch,
            t_in_epochs=False
        )
    elif 'plateau' == opt.sche_type:
        scheduler = sche.PlateauLRScheduler(
            optimizer=optimizer,
            decay_rate=opt.decay_rate,
            patience_t=opt.patience,
            warmup_t=opt.num_warmup_epochs * iters_per_epoch,
            warmup_lr_init=opt.warmup_lr,
            lr_min=opt.min_lr,
            mode='min',
        )

    return scheduler
