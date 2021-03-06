from torch import optim


# Implement your parameters settings here
# Set different learning rates and weight decays for parameters in the model according to the config
# Return the list of parameter dicts
def _set_params(model, opt):
    if 'finetune' == opt.train_stage:
        return model.parameters()
    
    pass


def build_optimizer(opt, model):
    params = _set_params(model, opt)

    if opt.optim == 'sgd':
        optimizer = optim.SGD(
            params=params,
            momentum=opt.momentum,
            nesterov=True,
            lr=opt.lr,
            weight_decay=opt.weight_decay)
    elif opt.optim == 'adamw':
        optimizer = optim.AdamW(
            params=params,
            eps=opt.eps,
            betas=opt.betas,
            lr=opt.lr,
            weight_decay=opt.weight_decay)
    
    return optimizer
