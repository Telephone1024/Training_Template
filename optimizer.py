from torch import optim

def build_optimizer(opt, model):
    params = set_params(model, opt)

    if opt.optim == 'sgd':
        optimizer = optim.SGD(
            params=params,
            momentum=opt.momentun,
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


# Implement your parameters setting here
# Set different learning rates and weight decays for parameters in the model according to the config
# Return the list of parameter dicts
def set_params(model, opt):
    pass