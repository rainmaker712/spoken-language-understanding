import torch
from torch.optim.lr_scheduler import LambdaLR

from optimizer.optimization import AdamW
from optimizer.scheduler import LinearWarmUpScheduler


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, global_step=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.global_step = global_step
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        step += self.global_step
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(config, model, num_train_steps, logger, global_step=0,
                    use_scheduler=True, local_rank=0):

    if config.fp16:
        param_optimizer = list([(n, p) for n, p in model.named_parameters() if 'pooler' not in n])
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if config.fp16:
        try:
            from apex import amp
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            # from apex.optimizers import FusedLAMB
        except ImportError:
            raise ImportError("Please install apex")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=config.learning_rate,
                              bias_correction=False,
                              betas=(config.adam_beta_1, config.adam_beta_2),
                              eps=config.adam_epsilon)

        if config.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=config.loss_scale)

        last_epoch = -1 if global_step == 0 else global_step
        scheduler = LinearWarmUpScheduler(optimizer, warmup=config.warmup_proportion,
                                          total_steps=num_train_steps,
                                          last_epoch=last_epoch)

        if local_rank == 0:
            logger.info("*** Using FusedAdam for fp16 training***")
        return optimizer, model, scheduler

    else:
        if use_scheduler:
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
            last_epoch = -1 if global_step == 0 else global_step
            scheduler = LinearWarmUpScheduler(optimizer, warmup=config.warmup_proportion,
                                              total_steps=num_train_steps,
                                              last_epoch=last_epoch)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            scheduler = None

        return optimizer, scheduler

