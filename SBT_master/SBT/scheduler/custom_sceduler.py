# custom_scheduler.py
import math
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        steps_per_epoch,
        start_lr,
        base_lr, # Added base_lr
        epochs,
        warmup_epochs=10,
        last_epoch=-1,
        final_lr=0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.base_lr = base_lr  # Store the base LR
        self.final_lr = final_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.T_max = self.total_steps - self.warmup_steps  # Cosine annealing steps (after warmup)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:  # Iterate through base learning rates (for each param group)
            if self._step_count < self.warmup_steps:
                progress = float(self._step_count) / float(max(1, self.warmup_steps))
                new_lr = self.start_lr + progress * (self.base_lr - self.start_lr)  # Use self.base_lr
            else:
                # -- progress after warmup
                progress = float(self._step_count - self.warmup_steps) / float(
                    max(1, self.T_max)
                )
                new_lr = max(
                    self.final_lr,
                    self.final_lr
                    + (self.base_lr - self.final_lr)  # Use self.base_lr
                    * 0.5
                    * (1.0 + math.cos(math.pi * progress)),
                )
            lrs.append(new_lr)
        return lrs

class CosineWDSchedule(object):
    def __init__(self, optimizer, epochs, steps_per_epoch, init_weight_decay, final_weight_decay=0.4):
        self.optimizer = optimizer
        self.init_weight_decay = init_weight_decay
        self.final_weight_decay = final_weight_decay
        self.total_steps = epochs * steps_per_epoch
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.total_steps
        new_wd = self.final_weight_decay + (
            self.init_weight_decay - self.final_weight_decay
        ) * 0.5 * (1.0 + math.cos(math.pi * progress))

        # Ensure new_wd doesn't go below 0 or exceed initial.
        new_wd = max(0.0, min(self.init_weight_decay, new_wd))

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd