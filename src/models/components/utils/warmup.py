# Pytorch learning rate sceduler for warmup
import warnings

from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        fract,
        total_step,
        min_lr,
        max_lr,
        last_epoch=-1,
        verbose=False,
    ):
        self.number_step = fract * total_step
        self.min_lr = min_lr
        self.max_lr = max_lr
        if len(optimizer.param_groups) > 1:
            warnings.warn(
                "optimizer contains more than one parameter group, the warup scheduler will treat all of them as a single group."
            )
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < self.number_step:
            return [
                self.min_lr
                + (self.max_lr - self.min_lr)
                * self.last_epoch
                / self.number_step
                for group in self.optimizer.param_groups
            ]
        elif self.last_epoch == self.number_step:
            return [self.max_lr for group in self.optimizer.param_groups]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.number_step:
            return [
                self.min_lr
                + (self.max_lr - self.min_lr)
                * self.last_epoch
                / self.number_step
                for group in self.optimizer.param_groups
            ]
        else:
            return [self.max_lr for group in self.optimizer.param_groups]
