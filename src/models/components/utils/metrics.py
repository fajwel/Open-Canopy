import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


class Masked_MAE(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        assert mask.shape[1] == 1 and mask.dim() == preds.dim()

        masked_preds = torch.where(mask, torch.zeros_like(preds), preds)
        masked_target = torch.where(mask, torch.zeros_like(target), target)
        sum_abs_error = torch.abs(masked_preds - masked_target).sum()
        num_obs = (~mask).sum() * masked_preds.shape[1]

        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return self.sum_abs_error / self.total


class Masked_RMSE(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_square_error", default=tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        assert mask.shape[1] == 1 and mask.dim() == preds.dim()

        masked_preds = torch.where(mask, torch.zeros_like(preds), preds)
        masked_target = torch.where(mask, torch.zeros_like(target), target)
        sum_square_error = torch.pow(masked_preds - masked_target, 2).sum()
        num_obs = (~mask).sum() * masked_preds.shape[1]

        self.sum_square_error += sum_square_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return torch.sqrt(self.sum_square_error / self.total)


class Masked_nMAE(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state(
            "sum_norm_abs_error", default=tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor) -> None:
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        assert mask.shape[1] == 1 and mask.dim() == preds.dim()

        masked_preds = torch.where(mask, torch.zeros_like(preds), preds)
        masked_target = torch.where(mask, torch.zeros_like(target), target)
        sum_norm_abs_error = (
            torch.abs(masked_preds - masked_target)
            / (torch.abs(masked_target) + 0.01)
        ).sum()
        num_obs = (~mask).sum() * masked_preds.shape[1]

        self.sum_norm_abs_error += sum_norm_abs_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return self.sum_norm_abs_error / self.total
