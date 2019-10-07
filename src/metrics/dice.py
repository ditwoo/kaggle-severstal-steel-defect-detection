import numpy as np
import torch
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from catalyst.dl.utils import dice


def dice_channel_torch(
    probability: torch.Tensor, truth: torch.Tensor, threshold: float
) -> float:
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.0
    with torch.no_grad():
        for j in range(channel_num):
            channel_dice = dice_single_channel(
                probability[:, j, :, :], truth[:, j, :, :], threshold, batch_size
            )
            mean_dice_channel += channel_dice.sum(0) / (batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(
    probability: torch.Tensor,
    truth: torch.Tensor,
    threshold: float,
    batch_size: int,
    eps: float = 1e-9,
) -> float:
    p = (probability.view(batch_size, -1) > threshold).float()
    t = (truth.view(batch_size, -1) > 0.5).float()
    dice = (2.0 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return dice


class MeanDiceCallback(Callback):
    def __init__(
        self,
        prefix: str = "mean_dice",
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.threshold = metric_params.get("threshold", 0.5)
        self.activation = metric_params.get("activation", "Sigmoid")

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        if self.activation == "Sigmoid":
            outputs = outputs.sigmoid()
        targets = state.input[self.input_key]

        mean_dice = dice_channel_torch(outputs, targets, self.threshold)

        state.metrics.add_batch_value(metrics_dict={self.prefix: mean_dice})


__all__ = ("MeanDiceCallback",)
