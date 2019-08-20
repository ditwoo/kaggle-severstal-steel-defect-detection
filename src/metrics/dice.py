import numpy as np
import torch
from catalyst.dl.core import Callback, RunnerState


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def np_dice(prediction: np.ndarray,
            ground_truth: np.ndarray, 
            from_logits: bool = True,
            smooth: float = 1e-5,
            dim=(2, 3)) -> np.ndarray:
    """
    Input shapes:
        pred: BxCxHxW
        pred: BxCxHxW
    """

    if from_logits:
        prediction = sigmoid(prediction)

    dim = (2, 3)
    intersection = np.sum(prediction * ground_truth, dim)
    union = np.sum(prediction, dim) + np.sum(ground_truth, dim) + smooth

    return np.mean(2 * intersection / union, 0)


class ChannelviseDiceMetricCallback(Callback):
    def __init__(self, 
                 prefix: str = 'dice', 
                 input_key: str = 'targets',
                 output_key: str = 'logits',
                 **metric_params):
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
    
    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        _dice = np_dice(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        _metrics = {f'{self.prefix}_{num}': val for num, val in enumerate(_dice)}
        
        state.metrics.add_batch_value(metrics_dict=_metrics)

__all__ = ['ChannelviseDiceMetricCallback']
