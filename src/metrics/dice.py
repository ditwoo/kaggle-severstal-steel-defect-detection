import numpy as np
import torch
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from catalyst.dl.utils import dice


class PositiveAndNegativeDiceMetricCallback(Callback):
    def __init__(self, 
                 prefix: str = 'dice', 
                 input_key: str = 'targets',
                 output_key: str = 'logits',
                 **metric_params):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.threshold = metric_params.get('threshold', 0.5)
        self.activation = metric_params.get('activation', 'Sigmoid')
    
    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        if self.activation == 'Sigmoid':
            outputs = outputs.sigmoid()
        targets = state.input[self.input_key]

        _positive_dice = dice((outputs >= self.threshold).float(), (targets > 0).float(), activation=self.activation)
        _negative_dice = dice((outputs < self.threshold).float(), (targets == 0).float(), activation=self.activation)
        state.metrics.add_batch_value(metrics_dict={
            'positive_dice': _positive_dice,
            'negative_dice': _negative_dice
        })

__all__ = ['PositiveAndNegativeDiceMetricCallback']
