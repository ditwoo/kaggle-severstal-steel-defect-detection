from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from catalyst.dl.core import Callback, CallbackOrder, RunnerState


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, num_classes):
    # a long 2xn array with each column being a pixel pair
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten())).T

    # add up confusion matrix
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes), (0, num_classes)],
    )
    confusion_matrix = confusion_matrix.astype(np.uint64)
    return confusion_matrix


def get_confusion_matrix(y_pred_logits: torch.Tensor, y_true: torch.Tensor):
    num_classes = y_pred_logits.shape[1]
    y_pred = torch.argmax(y_pred_logits, dim=1)
    ground_truth = y_true.cpu().numpy()
    prediction = y_pred.cpu().numpy()

    return calculate_confusion_matrix_from_arrays(prediction, ground_truth, num_classes)


def calculate_tp_fp_fn(confusion_matrix):
    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for index in range(confusion_matrix.shape[0]):
        true_positives[index] = confusion_matrix[index, index]
        false_positives[index] = (
            confusion_matrix[:, index].sum() - true_positives[index]
        )
        false_negatives[index] = (
            confusion_matrix[index, :].sum() - true_positives[index]
        )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_dice(tp_fp_fn_dict):
    epsilon = 1e-7

    dice = {}

    for i in range(len(tp_fp_fn_dict["true_positives"])):
        tp = tp_fp_fn_dict["true_positives"][i]
        fp = tp_fp_fn_dict["false_positives"][i]
        fn = tp_fp_fn_dict["true_positives"][i]

        dice[i] = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

        assert 0 <= dice[i] <= 1

    return dice


class MulticlassDiceMetricCallback(Callback):
    """
    A callback that returns dictionary
    """

    def __init__(
        self,
        prefix: str = "dice",
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.confusion_matrix = None
        self.class_names: OrderedDict = OrderedDict(metric_params["class_names"])
        self.avg_classes: list = metric_params["avg_classes"]

    def _reset_stats(self):
        self.confusion_matrix = None

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        #         import pdb; pdb.set_trace()

        confusion_matrix = get_confusion_matrix(outputs, targets)

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

        tp_fp_fn_dict = calculate_tp_fp_fn(confusion_matrix)

        batch_metrics: Dict = {
            self.class_names[key]: value
            for key, value in calculate_dice(tp_fp_fn_dict).items()
            if key in self.class_names
        }
        batch_metrics["avg_dice"] = np.mean(
            [batch_metrics[cls_name] for cls_name in self.avg_classes]
        )

        state.metrics.add_batch_value(metrics_dict=batch_metrics)

    def on_loader_end(self, state: RunnerState):

        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        batch_metrics: Dict = calculate_dice(tp_fp_fn_dict)

        for metric_id, dice_value in batch_metrics.items():
            metric_name = self.class_names[metric_id]
            state.metrics.epoch_values[state.loader_name][metric_name] = dice_value

        self._reset_stats()


__all__ = ["MulticlassDiceMetricCallback"]
