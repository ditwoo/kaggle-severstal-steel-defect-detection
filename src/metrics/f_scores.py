from typing import Tuple

import numpy as np
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from numba import njit, prange


@njit(parallel=True)
def _fast_f_score_parts(y_pred: np.ndarray, 
                        y_true: np.ndarray) -> Tuple[int, int, int, int]:
    size: int = y_true.shape[0]
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    for i in prange(size):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
    return tp, tn, fp, fn


def f_betta(outputs: np.ndarray,
            targets: np.ndarray, 
            beta: float = 1.0) -> float:
    """
    Compute f_beta score.

    :param outputs: predicted classes (binary)
    :param targets: target classes (binary)
    :param beta: beta coefficient
    :param smooth: smoothing value used to avoid zero division
    """
    tp, _, fp, fn = _fast_f_score_parts(outputs, targets)

    precision: float = tp / (tp + fp) if tp + fp > 0 else 0.
    recall: float = tp / (tp + fn) if tp + fn > 0 else 0.

    b2: float = beta * beta
    numerator: float = (1 + b2) * precision * recall
    denominator: float = (b2 * precision) + recall
    f: float = numerator / denominator if denominator > 0 else 0.
    
    return f


class F1Callback(Callback):
    def __init__(self,
                 prefix: str = 'f1',
                 input_key: str = 'targets',
                 output_key: str = 'logits',
                 threshold: float = 0.5,
                 **metric_params):
        super().__init__(CallbackOrder.Metric)
        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.threshold: float = threshold
        self.metric_params = metric_params

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = state.output[self.output_key].sigmoid()
        outputs = outputs.detach().cpu().view(-1).numpy()
        outputs = (outputs > self.threshold).astype(np.int8)

        targets = state.input[self.input_key]
        targets = targets.detach().cpu().view(-1).numpy().astype(np.int8)

        state.metrics.add_batch_value(metrics_dict={
            f'{self.prefix}': f_betta(outputs, targets, 1)
        })


class FBetaCallback(Callback):
    def __init__(self,
                 prefix: str = 'f_betta',
                 input_key: str = 'targets',
                 output_key: str = 'logits',
                 threshold: float = 0.5,
                 beta: float = 1.0,
                 **metric_params):
        """
        F_betta score callback.

        * To give more weight to Precision beta should be - `0 < beta < 1`
        * To give more weight to Recall, beta should be - `1 < beta < +inf`

        If `beta == 1` then will be computed simple F1 score.
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.threshold: float = threshold
        self.beta: float = beta
        self.metric_params = metric_params

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = state.output[self.output_key].sigmoid()
        outputs = outputs.detach().cpu().view(-1).numpy()
        outputs = (outputs > self.threshold).astype(np.int8)

        targets = state.input[self.input_key]
        targets = targets.detach().cpu().view(-1).numpy().astype(np.int8)

        state.metrics.add_batch_value(metrics_dict={
            f'{self.prefix}': f_betta(outputs, targets, self.beta)
        })


__all__ = ('F1Callback', 'FBetaCallback')


def test_f1():
    from sklearn.metrics import f1_score

    size = np.random.randint(0, 1000)

    _pred = np.random.randint(0, 2, size)
    _true = np.random.randint(0, 2, size)

    tp, _, fp, fn = _fast_f_score_parts(_pred, _true)
    precision: float = tp / (tp + fp)
    recall: float = tp / (tp + fn)

    fast_metric_score = 2 * precision * recall / (precision + recall)
    correct_score = f1_score(_true, _pred)

    print(f' Metric value: - {fast_metric_score}', flush=True)
    print(f'Correct value: - {correct_score}', flush=True)
    assert fast_metric_score == correct_score, 'Metrics should be the same!'


def test_fbeta():
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    size = np.random.randint(0, 1000)
    beta = 0.5

    _pred = np.random.randint(0, 2, size)
    _true = np.random.randint(0, 2, size)

    tp, _, fp, fn = _fast_f_score_parts(_pred, _true)
    precision: float = tp / (tp + fp)
    recall: float = tp / (tp + fn)
    beta2 = beta ** 2

    assert precision_score(_true, _pred) == precision
    assert recall_score(_true, _pred) == recall

    fast_metric_score = (1 + beta2) * precision * recall / ((beta2 * precision) + recall)
    correct_score = fbeta_score(_true, _pred, beta)

    print(f' Metric value: - {fast_metric_score}', flush=True)
    print(f'Correct value: - {correct_score}', flush=True)
    assert fast_metric_score == correct_score, 'Metrics should be the same!'


if __name__ == '__main__':
    test_f1()
    test_fbeta()
