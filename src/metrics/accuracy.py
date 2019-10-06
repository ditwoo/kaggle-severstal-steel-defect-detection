from typing import Tuple

import numpy as np
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from numba import njit, prange


@njit(parallel=True)
def _fast_accs(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    size: int = y_true.shape[0]
    overall_cnt: int = 0
    positive_cnt: int = 0
    negative_cnt: int = 0
    overall_total: int = size
    positive_total: int = 0
    negative_total: int = 0
    for i in prange(size):
        _is_matched = int(y_pred[i] == y_true[i])
        overall_cnt += _is_matched
        if y_true[i] == 1:
            positive_cnt += _is_matched
            positive_total += 1
        if y_true[i] == 0:
            negative_cnt += _is_matched
            negative_total += 1
    overall: float = overall_cnt / overall_total
    positive: float = positive_cnt / positive_total if positive_total != 0 else 0
    negative: float = negative_cnt / negative_total if negative_total != 0 else 0
    return overall, positive, negative


class AllAccuracyCallback(Callback):
    def __init__(
        self,
        prefix: str = "accuracy",
        input_key: str = "targets",
        output_key: str = "logits",
        threshold: float = 0.5,
        smooth: float = 1e-8,
        **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.threshold: float = threshold
        self.smooth: float = smooth
        self.metric_params = metric_params
        self._names = ("all", "positive", "negative")

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = state.output[self.output_key].sigmoid()
        outputs = outputs.detach().cpu().view(-1).numpy()
        outputs = (outputs > self.threshold).astype(np.int8)

        targets = state.input[self.input_key]
        targets = targets.detach().cpu().view(-1).numpy().astype(np.int8)

        accs = _fast_accs(outputs, targets)

        state.metrics.add_batch_value(
            metrics_dict={
                f"{self.prefix}_{name}": value for name, value in zip(self._names, accs)
            }
        )


__all__ = ("AllAccuracyCallback",)


def test():
    from sklearn.metrics import accuracy_score

    size = np.random.randint(0, 1000)

    _pred = np.random.randint(0, 2, size)
    _true = np.random.randint(0, 2, size)

    correct = _fast_accs(_pred, _true)
    predicted_all = accuracy_score(_true, _pred)
    predicted_positive = accuracy_score(_true[_true == 1], _pred[_true == 1])
    predicted_negative = accuracy_score(_true[_true == 0], _pred[_true == 0])

    for first, second, name in zip(
        (predicted_all, predicted_positive, predicted_negative),
        correct,
        ("all", "positive", "negative"),
    ):
        print(f"Type: {name}")
        print(f" Metric value: - {first}", flush=True)
        print(f"Correct value: - {second}", flush=True)
        assert first == second, "Metrics should be the same!"


if __name__ == "__main__":
    test()
