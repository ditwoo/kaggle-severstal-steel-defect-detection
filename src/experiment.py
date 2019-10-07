from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from .utils import get_dataset


class Experiment(ConfigExperiment):
    @staticmethod
    def get_transfors(state: str = None, mode: str = None):
        pass

    def get_datasets(
        self, stage: str, train: dict = None, validation: dict = None, **kwargs
    ):
        if stage == "stageFinal":
            train["transforms"] = validation["transforms"]
        datasets = OrderedDict()
        datasets["train"] = get_dataset(**train)
        datasets["valid"] = get_dataset(**validation)

        return datasets
