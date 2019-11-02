from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from .utils import get_dataset
from .datasets import BalancedSampler


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
        train_dataset = get_dataset(**train) 
        datasets["train"] = {
            "dataset": train_dataset,
            # "sampler": BalancedSampler(train_dataset),
            "shuffle": True,
        }
        datasets["valid"] = {
            "dataset": get_dataset(**validation),
            "shuffle": False,
        }

        return datasets
