from random import randint

from torch.utils.data import Dataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self.dataset[randint(0, len(self.dataset) - 1)]
