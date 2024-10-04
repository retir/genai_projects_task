from torch.utils.data import DataLoader


class InfiniteLoader(DataLoader):
    def __init__(
        self,
        *args,
        num_workers=0,
        **kwargs,
    ):
        super().__init__(
            *args,
            multiprocessing_context="fork" if num_workers > 0 else None,
            num_workers=num_workers,
            **kwargs,
        )
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            x = next(self.dataset_iterator)

        return x