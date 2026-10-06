import math
import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBatchSampler(Sampler[list[int]]):
    """Batches of indices that share a length, in a shuffled order.

    Each batch holds a single length, so the default collate can stack it, while the order of the batches
    mixes lengths, so training does not see all the short records first and all the long ones last.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 0):
        self.batch_size = batch_size
        self.groups: dict[int, list[int]] = defaultdict(list)

        for index, length in enumerate(lengths):
            self.groups[length].append(index)

        # One generator for the whole run: each epoch gets a new order, and each run the same sequence of orders.
        # Lightning does not call set_epoch on a custom batch sampler, so the epoch can't be used as a seed.
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[list[int]]:
        batches = []

        for indices in self.groups.values():
            indices = self._rng.sample(indices, len(indices))
            batches += [indices[i: i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        self._rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())
