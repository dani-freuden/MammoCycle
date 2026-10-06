import torch
from torch.utils.data import DataLoader

from src.data.sampler import LengthBatchSampler


# 5 items of length 2, 3 of length 3, 1 of length 4, interleaved.
LENGTHS = [2, 3, 2, 2, 3, 4, 2, 3, 2]


def test_each_batch_has_a_single_length():
    for batch in LengthBatchSampler(LENGTHS, batch_size=2):
        assert len({LENGTHS[i] for i in batch}) == 1


def test_every_index_appears_once_per_epoch():
    sampler = LengthBatchSampler(LENGTHS, batch_size=2)

    for _ in range(3):
        indices = [i for batch in sampler for i in batch]
        assert sorted(indices) == list(range(len(LENGTHS)))


def test_only_the_last_batch_of_each_length_is_smaller():
    sizes = {}

    for batch in LengthBatchSampler(LENGTHS, batch_size=2):
        sizes.setdefault(LENGTHS[batch[0]], []).append(len(batch))

    assert {length: sorted(s) for length, s in sizes.items()} == {2: [1, 2, 2], 3: [1, 2], 4: [1]}


def test_len_is_the_number_of_batches():
    sampler = LengthBatchSampler(LENGTHS, batch_size=2)

    assert len(sampler) == len(list(sampler)) == 6


def test_same_seed_gives_the_same_epochs():
    a = LengthBatchSampler(LENGTHS, batch_size=2, seed=7)
    b = LengthBatchSampler(LENGTHS, batch_size=2, seed=7)

    assert [list(a) for _ in range(3)] == [list(b) for _ in range(3)]


def test_each_epoch_has_a_new_order():
    sampler = LengthBatchSampler(list(range(2, 12)) * 10, batch_size=4)

    assert list(sampler) != list(sampler)


def test_batch_order_mixes_lengths():
    lengths = [2] * 100 + [3] * 100
    batch_lengths = [lengths[batch[0]] for batch in LengthBatchSampler(lengths, batch_size=4)]

    assert batch_lengths != sorted(batch_lengths)


def test_default_collate_stacks_every_batch():
    dataset = [torch.zeros(length) for length in LENGTHS]
    loader = DataLoader(dataset, batch_sampler=LengthBatchSampler(LENGTHS, batch_size=2))

    assert sorted(batch.shape[1] for batch in loader) == [2, 2, 2, 3, 3, 4]
