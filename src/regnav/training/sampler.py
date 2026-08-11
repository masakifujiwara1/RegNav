from collections import Counter
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class DomainBalancedSampler(Sampler[int]):
    def __init__(self, labels: Sequence[str], num_samples: int, seed: int):
        if not labels or num_samples <= 0:
            raise ValueError("labels and num_samples must be non-empty")
        counts = Counter(labels)
        self.weights = torch.tensor([1 / counts[label] for label in labels])
        self.num_samples = num_samples
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        return iter(
            torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
                generator=self.generator,
            ).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples
