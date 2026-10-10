"""Cross-segment associative recall: a direct test of whether memory works.

Each document has ``gap_segments + 2`` segments of ``segment_len`` tokens:

- **study**: ``pairs`` key/value pairs ``k1 v1 k2 v2 ...`` at a random offset, in filler;
- **gap**: filler only, competing for memory slots;
- **test**: queries ``Q k v`` for keys from the study segment, followed by filler.

The base model sees one segment at a time, so without memory a value in the test
segment is unpredictable (chance is ``1 / n_values``). The scored targets are the
value tokens of the test segment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from engramlm.model import EngramLM
from engramlm.segment import evaluate_document

PAD, QUERY = 0, 1


@dataclass(frozen=True)
class RecallTask:
    n_keys: int = 32
    n_values: int = 32
    n_filler: int = 32
    pairs: int = 8
    queries: int = 8
    segment_len: int = 32
    gap_segments: int = 2

    def __post_init__(self) -> None:
        if 2 * self.pairs > self.segment_len:
            raise ValueError("study pairs do not fit in one segment")
        if 3 * self.queries > self.segment_len:
            raise ValueError("queries do not fit in one segment")
        if self.queries > self.pairs or self.pairs > self.n_keys:
            raise ValueError("need queries <= pairs <= n_keys")

    @property
    def key_offset(self) -> int:
        return 2

    @property
    def value_offset(self) -> int:
        return self.key_offset + self.n_keys

    @property
    def filler_offset(self) -> int:
        return self.value_offset + self.n_values

    @property
    def vocab_size(self) -> int:
        return self.filler_offset + self.n_filler

    @property
    def num_segments(self) -> int:
        return self.gap_segments + 2

    @property
    def doc_len(self) -> int:
        return self.segment_len * self.num_segments

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def sample(self, batch_size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """Return ``(input_ids, target_mask)``, both ``(B, doc_len)``."""
        seg = self.segment_len
        ids = self.filler_offset + torch.randint(
            0, self.n_filler, (batch_size, self.doc_len), generator=generator
        )
        target = torch.zeros(batch_size, self.doc_len, dtype=torch.bool)
        test_start = seg * (self.num_segments - 1)
        for b in range(batch_size):
            keys = torch.randperm(self.n_keys, generator=generator)[: self.pairs]
            values = torch.randint(0, self.n_values, (self.pairs,), generator=generator)
            # Random offset so no scorer is favoured by where the pairs sit in the segment.
            offset = int(torch.randint(0, seg - 2 * self.pairs + 1, (1,), generator=generator))
            ids[b, offset : offset + 2 * self.pairs : 2] = self.key_offset + keys
            ids[b, offset + 1 : offset + 2 * self.pairs : 2] = self.value_offset + values
            asked = torch.randperm(self.pairs, generator=generator)[: self.queries]
            for q, pair in enumerate(asked.tolist()):
                pos = test_start + 3 * q
                ids[b, pos] = QUERY
                ids[b, pos + 1] = self.key_offset + keys[pair]
                ids[b, pos + 2] = self.value_offset + values[pair]
                target[b, pos + 2] = True
        return ids, target


@dataclass(frozen=True)
class RecallResult:
    accuracy: float
    target_nll: float
    n_targets: int


@torch.no_grad()
def evaluate_recall(
    model: EngramLM,
    task: RecallTask,
    *,
    n_docs: int = 512,
    batch_size: int = 64,
    seed: int = 12_345,
) -> RecallResult:
    """Accuracy and mean NLL on the value tokens of the test segment."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    device = model.device
    correct, total, nll_sum = 0, 0, 0.0
    test_start = task.segment_len * (task.num_segments - 1)
    for start in range(0, n_docs, batch_size):
        batch = min(batch_size, n_docs - start)
        ids, target = task.sample(batch, gen)
        ids, target = ids.to(device), target.to(device)
        result = evaluate_document(
            model, ids, segment_len=task.segment_len, score_mask=target, keep_logits=True
        )
        nll_sum += result.nll_sum
        test_logits = result.logits[-1].to(device)
        test_target = target[:, test_start:]
        test_ids = ids[:, test_start:]
        predicted = test_logits[:, :-1].argmax(dim=-1)
        hits = (predicted == test_ids[:, 1:]) & test_target[:, 1:]
        correct += int(hits.sum())
        total += int(test_target[:, 1:].sum())
    return RecallResult(
        accuracy=correct / max(total, 1), target_nll=nll_sum / max(total, 1), n_targets=total
    )
