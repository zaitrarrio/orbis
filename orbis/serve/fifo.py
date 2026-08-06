"""Serve co-design: bounded FIFO progressive delivery."""

from __future__ import annotations

from collections import deque
from typing import Deque, Generic, Iterator, Optional, TypeVar

T = TypeVar("T")


class BoundedFIFO(Generic[T]):
    """Fixed-capacity FIFO for progressive chunk delivery.

    When full, ``push`` drops the oldest undelivered item (backpressure) unless
    ``block_on_full`` is set — matching the paper's bounded-queue progressive
    decode path at toy/serve scale.
    """

    def __init__(self, capacity: int = 4, block_on_full: bool = False):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.block_on_full = block_on_full
        self._q: Deque[T] = deque()
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._q)

    @property
    def full(self) -> bool:
        return len(self._q) >= self.capacity

    def push(self, item: T) -> bool:
        """Enqueue ``item``. Returns False if an older item was dropped."""
        dropped = False
        if self.full:
            if self.block_on_full:
                raise RuntimeError("BoundedFIFO is full")
            self._q.popleft()
            self.dropped += 1
            dropped = True
        self._q.append(item)
        return not dropped

    def pop(self) -> Optional[T]:
        if not self._q:
            return None
        return self._q.popleft()

    def peek(self) -> Optional[T]:
        return self._q[0] if self._q else None

    def clear(self) -> None:
        self._q.clear()

    def drain(self) -> Iterator[T]:
        while self._q:
            yield self._q.popleft()
