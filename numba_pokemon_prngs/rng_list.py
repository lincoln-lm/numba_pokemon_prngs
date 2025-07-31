"""List of cached PRNG results to avoid needing to
re-initialize or go backwards to reuse previous rands"""

from __future__ import annotations
from typing import TypeVar, Callable, Type
from typing_extensions import Protocol
import numpy as np
from .options import USE_NUMBA
from .compilation import optional_jitclass, array_type

PRNG = TypeVar("PRNG")


class RNGList(Protocol):
    """List of cached PRNG results to avoid needing to
    re-initialize or go backwards to reuse previous rands"""

    def __init__(self, rng: PRNG) -> None: ...

    def re_init(self):
        """Reinitialize without creating a new object"""

    def advance_states(self, adv: np.uint32) -> None:
        """Advance the state of the RNGList by adv"""

    def advance_state(self) -> None:
        """Advance the state of the RNGList by one"""

    def advance(self, adv: np.uint32) -> None:
        """Advance within the state by adv"""

    def get_value(self):
        """Get value of the next rand"""


def build_rnglist(
    *, rng_class: PRNG, next_function_name: str, state_type: np.dtype, size: int
) -> Type[RNGList]:
    """Build RNGList class for rng_class of size using next_function_name"""
    assert size != 0 and (
        (size & (size - 1)) == 0
    ), "Size is not a perfect multiple of two"
    if USE_NUMBA:
        next_function: Callable[[PRNG], state_type] = rng_class.class_type.jit_methods[
            next_function_name
        ]
    else:
        next_function: Callable[[PRNG], state_type] = rng_class.__dict__[
            next_function_name
        ]

    @optional_jitclass(
        {
            "list": array_type(np.uint32),  # contiguous array
            "rng": rng_class.class_type.instance_type if USE_NUMBA else rng_class,
            "head": np.uint16,
            "pointer": np.uint16,
        }
    )
    class SpecificRNGList:
        """RNGList for {rng_class}"""

        def __init__(self, rng: PRNG) -> None:
            self.rng = rng
            self.list = np.empty(size, dtype=state_type)
            self.head = 0
            self.pointer = 0
            self.re_init()

        def re_init(self):
            """Reinitialize without creating a new object"""
            self.head = 0
            self.pointer = 0
            for i in range(size):
                self.list[i] = next_function(self.rng)

        def advance_states(self, adv: np.uint32) -> None:
            """Advance the state of the RNGList by adv"""
            adv = np.uint32(adv)
            for _ in range(adv):
                self.advance_state()

        def advance_state(self) -> None:
            """Advance the state of the RNGList by one"""
            self.list[self.head] = next_function(self.rng)
            self.head = (self.head + 1) & (size - 1)

            self.pointer = self.head

        def advance(self, adv: np.uint32) -> None:
            """Advance within the state by adv"""
            adv = np.uint32(adv)
            self.pointer = (self.pointer + adv) & (size - 1)

        def get_value(self):
            """Get value of the next rand"""
            result = self.list[self.pointer]
            self.pointer += 1
            self.pointer &= size - 1
            return result

    return SpecificRNGList
