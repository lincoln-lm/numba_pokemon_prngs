"""Xorshift128 Pseudo Random Number Generator"""

from __future__ import annotations
import numpy as np
from ..compilation import optional_jitclass, array_type


# TODO: jump tables
# TODO: staticmethod const functions
@optional_jitclass
class Xorshift128:
    """Xorshift128 Pseudo Random Number Generator"""

    state: array_type(np.uint32)  # contiguous array

    def __init__(
        self, seed_0: np.uint32, seed_1: np.uint32, seed_2: np.uint32, seed_3: np.uint32
    ) -> None:
        self.state = np.empty(4, dtype=np.uint32)
        self.re_init(
            np.uint32(seed_0), np.uint32(seed_1), np.uint32(seed_2), np.uint32(seed_3)
        )

    def re_init(
        self, seed_0: np.uint32, seed_1: np.uint32, seed_2: np.uint32, seed_3: np.uint32
    ) -> None:
        """Reinitialize without creating a new object"""
        self.state[0] = np.uint32(seed_0)
        self.state[1] = np.uint32(seed_1)
        self.state[2] = np.uint32(seed_2)
        self.state[3] = np.uint32(seed_3)

    def advance(self, adv: np.uint32) -> None:
        """Advance Xorshift128 sequence by adv"""
        for _ in range(adv):
            self.next()

    def next(self) -> np.uint32:
        """Advance and return the next 32-bit rand"""
        t_val = np.uint32(self.state[0])
        s_val = np.uint32(self.state[3])

        t_val ^= np.uint32(t_val) << np.uint32(11)
        t_val ^= np.uint32(t_val) >> np.uint32(8)
        t_val ^= np.uint32(s_val) ^ (np.uint32(s_val) >> np.uint32(19))

        self.state[0] = self.state[1]
        self.state[1] = self.state[2]
        self.state[2] = self.state[3]
        self.state[3] = t_val

        return np.uint32(t_val)

    def previous(self) -> np.uint32:
        """Advance backwards and return the last 32-bit rand"""
        rand = self.state[3]
        t_val = self.state[3]
        s_val = self.state[2]

        t_val ^= np.uint32(s_val) ^ (np.uint32(s_val) >> np.uint32(19))

        t_val ^= np.uint32(t_val) >> np.uint32(8)
        t_val ^= np.uint32(t_val) >> np.uint32(16)

        t_val ^= np.uint32(t_val) << np.uint32(11)
        t_val ^= np.uint32(t_val) << np.uint32(22)

        self.state[3] = self.state[2]
        self.state[2] = self.state[1]
        self.state[1] = self.state[0]
        self.state[0] = t_val

        return rand

    def next_randrange(self, minimum: np.int32, maximum: np.int32) -> np.uint32:
        """Generate and return the next [minimum, maximum) random uint"""
        # the usage of int32 here is intentional
        minimum = np.int32(minimum)
        maximum = np.int32(maximum)

        diff = np.uint32(maximum - minimum)

        return np.uint32((self.next() % diff) + minimum)

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint"""
        maximum = np.uint32(maximum)

        return self.next() % maximum

    def next_alternate_rand(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint
        via next_randrange(-0x80000000, 0x7FFFFFFF) % maximum"""
        return (
            self.next_randrange(
                np.int32(-0x7FFFFFFF) - np.uint32(1), np.int32(0x7FFFFFFF)
            )
            % maximum
        )

    def next_float_randrange(
        self, minimum: np.float32, maximum: np.float32
    ) -> np.float32:
        """Generate and return the next [minimum, maximum) random float"""
        minimum = np.float32(minimum)
        maximum = np.float32(maximum)

        rand_float = np.float32(self.next() & np.uint32(0x7FFFFF)) / np.float32(
            0x7FFFFF
        )

        return np.float32(np.float32(rand_float * minimum) + np.float32(np.float32(np.float32(1) - rand_float) * maximum))
