"""Tiny Mersenne Twister Pseudo Random Number Generator"""

from __future__ import annotations
import numpy as np
from ..compilation import optional_jitclass, array_type


# TODO: jump tables
# TODO: staticmethod const functions
# TODO: reverse next
@optional_jitclass
class TinyMersenneTwister:
    """Tiny Mersenne Twister Pseudo Random Number Generator"""

    state: array_type(np.uint32)  # contiguous array

    def __init__(self, seed: np.uint32) -> None:
        seed = np.uint32(seed)
        self.state = np.empty(4, dtype=np.uint32)
        self.re_init(seed)

    def re_init(self, seed: np.uint32) -> None:
        """Reinitialize without creating a new object"""
        seed = np.uint32(seed)
        self.state[0] = seed
        self.state[1] = 0x8F7011EE
        self.state[2] = 0xFC78FF1F
        self.state[3] = 0x3793FDFF
        for i in range(1, 8):
            self.state[i & 3] ^= np.uint32(
                np.uint32(0x6C078965)
                * (self.state[(i - 1) & 3] ^ (self.state[(i - 1) & 3] >> np.uint32(30)))
                + np.uint32(i)
            )

        if (
            self.state[0] & np.uint32(0x7FFFFFFF) == np.uint32(0)
            and self.state[1] == np.uint32(0)
            and self.state[2] == np.uint32(0)
            and self.state[3] == np.uint32(0)
        ):
            self.state[0] = ord("T")
            self.state[1] = ord("I")
            self.state[2] = ord("N")
            self.state[3] = ord("Y")

        for _ in range(8):
            self.shuffle()

    def advance(self, adv: np.uint32) -> None:
        """Advance Tiny Mersenne Twister sequence by adv"""
        for _ in range(adv):
            self.shuffle()

    def next(self) -> np.uint32:
        """Advance and return the next 32-bit tempered rand"""
        self.shuffle()
        return self.temper()

    def shuffle(self) -> None:
        """Advance the state by 1"""
        y_val = self.state[3]
        x_val = (self.state[0] & np.uint32(0x7FFFFFFF)) ^ self.state[1] ^ self.state[2]

        x_val ^= x_val << np.uint32(1)
        y_val ^= (y_val >> np.uint32(1)) ^ x_val

        self.state[0] = self.state[1]
        self.state[1] = self.state[2] ^ ((y_val & np.uint32(1)) * np.uint32(0x8F7011EE))
        self.state[2] = (
            x_val ^ (y_val << np.uint32(10)) ^ ((y_val & 1) * np.uint32(0xFC78FF1F))
        )
        self.state[3] = y_val

    def temper(self) -> np.uint32:
        """Access and return the next 32-bit tempered rand"""
        temper_0 = self.state[3]
        temper_1 = self.state[0] + (self.state[2] >> np.uint32(8))

        temper_0 ^= temper_1

        return temper_0 ^ ((temper_1 & 1) * np.uint32(0x3793FDFF))

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint
        via multiplication-shift distribution"""
        return np.uint32((np.uint64(self.next()) * np.uint64(maximum)) >> np.uint64(32))

    def next_rand_mod(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint via modulo distribution"""
        return self.next() % np.uint32(maximum)
