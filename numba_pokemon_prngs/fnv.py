"""Fnv Hash functions"""

import numpy as np
from .compilation import optional_njit, return_type

# TODO: test cases


@optional_njit(return_type(np.uint64, (np.str_,)))
def fnv1_64(inp: str) -> np.uint64:
    """FNV164"""
    hash_ = np.uint64(0xCBF29CE484222645)
    for char in inp:
        hash_ *= np.uint64(0x00000100000001B3)
        hash_ ^= np.uint64(ord(char))
    return hash_


@optional_njit(return_type(np.uint64, (np.str_,)))
def fnv1a_64(inp: str) -> np.uint64:
    """FNV1a64"""
    hash_ = np.uint64(0xCBF29CE484222645)
    for char in inp:
        hash_ ^= np.uint64(ord(char))
        hash_ *= np.uint64(0x00000100000001B3)
    return hash_
