"""Utility @njit compiled numba functions"""

import numpy as np
from .compilation import optional_njit, return_type


@optional_njit(return_type(np.uint32, (np.uint32,)), locals={"val": np.uint32})
def change_endian_u32(val: np.uint32) -> np.uint32:
    """Swap endian bytes"""
    val = np.uint32(val)
    val = ((val << np.uint32(8)) & np.uint32(0xFF00FF00)) | (
        (val >> np.uint32(8)) & np.uint32(0xFF00FF)
    )
    return np.uint32((val << np.uint32(16)) | (val >> np.uint32(16)))


@optional_njit(return_type(np.uint32, (np.uint32, np.uint8)), locals={"val": np.uint32})
def rotate_left_u32(val: np.uint32, count: np.uint8) -> np.uint32:
    """Rotate bits left by count"""
    val = np.uint32(val)
    count = np.uint8(count)
    return np.uint32((val << count) | (val >> (np.uint8(32) - count)))


@optional_njit(return_type(np.uint32, (np.uint32, np.uint8)), locals={"val": np.uint32})
def rotate_right_u32(val: np.uint32, count: np.uint8) -> np.uint32:
    """Rotate bits left by count"""
    val = np.uint32(val)
    count = np.uint8(count)
    return np.uint32((val << (np.uint8(32) - count)) | (val >> count))


@optional_njit(return_type(np.uint64, (np.uint64, np.uint8)), locals={"val": np.uint64})
def rotate_left_u64(val: np.uint64, count: np.uint8) -> np.uint64:
    """Rotate bits left by count"""
    val = np.uint64(val)
    count = np.uint8(count)
    return np.uint64((val << count) | (val >> (np.uint8(64) - count)))


@optional_njit(return_type(np.uint64, (np.uint64, np.uint8)), locals={"val": np.uint64})
def rotate_right_u64(val: np.uint64, count: np.uint8) -> np.uint64:
    """Rotate bits left by count"""
    val = np.uint64(val)
    count = np.uint8(count)
    return np.uint64((val << (np.uint8(64) - count)) | (val >> count))


NATURE_MODIFIERS = np.ones((25, 5), np.float32)
for i in range(25):
    NATURE_MODIFIERS[i][i % 5] -= 0.1
    NATURE_MODIFIERS[i][i // 5] += 0.1


@optional_njit(return_type(np.uint16, (np.uint16, np.uint8, np.uint8)))
def compute_stat(stat: np.uint16, nature: np.uint8, index: np.int8):
    """Compute stat after nature modifier"""
    return np.uint16(np.float32(stat) * NATURE_MODIFIERS[nature][index - 1])


HEX_LOOKUP = np.array([f"{i:02X}" for i in range(256)], dtype=np.str_)


@optional_njit(return_type(np.str_, (np.uint32,)))
def hex_32(num: np.uint32) -> np.str_:
    """Display 32-bit number as a 0 padded hex string"""
    return (
        HEX_LOOKUP[num >> 24]
        + HEX_LOOKUP[(num >> 16) & 0xFF]
        + HEX_LOOKUP[(num >> 8) & 0xFF]
        + HEX_LOOKUP[num & 0xFF]
    )
