"""Utility functions for the data submodule"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING
from typing_extensions import Protocol
import numpy as np


class DTypeDataclass(Protocol):
    """Protocol for a generic DTypeDataclass class"""

    dtype: np.dtype


if TYPE_CHECKING:
    BOOL8 = np.bool8
    U8 = np.uint8
    U16 = np.uint16
    U32 = np.uint32
    U64 = np.uint64
    F32 = np.float32
else:
    BOOL8 = np.dtype("b1")
    U8 = np.dtype("u1")
    U16 = np.dtype("<u2")
    U32 = np.dtype("<u4")
    U64 = np.dtype("<u8")
    F32 = np.dtype("<f4")


def dtype_dataclass(cls):
    """Decorator to turn a dataclass into a numpy dtype"""
    dataclass_cls = dataclass(cls)
    for field in fields(dataclass_cls):
        if hasattr(field, "__metadata__"):
            print(field.name, field.type.__metadata__)
    dataclass_cls.dtype = np.dtype(
        [
            (
                field.name,
                (
                    field.type.dtype
                    if hasattr(field.type, "dtype")
                    else (
                        field.type.__metadata__[0]
                        if hasattr(field.type, "__metadata__")
                        else field.type
                    )
                ),
            )
            for field in fields(dataclass_cls)
        ]
    )
    return dataclass_cls


def unused_bytes(size: int) -> np.dtype:
    """Numpy dtype for an unused byte array"""
    return dtype_array(U8, size)


def dtype_array(dtype: np.dtype, length: int) -> np.dtype:
    """Numpy dtype for an array of dtype of length"""
    return np.dtype([("", dtype, (length,))])[0]
