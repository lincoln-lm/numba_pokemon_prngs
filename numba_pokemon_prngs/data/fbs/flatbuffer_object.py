"""Generic FlatBuffer object"""

from __future__ import annotations
from io import FileIO
from enum import IntEnum
from typing import Type, Callable, Any, Union, Tuple
from typing_extensions import Self
import flatbuffers
import numpy as np

U8 = flatbuffers.number_types.Uint8Flags
U16 = flatbuffers.number_types.Uint16Flags
U32 = flatbuffers.number_types.Uint32Flags
U64 = flatbuffers.number_types.Uint64Flags
I8 = flatbuffers.number_types.Int8Flags
I16 = flatbuffers.number_types.Int16Flags
I32 = flatbuffers.number_types.Int32Flags
I64 = flatbuffers.number_types.Int64Flags
F32 = flatbuffers.number_types.Float32Flags
F64 = flatbuffers.number_types.Float64Flags

INT_TYPES = Union[
    Type[U8],
    Type[U16],
    Type[U32],
    Type[U64],
    Type[I8],
    Type[I16],
    Type[I32],
    Type[I64],
]
FLOAT_TYPES = Union[
    Type[F32],
    Type[F64],
]


class FlatBufferObject:
    """Generic FlatBuffer object"""

    def __init__(self, buf: bytearray, offset: int = None):
        # offset is None implies this is a root object
        if offset is None:
            offset = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, 0)
        self._table = flatbuffers.table.Table(buf, offset)
        self._offset = offset
        self._counter = 4

    def dump_binary(self, stream: FileIO):
        """Dump binary to file stream"""
        stream.write(self._table.Bytes)

    def read_int(self, _type: INT_TYPES, position: int, default: int = None):
        """Read value of _type at position"""
        pos_offset = self._table.Offset(position)
        if pos_offset:
            return self._table.Get(_type, pos_offset + self._offset)
        return default

    def read_float(self, _type: FLOAT_TYPES, position: int, default: float = None):
        """Read value of _type at position"""
        pos_offset = self._table.Offset(position)
        if pos_offset:
            return self._table.Get(_type, pos_offset + self._offset)
        return default

    def read_int_enum(
        self,
        _type: INT_TYPES,
        position: int,
        _enum: Type[IntEnum] | Callable,
        default: Any = None,
    ):
        """Read value of _type at position as _enum"""
        pos_offset = self._table.Offset(position)
        if pos_offset:
            return _enum(self._table.Get(_type, pos_offset + self._offset))
        return default

    def read_init_int(self, _type: INT_TYPES, default: int = None):
        """Read value of _type at the position of self._counter
        and increment the counter"""
        value = self.read_int(_type, self._counter, default=default)
        self._counter += 2
        return value

    def read_init_float(self, _type: FLOAT_TYPES, default: float = None):
        """Read value of _type at the position of self._counter
        and increment the counter"""
        value = self.read_float(_type, self._counter, default=default)
        self._counter += 2
        return value

    def read_init_int_enum(
        self, _type: INT_TYPES, _enum: Type[IntEnum] | Callable, default: Any = None
    ):
        """Read value of _type at the position of self._counter as _enum
        and increment the counter"""
        value = self.read_int_enum(_type, self._counter, _enum, default=default)
        self._counter += 2
        return value

    def read_object(self, _object_type: Type[Self], position: int, default: Any = None):
        """Read FlatBufferObject of _object_type at position"""
        pos_offset = self._table.Offset(position)
        if pos_offset:
            val_offset = self._table.Indirect(pos_offset + self._offset)
            return _object_type(self._table.Bytes, offset=val_offset)
        return default

    def read_init_object(self, _object_type: Type[Self], default: Any = None):
        """Read FlatBufferObject of _object_type at the position of self._counter
        and increment the counter"""
        value = self.read_object(_object_type, self._counter, default=default)
        self._counter += 2
        return value

    def read_object_array(self, _object_type: Type[Self], position: int):
        """Read an array of FlatBufferObjects of _object_type at position"""
        array = []
        pos_offset = self._table.Offset(position)
        if pos_offset:
            array_offset = self._table.Vector(pos_offset)
            array_len = self._table.VectorLen(pos_offset)
            for array_offset in range(array_offset, array_offset + array_len * 4, 4):
                val_offset = self._table.Indirect(array_offset)
                array.append(_object_type(self._table.Bytes, offset=val_offset))
        return array

    def read_init_object_array(self, _object_type: Type[Self]):
        """Read an array of FlatBufferObjects of _object_type at the position of self._counter
        and increment the counter"""
        array = self.read_object_array(_object_type, self._counter)
        self._counter += 2
        return array

    def read_string(self, position: int, default=None):
        """Read a string at position"""
        array = default
        pos_offset = self._table.Offset(position)
        if pos_offset:
            array_offset = self._table.Vector(pos_offset)
            array_len = self._table.VectorLen(pos_offset)
            array = self._table.Bytes[array_offset : array_offset + array_len]
        return array.decode("utf-8")

    def read_init_string(self, default=None):
        """Read a string at the position of self._counter and increment the counter"""
        value = self.read_string(self._counter, default=default)
        self._counter += 2
        return value

    def read_vec3f(self, position: int, default=None):
        """Read a Vec3f at position"""
        return self.read_object(Vec3F, position, default=default)

    def read_init_vec3f(self, default=None) -> Vec3F:
        """Read a Vec3f at the position of self._counter and increment the counter"""
        value = self.read_vec3f(self._counter, default=default)
        self._counter += 2
        return value

    def read_init_padding(self, size: int):
        """Read padding for unused/unknown data"""
        self._counter += 2 * size


class Vec3F(FlatBufferObject):
    """Vector of 3 floats"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.x: np.float32 = self.read_init_float(F32)
        self.y: np.float32 = self.read_init_float(F32)
        self.z: np.float32 = self.read_init_float(F32)

    def as_tuple(self) -> Tuple[np.float32, np.float32, np.float32]:
        """Return vector as tuple"""
        return (self.x, self.y, self.z)
