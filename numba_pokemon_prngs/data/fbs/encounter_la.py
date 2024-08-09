"""Encounter specifications for Pokemon: Legends Arceus"""

from typing import List, Dict, Tuple
import numpy as np
from .flatbuffer_object import FlatBufferObject, I32, U64, F32, I8, Vec3F


class EncounterMultiplier8aTable(FlatBufferObject):
    """Array of Encounter Multipliers (root object)"""

    def __init__(self, buf: bytearray):
        FlatBufferObject.__init__(self, buf)
        self.encounter_multipliers: List[
            EncounterMultiplier8a
        ] = self.read_init_object_array(EncounterMultiplier8a)
        self.multiplier_lookup: Dict[Tuple[int, int], EncounterMultiplier8a] = {
            (multiplier.species, multiplier.form or 0): multiplier
            for multiplier in self.encounter_multipliers
        }


class EncounterMultiplier8a(FlatBufferObject):
    """Encounter slot multipliers"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.species: np.int32 = self.read_init_int(I32)
        self.form: np.int32 = self.read_init_int(I32, 0)
        self.time_of_day_multipliers: Tuple[
            np.float32, np.float32, np.float32, np.float32, np.float32
        ] = tuple(self.read_init_float(F32, default=0.0) for _ in range(4))
        self.weather_multipliers: Tuple[
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
        ] = (1.0,) + tuple(self.read_init_float(F32, default=0.0) for _ in range(8))


class PokeMisc8aTable(FlatBufferObject):
    """Array of miscellaneous pokemon information (root object)"""

    def __init__(self, buf: bytearray):
        FlatBufferObject.__init__(self, buf)
        self.misc_list: List[
            PokeMisc8a
        ] = self.read_init_object_array(PokeMisc8a)
        self.misc_lookup: Dict[Tuple[int, int], PokeMisc8a] = {
            (misc.species, misc.form or 0): misc
            for misc in self.misc_list
        }


class PokeMisc8a(FlatBufferObject):
    """Miscellaneous pokemon information"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.species: np.int32 = self.read_init_int(I32)
        self.form: np.int32 = self.read_init_int(I32, 0)
        self.read_init_padding(4)
        self.alpha_level_index: np.int32 = self.read_init_int(I32, 0)
        # self.read_init_padding(16)

class PlacementSpawner8aTable(FlatBufferObject):
    """Array of Spawner specifications (root object)"""

    def __init__(self, buf: bytearray):
        FlatBufferObject.__init__(self, buf)
        self.spawners: List[PlacementSpawner8a] = self.read_init_object_array(
            PlacementSpawner8a
        )


class PlacementSpawner8a(FlatBufferObject):
    """Spawner specification"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.spawner_id: np.uint64 = self.read_init_int(U64)
        self.read_init_padding(1)
        # a little hacky
        self.coordinates: Vec3F = self.read_init_object_array(PlacementParameters8a)[
            0
        ].coordinates
        self.read_init_padding(4)
        self.min_spawn_count: np.int32 = self.read_init_int(I32)
        self.max_spawn_count: np.int32 = self.read_init_int(I32)
        self.read_init_padding(1)
        self.is_mass_outbreak: np.bool8 = self.read_init_int_enum(I8, bool)
        self.is_water: np.bool8 = self.read_init_int_enum(I8, bool)
        self.is_sky: np.bool8 = self.read_init_int_enum(I8, bool)
        self.read_init_padding(7)
        # a little hacky
        self.encounter_table_id: np.uint64 = self.read_init_object_array(
            PlacementSpawner8aF20
        )[0].encounter_table_id
        # self.read_init_padding(2)


class PlacementSpawner8aF20(FlatBufferObject):
    """Field 20 of PlacementSpawner8a"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.encounter_table_id: np.uint64 = self.read_init_int(U64)
        # self.read_init_padding(8)


class PlacementParameters8a(FlatBufferObject):
    """Parameters for the placement of a spawner"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.read_init_padding(7)
        self.coordinates: Vec3F = self.read_init_vec3f()
        # self.read_init_padding(2)


class EncounterTable8aTable(FlatBufferObject):
    """Table of EncounterTable8a (root object)"""

    def __init__(self, buf: bytearray):
        super().__init__(buf)
        self.encounter_tables: List[EncounterTable8a] = self.read_init_object_array(
            EncounterTable8a
        )
        self.encounter_table_lookup: Dict[np.uint64, EncounterTable8a] = {
            table.table_id: table for table in self.encounter_tables
        }


class EncounterTable8a(FlatBufferObject):
    """Table of encounter slots"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.table_id: np.uint64 = self.read_init_int(U64)
        self.min_level: np.int32 = self.read_init_int(I32)
        self.max_level: np.int32 = self.read_init_int(I32)
        self.encounter_slots: List[EncounterSlot8a] = self.read_init_object_array(
            EncounterSlot8a
        )


class EncounterSlot8a(FlatBufferObject):
    """Encounter slot specification"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.species: np.int32 = self.read_init_int(I32)
        self.read_init_padding(1) # slot id
        self.gender: np.int32 = self.read_init_int(I32, 0)
        self.form: np.int32 = self.read_init_int(I32)
        self.read_init_padding(21)
        self.guaranteed_ivs: np.int32 = self.read_init_int(I32)
        self.read_init_padding(3)
        self.base_probability: np.int32 = self.read_init_int(I32)
        self.override_min_level: np.int32 = self.read_init_int(I32, 0)
        self.override_max_level: np.int32 = self.read_init_int(I32, 0)
        self.read_init_padding(13)
        self.encounter_eligibility: EncounterEligibilityTraits8a = (
            self.read_init_object(EncounterEligibilityTraits8a)
        )
        self.is_alpha: bool = self.read_init_object(EncounterOybnTraits8a).is_alpha


class EncounterEligibilityTraits8a(FlatBufferObject):
    """Encounter slot eligibility traits"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.read_init_padding(7)
        self.time_of_day_multipliers: Tuple[
            np.float32, np.float32, np.float32, np.float32, np.float32
        ] = tuple(self.read_init_float(F32, default=0.0) for _ in range(4))
        self.weather_multipliers: Tuple[
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
        ] = (1.0,) + tuple(self.read_init_float(F32, default=0.0) for _ in range(8))


class EncounterOybnTraits8a(FlatBufferObject):
    """Encounter slot OYBN (alpha?) traits"""

    def __init__(self, buf: bytearray, offset: int):
        super().__init__(buf, offset)
        self.is_alpha: bool = any(self.read_init_int_enum(I8, bool) for _ in range(2))
        # self.read_init_padding(2)
