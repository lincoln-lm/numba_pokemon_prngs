"""Encounter Slot Area Specification for Pokemon: Legends Arceus"""

from typing import Annotated
import numpy as np
from ..util import dtype_array, dtype_dataclass, U8, U16, U32, F32
from ...enums import LAWeather, LATime
from ...compilation import optional_jitclass, array_type


@dtype_dataclass
class SlotLA:
    """Pokemon: Legends Arceus Encounter Slot"""

    species: U16
    gender: U8
    form: U8
    min_level: U8
    max_level: U8
    guaranteed_ivs: U8
    base_probability: U32
    is_alpha: U8
    time_multipliers: Annotated[list[LATime], dtype_array(F32, len(LATime))]
    weather_multipliers: Annotated[list[LAWeather], dtype_array(F32, len(LAWeather))]


SlotLA.dtype: np.dtype

# numba doesnt like SlotLA.dtype
SlotLA_dtype = SlotLA.dtype


@optional_jitclass
class EncounterAreaLA:
    """Pokemon: Legends Arceus Encounter Slot Area"""

    table_id: np.uint64
    slots: array_type(SlotLA.dtype)

    def __init__(self, table_id: np.uint64, slot_count: np.uint64) -> None:
        self.table_id = np.uint64(table_id)
        self.slots = np.zeros(slot_count, dtype=SlotLA_dtype)

    def calc_slot(self, rand: np.float32, time: LATime, weather: LAWeather) -> SlotLA:
        """Calculate encounter slot based on rand and time/weather"""
        probabilities = (
            self.slots.base_probability
            * self.slots.time_multipliers[:, time]
            * self.slots.weather_multipliers[:, weather]
        )
        scaled_rand = rand * np.sum(probabilities)
        for i, probability in enumerate(probabilities):
            if scaled_rand < probability:
                break
            scaled_rand -= probability
        return self.slots[i]
