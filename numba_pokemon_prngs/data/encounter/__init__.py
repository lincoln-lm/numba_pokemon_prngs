"""Encounter data for Pokemon games"""

import re
import json
import pickle
import os
import pathlib
import importlib.resources as pkg_resources
import numpy as np
from platformdirs import user_cache_dir
from .encounter_area_3 import EncounterArea3
from .encounter_area_la import EncounterAreaLA
from ..fbs.encounter_la import (
    EncounterTable8aTable,
    PlacementSpawner8aTable,
    EncounterMultiplier8aTable,
    PokeMisc8aTable,
    NewHugeOutbreakGroupLotteryTable8a,
    NewHugeOutbreakGroupTable8a,
)
from .. import CONSTANT_CASE_SPECIES_EN
from ...resources import encount
from ...enums import Game, LAArea
from ...compilation import TypedDict
from ...fnv import fnv1a_64


def pickle_cached(filename: str, encode=None, decode=None):
    """Cache results of a function next to the executable"""
    cache_dir = user_cache_dir("numba_pokemon_prngs", "Lincoln-LM")
    filename = cache_dir + "/" + filename
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def wrapper(function):
        def wrapped(*args, **kwargs):
            pkl_filename = filename.format(*args, **kwargs)
            if os.path.exists(pkl_filename):
                with open(pkl_filename, "rb") as pickled_file:
                    data = pickled_file.read()
                    if data:
                        data = pickle.loads(data)
                        if decode is not None:
                            return decode(data)
                        return data
            result = function(*args, **kwargs)
            with open(pkl_filename, "wb+") as pickled_file:
                if encode is not None:
                    pickled_file.write(pickle.dumps(encode(result)))
                else:
                    pickle.dump(result, pickled_file)
            return result

        return wrapped

    return wrapper


ENCOUNTER_DATA_FILES = {
    Game.RUBY: "rs_wild_encounters.json",
    Game.SAPPHIRE: "rs_wild_encounters.json",
    Game.EMERALD: "e_wild_encounters.json",
    Game.FIRE_RED: "frlg_wild_encounters.json",
    Game.LEAF_GREEN: "frlg_wild_encounters.json",
}

UNOWN_ENCOUNTERS = {
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_MONEAN_CHAMBER": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        27,
    ],
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_LIPTOO_CHAMBER": [
        2,
        2,
        2,
        3,
        3,
        3,
        7,
        7,
        7,
        20,
        20,
        14,
    ],
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_WEEPTH_CHAMBER": [
        13,
        13,
        13,
        13,
        18,
        18,
        18,
        18,
        8,
        8,
        4,
        4,
    ],
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_DILFORD_CHAMBER": [
        15,
        15,
        11,
        11,
        9,
        9,
        17,
        17,
        17,
        16,
        16,
        16,
    ],
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_SCUFIB_CHAMBER": [
        24,
        24,
        19,
        19,
        6,
        6,
        6,
        5,
        5,
        5,
        10,
        10,
    ],
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_RIXY_CHAMBER": [
        21,
        21,
        21,
        22,
        22,
        22,
        23,
        23,
        12,
        12,
        1,
        1,
    ],
    "MAP_SEVEN_ISLAND_TANOBY_RUINS_VIAPOIS_CHAMBER": [
        25,
        25,
        25,
        25,
        25,
        25,
        25,
        25,
        25,
        25,
        25,
        26,
    ],
}


def clean_map_string(map_string: str) -> str:
    """Clean a map string to be easily readable"""
    map_string = map_string.replace("MAP_", "").replace("_", " ")

    strings = map_string.split()
    for i, string in enumerate(strings):
        match = re.match(r"Route(\d+)", string, re.IGNORECASE)
        if match:
            strings[i] = f"Route {match[1]}"
        else:
            match = re.match(r"Room(\d+)", string, re.IGNORECASE)
            if match:
                strings[i] = f"Room {match[1]}"
            elif re.match(r"B(\d+)F", string):
                strings[i] = string
            elif re.match(r"(\d+)F", string):
                strings[i] = string
            elif re.match(r"(\d+)R", string):
                strings[i] = string
            else:
                strings[i] = string.capitalize()

    return " ".join(strings)


@pickle_cached("encounter_3_{}.pkl")
def load_encounter_3(game: Game) -> tuple[tuple[str], np.recarray]:
    """Load Gen 3 encounter data"""
    with pkg_resources.open_text(
        encount, ENCOUNTER_DATA_FILES[game]
    ) as wild_encounters:
        encounters = json.load(wild_encounters)["wild_encounter_groups"][0][
            "encounters"
        ]

    if game == Game.RUBY:
        encounters = tuple(filter(lambda x: "Ruby" in x["base_label"], encounters))
    elif game == Game.SAPPHIRE:
        encounters = tuple(filter(lambda x: "Sapphire" in x["base_label"], encounters))
    elif game == Game.FIRE_RED:
        encounters = tuple(filter(lambda x: "FireRed" in x["base_label"], encounters))
    elif game == Game.LEAF_GREEN:
        encounters = tuple(filter(lambda x: "LeafGreen" in x["base_label"], encounters))

    encounter_information = np.zeros(len(encounters), dtype=EncounterArea3.dtype).view(
        np.recarray
    )
    map_names = []
    for map_number, encounter in enumerate(encounters):
        if re.match(r"gAlteringCave[2-9]", encounter["base_label"]):
            map_names.append("")
            continue

        if re.search(r"AlteringCave_[2-9]", encounter["base_label"]):
            map_names.append("")
            continue

        if "Unused" in encounter["base_label"]:
            map_names.append("")
            continue

        map_names.append(clean_map_string(encounter["map"]))

        encounter_area: EncounterArea3 = encounter_information[map_number]
        encounter_area.location = map_number

        land_mons = "land_mons" in encounter
        water_mons = "water_mons" in encounter
        rock_smash_mons = "rock_smash_mons" in encounter
        fishing_mons = "fishing_mons" in encounter

        if land_mons:
            encounter_area.land_rate = encounter["land_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["land_mons"]["mons"]):
                encounter_slot = encounter_area.land[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])
                if encounter["map"] in UNOWN_ENCOUNTERS:
                    form = UNOWN_ENCOUNTERS[encounter["map"]][i]
                    encounter_slot.species |= form << 11
        if water_mons:
            encounter_area.water_rate = encounter["water_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["water_mons"]["mons"]):
                encounter_slot = encounter_area.water[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])
        if rock_smash_mons:
            encounter_area.rock_rate = encounter["rock_smash_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["rock_smash_mons"]["mons"]):
                encounter_slot = encounter_area.rock[i]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])
        if fishing_mons:
            encounter_area.fish_rate = encounter["fishing_mons"]["encounter_rate"]
            for i, slot in enumerate(encounter["fishing_mons"]["mons"]):
                if i < 2:
                    encounter_slot = encounter_area.fish_old[i]
                elif i < 5:
                    encounter_slot = encounter_area.fish_good[i - 2]
                else:
                    encounter_slot = encounter_area.fish_super[i - 5]
                encounter_slot.min_level = slot["min_level"]
                encounter_slot.max_level = slot["max_level"]
                encounter_slot.species = CONSTANT_CASE_SPECIES_EN.index(slot["species"])

    return tuple(map_names), encounter_information


MAP_NAMES_RUBY, ENCOUNTER_INFORMATION_RUBY = load_encounter_3(Game.RUBY)
MAP_NAMES_SAPPHIRE, ENCOUNTER_INFORMATION_SAPPHIRE = load_encounter_3(Game.SAPPHIRE)
MAP_NAMES_EMERALD, ENCOUNTER_INFORMATION_EMERALD = load_encounter_3(Game.EMERALD)
MAP_NAMES_FIRE_RED, ENCOUNTER_INFORMATION_FIRE_RED = load_encounter_3(Game.FIRE_RED)
MAP_NAMES_LEAF_GREEN, ENCOUNTER_INFORMATION_LEAF_GREEN = load_encounter_3(
    Game.LEAF_GREEN
)

MAP_NAMES_GEN3 = {
    Game.RUBY: MAP_NAMES_RUBY,
    Game.SAPPHIRE: MAP_NAMES_SAPPHIRE,
    Game.EMERALD: MAP_NAMES_EMERALD,
    Game.FIRE_RED: MAP_NAMES_FIRE_RED,
    Game.LEAF_GREEN: MAP_NAMES_LEAF_GREEN,
}

ENCOUNTER_INFORMATION_GEN3 = {
    Game.RUBY: ENCOUNTER_INFORMATION_RUBY,
    Game.SAPPHIRE: ENCOUNTER_INFORMATION_SAPPHIRE,
    Game.EMERALD: ENCOUNTER_INFORMATION_EMERALD,
    Game.FIRE_RED: ENCOUNTER_INFORMATION_FIRE_RED,
    Game.LEAF_GREEN: ENCOUNTER_INFORMATION_LEAF_GREEN,
}


def encounter_tables_la_decode(regular_dict):
    map_encounter_area_lookup = TypedDict(
        key_type=np.uint64, value_type=EncounterAreaLA
    )
    for key, value in regular_dict.items():
        table_id, slots = value
        value = EncounterAreaLA(np.uint64(table_id), len(slots))
        value.slots = slots
        map_encounter_area_lookup[np.uint64(key)] = value
    return map_encounter_area_lookup


def encounter_tables_la_encode(typed_dict):
    regular_dict = {}
    for key in typed_dict.keys():
        value = typed_dict[np.uint64(key)]
        regular_dict[int(key)] = (value.table_id, value.slots)
    return regular_dict


@pickle_cached(
    "encounter_tables_la_{}.pkl",
    encode=encounter_tables_la_encode,
    decode=encounter_tables_la_decode,
)
def load_encounter_tables_la(map_area: LAArea):
    """Load encounter tables for a map in Pokemon: Legends Arceus"""
    map_encounter_area_lookup = TypedDict(
        key_type=np.uint64, value_type=EncounterAreaLA
    )
    flatbuffer_encounter_lookup = EncounterTable8aTable(
        pkg_resources.read_binary(encount.la, f"ha_area{map_area:02d}_encounter.bin")
    ).encounter_tables
    for encounter_area_fb in flatbuffer_encounter_lookup:
        encounter_area = EncounterAreaLA(
            np.uint64(encounter_area_fb.table_id),
            len(encounter_area_fb.encounter_slots),
        )
        encounter_area_slots = encounter_area.slots.view(np.recarray)
        for i, slot in enumerate(encounter_area_fb.encounter_slots):
            encounter_area_slots[i].species = slot.species
            encounter_area_slots[i].gender = slot.gender
            encounter_area_slots[i].form = slot.form or 0
            encounter_area_slots[i].guaranteed_ivs = slot.guaranteed_ivs or 0
            encounter_area_slots[i].min_level = (
                slot.override_min_level
                if 0 < slot.override_min_level <= 100
                else encounter_area_fb.min_level
            )
            encounter_area_slots[i].max_level = (
                slot.override_max_level
                if 0 < slot.override_max_level <= 100
                else encounter_area_fb.max_level
            )
            encounter_area_slots[i].base_probability = slot.base_probability
            encounter_area_slots[i].is_alpha = slot.is_alpha
            # apply OYBN level range boosts
            if slot.is_alpha:
                poke_misc_data = POKE_MISC_LA.misc_lookup[
                    (slot.species, slot.form or 0)
                ]
                level_boost = ALPHA_LEVEL_BOOSTS_LA[
                    poke_misc_data.alpha_level_index - 1
                ]
                encounter_area_slots[i].min_level += level_boost
                encounter_area_slots[i].max_level += level_boost
            # bake multipliers into encounter area table

            encounter_area_slots[i].time_multipliers = np.where(
                # if multiplier < 0 (-1.0), use species specific multiplier
                np.array(slot.encounter_eligibility.time_of_day_multipliers) < 0.0,
                ENCOUNTER_MULTIPLIER_LA.multiplier_lookup[
                    (slot.species, slot.form or 0)
                ].time_of_day_multipliers,
                slot.encounter_eligibility.time_of_day_multipliers,
            )
            encounter_area_slots[i].time_multipliers = np.where(
                # if species specific multiplier < 0 (-1.0), use 1.0
                encounter_area_slots[i].time_multipliers < 0.0,
                1.0,
                encounter_area_slots[i].time_multipliers,
            )
            encounter_area_slots[i].weather_multipliers = np.where(
                # if multiplier < 0 (-1.0), use species specific multiplier
                np.array(slot.encounter_eligibility.weather_multipliers) < 0.0,
                ENCOUNTER_MULTIPLIER_LA.multiplier_lookup[
                    (slot.species, slot.form or 0)
                ].weather_multipliers,
                slot.encounter_eligibility.weather_multipliers,
            )
            encounter_area_slots[i].weather_multipliers = np.where(
                # if species specific multiplier < 0 (-1.0), use 1.0
                encounter_area_slots[i].weather_multipliers < 0.0,
                1.0,
                encounter_area_slots[i].weather_multipliers,
            )
        map_encounter_area_lookup[np.uint64(encounter_area.table_id)] = encounter_area
    return map_encounter_area_lookup


ENCOUNTER_MULTIPLIER_LA = EncounterMultiplier8aTable(
    pkg_resources.read_binary(encount.la, "poke_encount.bin")
)

POKE_MISC_LA = PokeMisc8aTable(pkg_resources.read_binary(encount.la, "poke_misc.bin"))

NHO_LOTTERY_TABLE_LA = NewHugeOutbreakGroupLotteryTable8a(
    pkg_resources.read_binary(encount.la, "new_huge_outbreak_group_lottery.bin")
)

NHO_GROUP_TABLE_LA = NewHugeOutbreakGroupTable8a(
    pkg_resources.read_binary(encount.la, "new_huge_outbreak_group.bin")
)

ALPHA_LEVEL_BOOSTS_LA = (15, 15, 15, 20, 20)

ENCOUNTER_INFORMATION_LA = {
    map_area: load_encounter_tables_la(map_area)
    for map_area in (
        LAArea.OBSIDIAN_FIELDLANDS,
        LAArea.CRIMSON_MIRELANDS,
        LAArea.CORONET_HIGHLANDS,
        LAArea.COBALT_COASTLANDS,
        LAArea.ALABASTER_ICELANDS,
    )
}

SPAWNER_INFORMATION_LA = {
    map_area: PlacementSpawner8aTable(
        pkg_resources.read_binary(encount.la, f"ha_area{map_area:02d}_spawner.bin")
    )
    for map_area in (
        LAArea.OBSIDIAN_FIELDLANDS,
        LAArea.CRIMSON_MIRELANDS,
        LAArea.CORONET_HIGHLANDS,
        LAArea.COBALT_COASTLANDS,
        LAArea.ALABASTER_ICELANDS,
    )
}

SPAWNER_NAMES_LA = json.loads(
    pkg_resources.read_text(encount.la, "spawner_names.json"),
    object_hook=lambda x: {int(k): v for k, v in x.items()},
)

ENCOUNTER_TABLE_NAMES_LA = json.loads(
    pkg_resources.read_text(encount.la, "encounter_table_names.json"),
    object_hook=lambda x: {int(k): v for k, v in x.items()},
)

# SPAWNER_NAMES_LA = {}
# for i in range(10000):
#     SPAWNER_NAMES_LA[fnv1a_64(f"{i:04d}")] = f"{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"1{i:04d}")] = f"1{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"02{i:04d}")] = f"02{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"03{i:04d}")] = f"03{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"4{i:04d}")] = f"4{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"5{i:04d}")] = f"5{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"ev{i:04d}")] = f"ev{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"ex_{i:04d}")] = f"ex_{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"eve_ex_{i:04d}")] = f"eve_ex_{i:04d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"huge_{i:04d}")] = f"huge_{i:04d}"
#     for j in range(20):
#         SPAWNER_NAMES_LA[fnv1a_64(f"{i:04d}_{j:02d}")] = f"{i:04d}_{j:02d}"
# for i in range(100):
#     SPAWNER_NAMES_LA[fnv1a_64(f"area00_{i:02d}")] = f"area00_{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"poke{i:02d}")] = f"poke{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"sky{i:02d}")] = f"sky{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"lnd_no_{i:02d}")] = f"lnd_no_{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"sky_{i:02d}")] = f"sky_{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"ex_mkrg_{i:02d}")] = f"ex_mkrg_{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"ex_unnn_{i:02d}")] = f"ex_unnn_{i:02d}"
#     SPAWNER_NAMES_LA[fnv1a_64(f"ex_trs_{i:02d}")] = f"ex_trs_{i:02d}"

# SPAWNER_NAMES_LA[fnv1a_64("ha_area01_s01_ev001")] = "ha_area01_s01_ev001"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area02_s02_ev001")] = "ha_area02_s02_ev001"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area02_s02_ev002")] = "ha_area02_s02_ev002"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area03_s03_ev001")] = "ha_area03_s03_ev001"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area04_ev001")] = "ha_area04_ev001"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area05_s03_ev001")] = "ha_area05_s03_ev001"

# SPAWNER_NAMES_LA[fnv1a_64("area03_s04_ev001")] = "area03_s04_ev001"
# SPAWNER_NAMES_LA[fnv1a_64("area03_s04_ev002")] = "area03_s04_ev002"
# SPAWNER_NAMES_LA[fnv1a_64("area03_s04_ev003")] = "area03_s04_ev003"
# SPAWNER_NAMES_LA[fnv1a_64("area03_s04_ev003")] = "area03_s04_ev003"
# SPAWNER_NAMES_LA[fnv1a_64("area03_s04_ev005")] = "area03_s04_ev005"

# SPAWNER_NAMES_LA[fnv1a_64("ha_area01_s01_1000")] = "ha_area01_s01_1000"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area02_s02_1000")] = "ha_area02_s02_1000"
# SPAWNER_NAMES_LA[fnv1a_64("ha_area05_s03_1000")] = "ha_area05_s03_1000"

# for i in range(1, 8):
#     for j in range(1, 4):
#         for k in range(1, 7):
#             SPAWNER_NAMES_LA[
#                 fnv1a_64(f"main_whSpawner{i:02d}{j:02d}_{k:02d}")
#             ] = f"main_whSpawner{i:02d}{j:02d}_{k:02d}"
#             SPAWNER_NAMES_LA[
#                 fnv1a_64(f"sub_whSpawner{i:02d}{j:02d}_{k:02d}")
#             ] = f"sub_whSpawner{i:02d}{j:02d}_{k:02d}"

# ENCOUNTER_TABLE_NAMES_LA = {}
# for prefix in ("eve", "fly", "gmk", "lnd", "mas", "oyb", "swm", "whl"):
#     for kind in ("ex", "no", "ra"):
#         for i in range(150):
#             ENCOUNTER_TABLE_NAMES_LA[
#                 fnv1a_64(f"{prefix}_{kind}_{i:02d}")
#             ] = f"{prefix}_{kind}_{i:02d}"
# for area in range(6):
#     for i in range(10):
#         ENCOUNTER_TABLE_NAMES_LA[
#             fnv1a_64(f"sky_area{area}_{i:02d}")
#         ] = f"sky_area{area}_{i:02d}"
# ENCOUNTER_TABLE_NAMES_LA[fnv1a_64("eve_ex_16_b")] = "eve_ex_16_b"
# ENCOUNTER_TABLE_NAMES_LA[fnv1a_64("eve_ex_17_b")] = "eve_ex_17_b"
# ENCOUNTER_TABLE_NAMES_LA[fnv1a_64("eve_ex_18_b")] = "eve_ex_18_b"

# for gimmick in ("no", "tree", "rock", "crystal", "snow", "box"):
#     for i in range(100):
#         ENCOUNTER_TABLE_NAMES_LA[
#             fnv1a_64(f"gmk_{gimmick}_{i:02d}")
#         ] = f"gmk_{gimmick}_{i:02d}"
#         for j in range(3):
#             ENCOUNTER_TABLE_NAMES_LA[
#                 fnv1a_64(f"gmk_{gimmick}_{i:02d}_{j:02d}")
#             ] = f"gmk_{gimmick}_{i:02d}_{j:02d}"
#             for k in range(3):
#                 ENCOUNTER_TABLE_NAMES_LA[
#                     fnv1a_64(f"gmk_{gimmick}_{i:02d}_{j:02d}_{k:02d}")
#                 ] = f"gmk_{gimmick}_{i:02d}_{j:02d}_{k:02d}"
