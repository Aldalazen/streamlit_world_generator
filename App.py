import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Traveller Advanced System Generator", layout="wide")

# =============================================================================
# Core helpers
# =============================================================================
EHEX = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
HEX_ONLY_16 = "0123456789ABCDEF"
STARPORT_TABLE = [(2, "X"), (4, "E"), (6, "D"), (8, "C"), (10, "B"), (12, "A")]


def roll_d6(n: int = 1) -> int:
    return sum(random.randint(1, 6) for _ in range(n))


def roll_2d6(dm: int = 0) -> int:
    return roll_d6(2) + dm


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def to_ehex(value: int, max_base_16: bool = False) -> str:
    digits = HEX_ONLY_16 if max_base_16 else EHEX
    value = max(0, min(value, len(digits) - 1))
    return digits[value]


def weighted_choice(options: List[Tuple[str, int]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    total = sum(weight for _, weight in options)
    pick = rng.randint(1, total)
    run = 0
    for value, weight in options:
        run += weight
        if pick <= run:
            return value
    return options[-1][0]


def make_system_seed(subsector_seed: str, hex_code: str) -> str:
    raw = f"{subsector_seed}:{hex_code}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def with_temp_seed(seed: str):
    class _TempSeed:
        def __enter__(self_inner):
            self_inner.state = random.getstate()
            random.seed(seed)
            return self_inner
        def __exit__(self_inner, exc_type, exc, tb):
            random.setstate(self_inner.state)
    return _TempSeed()


def zone_label(orbit_au: float, hz_inner: float, hz_outer: float) -> str:
    if orbit_au < hz_inner:
        return "Inner"
    if orbit_au <= hz_outer:
        return "Habitable"
    return "Outer"


def safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))


def parsec_flux_relative(luminosity_sol: float, distance_pc: float = 1.0) -> float:
    # Relative apparent brightness proxy at a chosen parsec distance.
    # At fixed distance, flux is proportional to luminosity / d^2.
    if distance_pc <= 0:
        distance_pc = 1.0
    return luminosity_sol / (distance_pc ** 2)


# =============================================================================
# Stellar data
# =============================================================================
STAR_MAIN_SEQUENCE = {
    "O0": {"mass": 90.0, "temp": 50000, "lum": 1_000_000},
    "O5": {"mass": 60.0, "temp": 40000, "lum": 300_000},
    "B0": {"mass": 18.0, "temp": 30000, "lum": 20_000},
    "B5": {"mass": 5.0, "temp": 15000, "lum": 800},
    "A0": {"mass": 2.2, "temp": 10000, "lum": 80},
    "A5": {"mass": 1.8, "temp": 8000, "lum": 20},
    "F0": {"mass": 1.5, "temp": 7500, "lum": 6.0},
    "F5": {"mass": 1.3, "temp": 6500, "lum": 3.0},
    "G0": {"mass": 1.1, "temp": 6000, "lum": 1.4},
    "G5": {"mass": 0.9, "temp": 5600, "lum": 0.8},
    "K0": {"mass": 0.8, "temp": 5200, "lum": 0.45},
    "K5": {"mass": 0.7, "temp": 4400, "lum": 0.15},
    "M0": {"mass": 0.5, "temp": 3800, "lum": 0.07},
    "M5": {"mass": 0.2, "temp": 3100, "lum": 0.007},
    "M9": {"mass": 0.08, "temp": 2400, "lum": 0.0008},
}

BROWN_DWARFS = {
    "L0": {"mass": 0.080, "temp": 2400, "lum": 0.00029},
    "L5": {"mass": 0.060, "temp": 1850, "lum": 0.000066},
    "T0": {"mass": 0.050, "temp": 1300, "lum": 0.000020},
    "T5": {"mass": 0.040, "temp": 900, "lum": 0.0000070},
    "Y0": {"mass": 0.025, "temp": 550, "lum": 0.00000081},
    "Y5": {"mass": 0.013, "temp": 300, "lum": 0.000000072},
}

WHITE_DWARFS = {
    "D0": {"mass": 1.20, "temp": 40000, "lum": 0.10},
    "D5": {"mass": 0.90, "temp": 12000, "lum": 0.01},
    "D9": {"mass": 0.60, "temp": 5000, "lum": 0.0005},
}

SUBTYPE_NUMERIC = {2: 0, 3: 1, 4: 3, 5: 5, 6: 7, 7: 9, 8: 8, 9: 6, 10: 4, 11: 2, 12: 0}
SUBTYPE_M_PRIMARY = {2: 8, 3: 6, 4: 5, 5: 4, 6: 0, 7: 2, 8: 1, 9: 3, 10: 5, 11: 7, 12: 9}
SPECIAL_PRIMARY_TYPES = ["Protostar", "Nebula", "Star Cluster", "Neutron Star", "Black Hole", "Anomaly"]


# =============================================================================
# Data classes
# =============================================================================
@dataclass
class Star:
    role: str
    spectral_type: str
    subtype: int
    luminosity_class: str
    mass_sol: float
    temp_k: int
    luminosity_sol: float
    semi_major_axis_au: Optional[float] = None
    eccentricity: Optional[float] = None
    parent_role: Optional[str] = None
    notes: str = ""

    @property
    def code(self) -> str:
        if self.spectral_type == "BD":
            return f"BD {self.luminosity_class}{self.subtype}"
        if self.spectral_type == "D":
            return f"D{self.subtype}"
        if self.spectral_type in SPECIAL_PRIMARY_TYPES:
            return self.spectral_type
        return f"{self.spectral_type}{self.subtype} {self.luminosity_class}".strip()


@dataclass
class Moon:
    name: str
    orbit_km: int
    size: int
    atmosphere: int
    hydrographics: int
    temperature_k: int
    habitability: int
    planet_class: str
    potentially_habitable: bool = False
    possible_mainworld: bool = False
    notes: str = ""


@dataclass
class OrbitingWorld:
    orbit_au: float
    zone: str
    world_type: str
    size: int
    atmosphere: int
    hydrographics: int
    population: int = 0
    government: int = 0
    law: int = 0
    tech_level: int = 0
    starport: str = "X"
    trade_codes: List[str] = field(default_factory=list)
    uwp: str = ""
    temperature_k: int = 0
    habitability: int = 0
    primary_role: str = "Primary"
    planet_class: str = ""
    economics: str = ""
    history: str = ""
    description: str = ""
    atmosphere_pressure_bar: float = 0.0
    thermal_band: str = ""
    seasonal_variation_k: int = 0
    tidally_locked: bool = False
    day_temp_k: int = 0
    night_temp_k: int = 0
    moon_mainworld_candidate: Optional[str] = None
    moons: List[Moon] = field(default_factory=list)
    notes: str = ""


@dataclass
class GeneratedSystem:
    hex_code: str
    name: str
    stars: List[Star]
    worlds: List[OrbitingWorld]
    habitable_zones: Dict[str, Dict[str, float]]
    detectability: str
    detectability_flux_at_1pc: float
    mainworld_index: Optional[int]
    pbg: str
    remarks: str
    importance: int
    seed_used: str
    subsector_seed: str
    system_seed: str
    empty_system: bool = False
    empty_reason: str = ""


# =============================================================================
# Stellar generation helpers
# =============================================================================
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_sequence(spectral_type: str, subtype: int) -> Dict[str, float]:
    anchors = {
        "O": [(0, STAR_MAIN_SEQUENCE["O0"]), (5, STAR_MAIN_SEQUENCE["O5"])],
        "B": [(0, STAR_MAIN_SEQUENCE["B0"]), (5, STAR_MAIN_SEQUENCE["B5"])],
        "A": [(0, STAR_MAIN_SEQUENCE["A0"]), (5, STAR_MAIN_SEQUENCE["A5"] )],
        "F": [(0, STAR_MAIN_SEQUENCE["F0"]), (5, STAR_MAIN_SEQUENCE["F5"])],
        "G": [(0, STAR_MAIN_SEQUENCE["G0"]), (5, STAR_MAIN_SEQUENCE["G5"]), (10, STAR_MAIN_SEQUENCE["K0"])],
        "K": [(0, STAR_MAIN_SEQUENCE["K0"]), (5, STAR_MAIN_SEQUENCE["K5"]), (10, STAR_MAIN_SEQUENCE["M0"])],
        "M": [(0, STAR_MAIN_SEQUENCE["M0"]), (5, STAR_MAIN_SEQUENCE["M5"]), (9, STAR_MAIN_SEQUENCE["M9"])],
    }
    points = anchors[spectral_type]
    for i in range(len(points) - 1):
        s0, p0 = points[i]
        s1, p1 = points[i + 1]
        if s0 <= subtype <= s1:
            t = 0.0 if s1 == s0 else (subtype - s0) / (s1 - s0)
            return {
                "mass": lerp(p0["mass"], p1["mass"], t),
                "temp": int(lerp(p0["temp"], p1["temp"], t)),
                "lum": lerp(p0["lum"], p1["lum"], t),
            }
    last = points[-1][1]
    return {"mass": last["mass"], "temp": last["temp"], "lum": last["lum"]}


def generate_primary_type(allow_dim_primaries: bool, allow_special_cases: bool) -> Tuple[str, str]:
    if allow_special_cases and roll_2d6() == 2:
        special = weighted_choice([
            ("Protostar", 20),
            ("Nebula", 20),
            ("Star Cluster", 15),
            ("Neutron Star", 15),
            ("Black Hole", 15),
            ("Anomaly", 15),
        ])
        return special, special
    r = roll_2d6()
    if r == 2:
        if allow_dim_primaries:
            return random.choice([("BD", "L"), ("D", "D"), ("M", "VI"), ("K", "IV")])
        return ("M", "VI")
    if r in (3, 4, 5, 6):
        return ("M", "V")
    if r in (7, 8):
        return ("K", "V")
    if r in (9, 10):
        return ("G", "V")
    if r == 11:
        return ("F", "V")
    hot = roll_2d6()
    if hot <= 8:
        return ("A", "V")
    if hot <= 11:
        return ("B", "V")
    return ("O", "V")


def generate_subtype(spectral_type: str, primary: bool = False) -> int:
    r = roll_2d6()
    if spectral_type == "M" and primary:
        return SUBTYPE_M_PRIMARY[r]
    return SUBTYPE_NUMERIC[r]


def variance_factor(scale: float = 0.2) -> float:
    return 1.0 + ((roll_2d6() - 7) / 5.0) * scale


def generate_named_brown_dwarf(role: str) -> Star:
    code = random.choice(list(BROWN_DWARFS.keys()))
    d = BROWN_DWARFS[code]
    return Star(role=role, spectral_type="BD", subtype=int(code[1]), luminosity_class=code[0], mass_sol=round(d["mass"], 3), temp_k=d["temp"], luminosity_sol=round(d["lum"], 8), notes=f"Brown dwarf {code}")


def generate_named_white_dwarf(role: str) -> Star:
    code = random.choice(list(WHITE_DWARFS.keys()))
    d = WHITE_DWARFS[code]
    return Star(role=role, spectral_type="D", subtype=int(code[1]), luminosity_class="D", mass_sol=round(d["mass"], 3), temp_k=d["temp"], luminosity_sol=round(d["lum"], 8), notes="White dwarf")


def generate_special_star(role: str, special_type: str) -> Star:
    if special_type == "Protostar":
        return Star(role=role, spectral_type="Protostar", subtype=0, luminosity_class="P", mass_sol=round(random.uniform(0.2, 3.0), 3), temp_k=random.randint(2200, 6000), luminosity_sol=round(random.uniform(0.01, 5.0), 8), notes="Young unstable protostar with incomplete planetary formation")
    if special_type == "Nebula":
        return Star(role=role, spectral_type="Nebula", subtype=0, luminosity_class="N", mass_sol=0.0, temp_k=random.randint(20, 300), luminosity_sol=round(random.uniform(0.0, 0.001), 8), notes="Dense nebular region; visibility and navigation degraded")
    if special_type == "Star Cluster":
        return Star(role=role, spectral_type="Star Cluster", subtype=0, luminosity_class="C", mass_sol=round(random.uniform(5.0, 50.0), 3), temp_k=6000, luminosity_sol=round(random.uniform(5.0, 500.0), 8), notes="Cluster core abstracted as a single local phenomenon")
    if special_type == "Neutron Star":
        return Star(role=role, spectral_type="Neutron Star", subtype=0, luminosity_class="NS", mass_sol=round(random.uniform(1.1, 2.2), 3), temp_k=random.randint(100000, 1000000), luminosity_sol=round(random.uniform(0.0001, 0.01), 8), notes="Compact remnant; hazardous radiation possible")
    if special_type == "Black Hole":
        return Star(role=role, spectral_type="Black Hole", subtype=0, luminosity_class="BH", mass_sol=round(random.uniform(3.0, 30.0), 3), temp_k=0, luminosity_sol=round(random.uniform(0.0, 0.00001), 8), notes="Dark compact object; only accretion effects are visible")
    return Star(role=role, spectral_type="Anomaly", subtype=0, luminosity_class="X", mass_sol=round(random.uniform(0.0, 10.0), 3), temp_k=random.randint(0, 20000), luminosity_sol=round(random.uniform(0.0, 0.1), 8), notes="Unclassified anomaly")


def generate_standard_star(role: str, spectral_type: str, lum_class: str, primary: bool, with_variance: bool) -> Star:
    subtype = generate_subtype(spectral_type, primary=primary)
    base = interpolate_sequence(spectral_type, subtype)
    mass = base["mass"]
    temp = base["temp"]
    lum = base["lum"]
    if lum_class == "IV":
        mass *= 1.15
        lum *= 2.5
    elif lum_class == "VI":
        mass *= 0.9
        lum *= 0.4
    elif lum_class == "III":
        mass *= 1.4
        lum *= 40
    elif lum_class == "II":
        mass *= 1.8
        lum *= 200
    if with_variance:
        v = variance_factor(0.20)
        mass *= v
        lum *= max(0.25, v)
    return Star(role=role, spectral_type=spectral_type, subtype=subtype, luminosity_class=lum_class, mass_sol=round(mass, 3), temp_k=int(temp), luminosity_sol=round(lum, 8))


def generate_primary_star(allow_dim_primaries: bool, with_variance: bool, allow_special_cases: bool) -> Star:
    spectral_type, lum_class = generate_primary_type(allow_dim_primaries, allow_special_cases)
    if spectral_type in SPECIAL_PRIMARY_TYPES:
        return generate_special_star("Primary", spectral_type)
    if spectral_type == "BD":
        return generate_named_brown_dwarf("Primary")
    if spectral_type == "D":
        return generate_named_white_dwarf("Primary")
    return generate_standard_star("Primary", spectral_type, lum_class, True, with_variance)


def determine_multiplicity(empty_system: bool = False) -> int:
    if empty_system:
        return 1 if roll_2d6() <= 10 else 2
    r = roll_2d6()
    if r <= 6:
        return 1
    if r <= 10:
        return 2
    return 3


def companion_type_from_primary(primary: Star, with_variance: bool, role: str, allow_special_cases: bool) -> Star:
    if allow_special_cases and roll_2d6() == 2:
        return generate_special_star(role, weighted_choice([("Neutron Star", 30), ("Black Hole", 20), ("Anomaly", 20), ("Protostar", 15), ("Nebula", 15)]))
    options = [("M", 30), ("K", 24), ("G", 16), ("F", 10), ("BD", 10), ("D", 4), ("A", 4), ("MVI", 2)]
    pick = weighted_choice(options)
    if pick == "BD":
        return generate_named_brown_dwarf(role)
    if pick == "D":
        return generate_named_white_dwarf(role)
    if pick == "MVI":
        return generate_standard_star(role, "M", "VI", False, with_variance)
    return generate_standard_star(role, pick, "V", False, with_variance)


def generate_stellar_orbit() -> Tuple[float, float, str]:
    band = weighted_choice([("Close", 25), ("Moderate", 30), ("Far", 30), ("Distant", 15)])
    if band == "Close":
        a = round(random.uniform(0.05, 0.5), 3)
    elif band == "Moderate":
        a = round(random.uniform(0.6, 6.0), 3)
    elif band == "Far":
        a = round(random.uniform(8.0, 50.0), 3)
    else:
        a = round(random.uniform(60.0, 400.0), 3)
    e = round(random.uniform(0.0, 0.6), 2)
    return a, e, band


def generate_star_system(allow_dim_primaries: bool, with_variance: bool, allow_special_cases: bool, empty_system: bool) -> List[Star]:
    primary = generate_primary_star(allow_dim_primaries or empty_system, with_variance, allow_special_cases)
    stars = [primary]
    multiplicity = determine_multiplicity(empty_system)
    if multiplicity >= 2:
        sec = companion_type_from_primary(primary, with_variance, "Secondary", allow_special_cases)
        a, e, band = generate_stellar_orbit()
        sec.semi_major_axis_au = a
        sec.eccentricity = e
        sec.parent_role = "Primary"
        sec.notes = f"{band} companion"
        stars.append(sec)
    if multiplicity == 3 and not empty_system:
        tert = companion_type_from_primary(primary, with_variance, "Tertiary", allow_special_cases)
        parent_role = random.choice(["Primary", "Secondary"])
        a, e, band = generate_stellar_orbit()
        if parent_role == "Secondary" and len(stars) > 1:
            a = round(max(0.02, a / 10.0), 3)
        tert.semi_major_axis_au = a
        tert.eccentricity = e
        tert.parent_role = parent_role
        tert.notes = f"{band} companion of {parent_role}"
        stars.append(tert)
    return stars


# =============================================================================
# Habitable zone and detectability
# =============================================================================
def pair_luminosity(stars: List[Star], roles: List[str]) -> float:
    return sum(s.luminosity_sol for s in stars if s.role in roles)


def habitable_zone(luminosity_sol: float) -> Dict[str, float]:
    inner = safe_sqrt(luminosity_sol / 1.1)
    center = safe_sqrt(max(luminosity_sol, 1e-9))
    outer = safe_sqrt(luminosity_sol / 0.53)
    return {"inner": round(inner, 3), "center": round(center, 3), "outer": round(outer, 3)}


def get_habitable_zones(stars: List[Star]) -> Dict[str, Dict[str, float]]:
    zones: Dict[str, Dict[str, float]] = {}
    primary_lum = pair_luminosity(stars, ["Primary"])
    if primary_lum > 0:
        zones["Primary"] = habitable_zone(primary_lum)
    sec = next((s for s in stars if s.role == "Secondary"), None)
    tert = next((s for s in stars if s.role == "Tertiary"), None)
    if sec and sec.semi_major_axis_au and sec.semi_major_axis_au >= 8 and sec.luminosity_sol > 0:
        zones["Secondary"] = habitable_zone(sec.luminosity_sol)
    if tert and tert.parent_role == "Secondary" and tert.semi_major_axis_au and tert.semi_major_axis_au >= 0.2 and tert.luminosity_sol > 0:
        zones["Tertiary"] = habitable_zone(tert.luminosity_sol)
    elif tert and tert.parent_role == "Primary" and tert.semi_major_axis_au and tert.semi_major_axis_au >= 8 and tert.luminosity_sol > 0:
        zones["Tertiary"] = habitable_zone(tert.luminosity_sol)
    if sec and sec.parent_role == "Primary" and sec.semi_major_axis_au and sec.semi_major_axis_au <= 1.5:
        pair_lum = pair_luminosity(stars, ["Primary", "Secondary"])
        if pair_lum > 0:
            zones["Primary+Secondary"] = habitable_zone(pair_lum)
    return zones


def detectability(stars: List[Star], distance_pc: float = 1.0) -> Tuple[str, float]:
    total_lum = sum(s.luminosity_sol for s in stars)
    flux = parsec_flux_relative(total_lum, distance_pc)
    if flux < 0.0005:
        return "Não detectável a 1 parsec em survey comum", flux
    if flux < 0.001:
        return "Muito difícil de detectar fora do próprio sistema", flux
    if flux < 0.01:
        return "Difícil de detectar a longa distância", flux
    return "Detectável em levantamentos estelares normais", flux


# =============================================================================
# World generation
# =============================================================================
def base_world_counts(luminous: float, empty_system: bool = False, special_primary: bool = False) -> Tuple[int, int, int]:
    if empty_system:
        chance = roll_2d6()
        if chance <= 9:
            return 0, 0, 0
        if chance <= 11:
            return 0, 0, 1
        return 0, 1, 0
    if special_primary:
        chance = roll_2d6()
        if chance <= 6:
            return 0, 0, 0
        if chance <= 9:
            return 0, 1, 0
        return 0, 0, 1
    gg = 0 if roll_2d6() >= 10 else random.randint(1, 4)
    pb = random.randint(0, 2) if roll_2d6() >= 8 else random.randint(0, 1)
    tp_dm = 1 if luminous >= 0.1 else -1
    tp = max(0, roll_2d6(tp_dm) - 2)
    return gg, pb, tp


def generate_orbit_slots(hz_outer: float, role: str) -> List[float]:
    if hz_outer <= 0:
        hz_outer = 0.1
    inner = max(0.05, hz_outer * 0.12)
    current = inner
    slots: List[float] = []
    for _ in range(14):
        slots.append(round(current, 3))
        current *= random.uniform(1.45, 1.85)
    if role != "Primary":
        slots = [round(s, 3) for s in slots if s <= 20.0]
    return sorted(set(slots))


def generate_size(zone: str, world_type: str) -> int:
    if world_type == "Planetoid Belt":
        return 0
    if world_type == "Gas Giant":
        return random.randint(12, 15)
    base = roll_2d6() - 2
    if zone == "Inner":
        base += random.choice([-2, -1, 0, 1])
    elif zone == "Outer":
        base += random.choice([-1, 0, 1, 2])
    return clamp(base, 0, 10)


def generate_atmosphere(size: int, zone: str, world_type: str) -> int:
    if world_type in ("Planetoid Belt", "Gas Giant"):
        return 0 if world_type == "Planetoid Belt" else 15
    if size == 0:
        return 0
    a = size + roll_2d6() - 7
    if zone == "Inner":
        a -= random.randint(0, 3)
    if zone == "Outer":
        a -= random.randint(0, 2)
    return clamp(a, 0, 15)


def generate_hydro(size: int, atmosphere: int, zone: str, world_type: str) -> int:
    if world_type in ("Planetoid Belt", "Gas Giant"):
        return 0 if world_type == "Planetoid Belt" else 10
    if size <= 1:
        return 0
    h = roll_2d6() - 2
    if atmosphere in (0, 1, 10, 11, 12):
        h -= 4
    if atmosphere >= 13:
        h -= 2
    if zone == "Inner":
        h -= random.randint(1, 5)
    elif zone == "Outer":
        h -= random.randint(0, 3)
    else:
        h += random.randint(0, 2)
    return clamp(h, 0, 10)


def equilibrium_temperature_k(luminosity_sol: float, orbit_au: float, atmosphere: int) -> int:
    if orbit_au <= 0:
        orbit_au = 0.01
    base = 278 * ((max(luminosity_sol, 1e-9) ** 0.25) / safe_sqrt(orbit_au))
    greenhouse = 0
    if 4 <= atmosphere <= 9:
        greenhouse = 12
    elif atmosphere in (10, 13, 14, 15):
        greenhouse = 35
    elif atmosphere in (11, 12):
        greenhouse = -10
    return int(round(base + greenhouse))


def habitability_score(size: int, atmosphere: int, hydro: int, temperature_k: int) -> int:
    score = 10
    if size < 4:
        score -= 2
    if atmosphere in (0, 1, 2, 3, 10, 11, 12, 13, 14, 15):
        score -= 3
    if hydro == 0:
        score -= 2
    elif hydro >= 3:
        score += 1
    if temperature_k < 240:
        score -= 2
    elif temperature_k > 323:
        score -= 2
    return clamp(score, 0, 12)


def choose_mainworld(worlds: List[OrbitingWorld]) -> Optional[int]:
    best_i = None
    best_score = -999
    for i, w in enumerate(worlds):
        if w.world_type != "Terrestrial":
            continue
        score = w.habitability
        if 4 <= w.atmosphere <= 9:
            score += 2
        if w.hydrographics >= 2:
            score += 1
        if 4 <= w.size <= 9:
            score += 1
        if w.moon_mainworld_candidate:
            score += 1
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def population_from_world(w: OrbitingWorld) -> int:
    dm = 0
    if w.habitability >= 8:
        dm += 3
    elif w.habitability >= 6:
        dm += 1
    elif w.habitability <= 2:
        dm -= 4
    if w.zone == "Outer":
        dm -= 1
    return clamp(roll_2d6(dm) - 2, 0, 12)


def government_from_population(population: int) -> int:
    if population == 0:
        return 0
    return clamp(population + roll_2d6() - 7, 0, 15)


def law_from_government(government: int) -> int:
    return clamp(government + roll_2d6() - 7, 0, 15)


def tech_level(starport: str, size: int, atmosphere: int, hydro: int, population: int, government: int) -> int:
    tl = roll_d6()
    tl += {"A": 6, "B": 4, "C": 2, "D": 0, "E": -2, "X": -4}.get(starport, 0)
    if size <= 1:
        tl += 2
    elif 2 <= size <= 4:
        tl += 1
    if atmosphere <= 3 or atmosphere >= 10:
        tl += 1
    if hydro == 9:
        tl += 1
    elif hydro == 10:
        tl += 2
    if 1 <= population <= 5:
        tl += 1
    elif population == 9:
        tl += 2
    elif population >= 10:
        tl += 4
    if government in (0, 5):
        tl += 1
    elif government == 13:
        tl -= 2
    return clamp(tl, 0, 16)


def starport_from_population(population: int) -> str:
    if population == 0:
        return "X"
    r = roll_2d6() + min(population, 6)
    for threshold, starport in STARPORT_TABLE:
        if r <= threshold:
            return starport
    return "A"


def tl_of(w: OrbitingWorld) -> int:
    return w.tech_level


def trade_codes(w: OrbitingWorld) -> List[str]:
    codes: List[str] = []
    atmo = w.atmosphere
    hydro = w.hydrographics
    pop = w.population
    gov = w.government
    if 4 <= atmo <= 9 and 4 <= hydro <= 8 and 5 <= pop <= 7:
        codes.append("Ag")
    if w.size == 0 and atmo == 0 and hydro == 0:
        codes.append("As")
    if pop == 0 and gov == 0 and w.law == 0:
        codes.append("Ba")
    if atmo >= 2 and hydro == 0:
        codes.append("De")
    if atmo in (10, 11, 12) and hydro >= 1:
        codes.append("Fl")
    if w.size >= 10 and 1 <= atmo <= 9 and hydro >= 1:
        codes.append("Ht")
    if pop >= 9:
        codes.append("Hi")
    if tl_of(w) >= 12:
        codes.append("Htch")
    if atmo in (0, 1) and hydro >= 1:
        codes.append("Ic")
    if atmo in (0, 1, 2, 4, 7, 9) and pop >= 9:
        codes.append("In")
    if 4 <= atmo <= 9 and 4 <= pop <= 6 and gov in (4, 5, 6, 7, 8, 9):
        codes.append("Ni")
    if pop <= 3:
        codes.append("Lo")
    if atmo <= 3 and hydro <= 3 and pop >= 6:
        codes.append("Na")
    if 6 <= atmo <= 8 and 6 <= pop <= 8 and 4 <= gov <= 9:
        codes.append("Ri")
    if atmo == 0:
        codes.append("Va")
    if hydro == 10:
        codes.append("Wa")
    return sorted(set(codes))


def pressure_from_atmo(atmo: int) -> float:
    if atmo == 0:
        return 0.0
    if atmo == 1:
        return round(random.uniform(0.05, 0.2), 2)
    if atmo in (2, 3):
        return round(random.uniform(0.2, 0.5), 2)
    if 4 <= atmo <= 9:
        return round(random.uniform(0.6, 1.8), 2)
    if atmo in (10, 11, 12):
        return round(random.uniform(0.8, 2.5), 2)
    return round(random.uniform(1.5, 8.0), 2)


def classify_thermal_band(temp_k: int) -> str:
    if temp_k < 230:
        return "Frozen"
    if temp_k < 260:
        return "Cold"
    if temp_k < 310:
        return "Temperate"
    if temp_k < 340:
        return "Warm"
    return "Hot"


def tidal_lock_chance(orbit_au: float, size: int, primary_role: str) -> bool:
    lock_threshold = 0.15 if primary_role == "Primary" else 0.2
    mod = 0.05 if size >= 8 else 0.0
    return orbit_au <= (lock_threshold + mod) and random.random() < 0.7


def climate_profile(temp_k: int, atmo: int, orbit_au: float, size: int, primary_role: str) -> Tuple[str, int, bool, int, int]:
    band = classify_thermal_band(temp_k)
    seasonal = random.randint(4, 12)
    if band in ("Cold", "Frozen", "Hot"):
        seasonal += random.randint(2, 12)
    locked = tidal_lock_chance(orbit_au, size, primary_role)
    if locked:
        day_temp = temp_k + random.randint(20, 70)
        night_temp = max(20, temp_k - random.randint(40, 120))
    else:
        diurnal = random.randint(4, 18)
        if atmo <= 1:
            diurnal += random.randint(20, 80)
        elif atmo >= 10:
            diurnal -= random.randint(0, 6)
        day_temp = temp_k + max(2, diurnal)
        night_temp = max(20, temp_k - max(2, diurnal))
    return band, seasonal, locked, day_temp, night_temp


def classify_planet(w: OrbitingWorld) -> str:
    if w.world_type == "Gas Giant":
        return random.choice(["Gas Giant", "Large Gas Giant", "Ice Giant"])
    if w.world_type == "Planetoid Belt":
        return random.choice(["Rocky Belt", "Ice Belt", "Metal-Rich Belt"])
    if w.size == 0 and w.atmosphere == 0:
        return "Airless Rock"
    if w.hydrographics == 10:
        return "Ocean World"
    if w.hydrographics == 0 and w.atmosphere >= 2:
        return "Desert World"
    if w.temperature_k < 240:
        return "Ice World"
    if w.temperature_k > 340:
        return "Hot World"
    if w.size >= 9 and w.atmosphere >= 8:
        return "Superterran"
    if w.zone == "Habitable" and 4 <= w.atmosphere <= 9 and w.hydrographics >= 3:
        return "Temperate Terrestrial"
    return "Rocky World"


def classify_moon(size: int, atmo: int, hydro: int, temp_k: int) -> str:
    if hydro == 10:
        return "Oceanic Moon"
    if temp_k < 230:
        return "Icy Moon"
    if atmo == 0:
        return "Airless Moon"
    if hydro == 0 and atmo >= 2:
        return "Dry Rocky Moon"
    if 4 <= atmo <= 9 and hydro >= 2:
        return "Temperate Moon"
    return "Rocky Moon"


def generate_detailed_moons(w: OrbitingWorld) -> List[Moon]:
    moons: List[Moon] = []
    if w.world_type == "Gas Giant":
        count = random.randint(6, 16)
    elif w.world_type == "Terrestrial" and w.size >= 6 and roll_2d6() >= 8:
        count = 1 if roll_2d6() <= 10 else 2
    else:
        count = 0
    for i in range(count):
        size = clamp(random.randint(0, max(1, min(8, w.size - 1 if w.world_type == "Terrestrial" else 8))), 0, 8)
        atmo = 0 if size == 0 else clamp(size + roll_2d6() - 8, 0, 12)
        hydro = 0 if size <= 1 else clamp(roll_2d6() - 5 + (2 if w.zone == "Habitable" else 0), 0, 10)
        temp = max(20, w.temperature_k + random.randint(-20, 20))
        habit = habitability_score(size, atmo, hydro, temp)
        pclass = classify_moon(size, atmo, hydro, temp)
        hab = size >= 4 and 4 <= atmo <= 9 and hydro >= 2 and 250 <= temp <= 320
        main_candidate = hab and roll_2d6() >= 9
        if w.world_type == "Gas Giant":
            orbit_km = random.randint(100000, 3000000)
        else:
            orbit_km = random.randint(50000, 600000)
        moons.append(Moon(name=f"{i+1}", orbit_km=orbit_km, size=size, atmosphere=atmo, hydrographics=hydro, temperature_k=temp, habitability=habit, planet_class=pclass, potentially_habitable=hab, possible_mainworld=main_candidate, notes=""))
    return moons


def world_economics(w: OrbitingWorld) -> str:
    if w.population == 0:
        return "No formal economy"
    if "In" in w.trade_codes:
        return "Industrial exports, manufactured goods, machinery"
    if "Ag" in w.trade_codes:
        return "Agricultural exports, foodstuffs, bio-products"
    if "Ri" in w.trade_codes:
        return "Diversified economy, finance, high-value trade"
    if "As" in w.trade_codes:
        return "Mining and prospecting"
    if "De" in w.trade_codes or "Na" in w.trade_codes:
        return "Resource extraction and imported essentials"
    if "Lo" in w.trade_codes:
        return "Local subsistence and limited trade"
    return "Mixed local economy"


def world_history(w: OrbitingWorld) -> str:
    if w.population == 0:
        return random.choice(["Uninhabited and largely unexplored", "Surveyed but never settled", "Abandoned outpost traces remain"])
    return random.choice([
        "former colony project that survived early hardship",
        "trade outpost that expanded into a regional hub",
        "resource settlement shaped by offworld investors",
        "frontier world with periodic unrest",
        "old colony with strong local identity",
        "strategic stopover shaped by its starport",
    ])


def atmosphere_description(atmo: int, pressure: float) -> str:
    if atmo == 0:
        return "airless vacuum"
    if atmo in (1, 2, 3):
        return f"thin atmosphere at about {pressure} bar"
    if 4 <= atmo <= 9:
        return f"breathable-range atmosphere around {pressure} bar"
    if atmo in (10, 11, 12):
        return f"exotic tainted atmosphere around {pressure} bar"
    return f"dense hostile atmosphere around {pressure} bar"


def sky_description(w: OrbitingWorld, system: GeneratedSystem) -> str:
    extra = len(system.stars) - 1
    if extra <= 0:
        base = "The sky is dominated by a single primary sun"
    elif extra == 1:
        base = "The sky shows a primary sun and a companion star that marks the heavens"
    else:
        base = "Multiple suns or stellar companions make the sky visually complex"
    if any(s.spectral_type == "Nebula" for s in system.stars):
        base += ", with diffuse nebular light staining the dark"
    if any(s.spectral_type == "Star Cluster" for s in system.stars):
        base += ", and dense stellar fields are visible overhead"
    if w.tidally_locked:
        base += "; one hemisphere endures prolonged daylight while the other fades into deep night"
    return base + "."


def settlement_description(w: OrbitingWorld) -> str:
    if w.population == 0:
        return "No permanent settlements are confirmed."
    if w.population <= 3:
        return "Settlements are sparse, scattered, and heavily dependent on imported infrastructure."
    if w.population <= 6:
        return "A modest network of towns, stations, or domed settlements supports local life."
    if w.population <= 8:
        return "Large cities and major ports dominate the settled regions."
    return "Dense urbanization, orbital traffic, and extensive infrastructure define the inhabited zones."


def adventure_hooks(w: OrbitingWorld) -> str:
    options = []
    if w.population == 0:
        options.extend(["survey expedition gone missing", "derelict station signal", "sealed research cache"])
    if w.tidally_locked:
        options.append("conflict over the twilight settlement belt")
    if w.moon_mainworld_candidate:
        options.append("habitable moon with rival colonial claims")
    if "As" in w.trade_codes:
        options.append("violent dispute over mineral rights")
    if "Ri" in w.trade_codes:
        options.append("elite political scandal with offworld consequences")
    if "In" in w.trade_codes:
        options.append("industrial sabotage on a strategic line")
    if not options:
        options = ["smuggling corridor hidden in local traffic", "ancient survey anomaly in the wilderness", "offworld pressure on local authorities"]
    return random.choice(options)


def world_description(w: OrbitingWorld, system: GeneratedSystem) -> str:
    moon_text = f" It has {len(w.moons)} moon(s)." if w.moons else ""
    moon_candidate = f" A moon may rival the planet as the best settlement site: {w.moon_mainworld_candidate}." if w.moon_mainworld_candidate else ""
    return (
        f"{w.planet_class} in the {w.zone.lower()} zone, orbiting {w.primary_role}. "
        f"Thermal class {w.thermal_band.lower()}, mean temperature {w.temperature_k}K, day/night range {w.night_temp_k}-{w.day_temp_k}K. "
        f"It has {atmosphere_description(w.atmosphere, w.atmosphere_pressure_bar)} and hydrographics {w.hydrographics}. "
        f"{sky_description(w, system)} {settlement_description(w)} "
        f"Its recent history suggests a {w.history}. A likely adventure hook is {adventure_hooks(w)}.{moon_text}{moon_candidate}"
    )


def build_uwp(w: OrbitingWorld) -> str:
    return f"{w.starport}{to_ehex(w.size, True)}{to_ehex(w.atmosphere, True)}{to_ehex(w.hydrographics, True)}{to_ehex(w.population, True)}{to_ehex(w.government, True)}{to_ehex(w.law, True)}-{to_ehex(w.tech_level, True)}"


def generate_worlds_for_role(role: str, stars: List[Star], hz: Dict[str, float], empty_system: bool = False) -> List[OrbitingWorld]:
    role_lum = pair_luminosity(stars, role.split("+") if "+" in role else [role])
    special_primary = any(s.role == "Primary" and s.spectral_type in SPECIAL_PRIMARY_TYPES for s in stars)
    gg, pb, tp = base_world_counts(role_lum, empty_system=empty_system, special_primary=special_primary)
    count = gg + pb + tp
    if count <= 0:
        return []
    slots = generate_orbit_slots(hz["outer"], role)
    count = min(count, len(slots))
    chosen = sorted(random.sample(slots, count))
    world_tags = (["Gas Giant"] * min(gg, count)) + (["Planetoid Belt"] * min(pb, max(0, count - gg)))
    while len(world_tags) < count:
        world_tags.append("Terrestrial")
    random.shuffle(world_tags)
    worlds: List[OrbitingWorld] = []
    for orbit, tag in zip(chosen, world_tags):
        zone = zone_label(orbit, hz["inner"], hz["outer"])
        size = generate_size(zone, tag)
        atmo = generate_atmosphere(size, zone, tag)
        hydro = generate_hydro(size, atmo, zone, tag)
        temp = equilibrium_temperature_k(role_lum, orbit, atmo)
        habit = habitability_score(size, atmo, hydro, temp)
        worlds.append(OrbitingWorld(orbit_au=orbit, zone=zone, world_type=tag, size=size, atmosphere=atmo, hydrographics=hydro, temperature_k=temp, habitability=habit, primary_role=role))
    return worlds


def enrich_world_data(system: GeneratedSystem) -> None:
    for i, world in enumerate(system.worlds):
        if world.world_type == "Gas Giant":
            world.population = 0
            world.government = 0
            world.law = 0
            world.starport = "X"
            world.tech_level = 0
        elif world.world_type == "Planetoid Belt":
            base_pop = clamp(population_from_world(world) - 2, 0, 10)
            world.population = base_pop
            world.government = government_from_population(world.population)
            world.law = law_from_government(world.government)
            world.starport = starport_from_population(world.population)
            world.tech_level = tech_level(world.starport, world.size, world.atmosphere, world.hydrographics, world.population, world.government)
        else:
            world.population = population_from_world(world)
            if system.mainworld_index is not None and i == system.mainworld_index:
                world.population = max(world.population, random.randint(4, 9))
            world.government = government_from_population(world.population)
            world.law = law_from_government(world.government)
            world.starport = starport_from_population(world.population)
            world.tech_level = tech_level(world.starport, world.size, world.atmosphere, world.hydrographics, world.population, world.government)
        world.trade_codes = trade_codes(world)
        world.uwp = build_uwp(world)
        world.atmosphere_pressure_bar = pressure_from_atmo(world.atmosphere)
        band, seasonal, locked, day_temp, night_temp = climate_profile(world.temperature_k, world.atmosphere, world.orbit_au, world.size, world.primary_role)
        world.thermal_band = band
        world.seasonal_variation_k = seasonal
        world.tidally_locked = locked
        world.day_temp_k = day_temp
        world.night_temp_k = night_temp
        world.planet_class = classify_planet(world)
        world.moons = generate_detailed_moons(world)
        candidates = [m.name for m in world.moons if m.possible_mainworld]
        world.moon_mainworld_candidate = candidates[0] if candidates else None
        world.economics = world_economics(world)
        world.history = world_history(world)
        world.description = world_description(world, system)


# =============================================================================
# System-level builders
# =============================================================================
def importance_score(world: Optional[OrbitingWorld]) -> int:
    if world is None:
        return -2
    score = 0
    if world.starport in ("A", "B"):
        score += 1
    if world.population >= 9:
        score += 1
    if world.tech_level >= 10:
        score += 1
    if "Ag" in world.trade_codes or "Ri" in world.trade_codes or "In" in world.trade_codes:
        score += 1
    if world.starport == "X":
        score -= 1
    if world.population <= 3:
        score -= 1
    return score


def build_pbg(worlds: List[OrbitingWorld], mainworld_index: Optional[int]) -> str:
    pop_mult = random.randint(1, 9) if mainworld_index is not None and worlds[mainworld_index].population > 0 else 0
    belts = sum(1 for w in worlds if w.world_type == "Planetoid Belt")
    gas_giants = sum(1 for w in worlds if w.world_type == "Gas Giant")
    return f"{pop_mult}{to_ehex(belts, True)}{to_ehex(gas_giants, True)}"


def build_remarks(world: Optional[OrbitingWorld]) -> str:
    if world is None:
        return "Ba"
    return " ".join(world.trade_codes)


def generate_single_system(hex_code: str, name: str, allow_dim_primaries: bool, with_variance: bool, allow_special_cases: bool, seed_used: str, empty_system: bool = False, subsector_seed: str = "", system_seed: Optional[str] = None) -> GeneratedSystem:
    if system_seed is None:
        system_seed = make_system_seed(subsector_seed or seed_used, hex_code)
    canonical_name = name.strip() if name.strip() else random_system_name(hex_code, "", system_seed)
    with with_temp_seed(system_seed):
        stars = generate_star_system(allow_dim_primaries, with_variance, allow_special_cases, empty_system)
        hz_map = get_habitable_zones(stars)
        worlds: List[OrbitingWorld] = []
        if hz_map:
            for role, hz in hz_map.items():
                worlds.extend(generate_worlds_for_role(role, stars, hz, empty_system=empty_system))
        if empty_system:
            worlds = [w for w in worlds if w.world_type in ("Planetoid Belt", "Terrestrial")][:1]
        mainworld_index = choose_mainworld(worlds)
        det_label, det_flux = detectability(stars, 1.0)
        system = GeneratedSystem(hex_code=hex_code, name=canonical_name, stars=stars, worlds=worlds, habitable_zones=hz_map, detectability=det_label, detectability_flux_at_1pc=det_flux, mainworld_index=mainworld_index, pbg="000", remarks="", importance=0, seed_used=seed_used, subsector_seed=subsector_seed or seed_used, system_seed=system_seed, empty_system=empty_system)
        if empty_system and det_flux >= 0.0005:
            system.empty_reason = "Marked as empty, but stellar output remains marginally detectable at 1 parsec."
        elif empty_system:
            system.empty_reason = "Intentionally empty or nearly empty system with no normal survey target."
        enrich_world_data(system)
        mw = system.worlds[system.mainworld_index] if system.mainworld_index is not None else None
        system.pbg = build_pbg(system.worlds, system.mainworld_index)
        system.remarks = build_remarks(mw)
        system.importance = importance_score(mw)
    return system



# =============================================================================
# Exports
# =============================================================================
def stars_string(stars: List[Star]) -> str:
    return " ".join(s.code for s in stars)


def traveller_map_record(system: GeneratedSystem) -> Dict[str, str]:
    if system.mainworld_index is None:
        uwp = "X000000-0"
        remarks = "Ba"
    else:
        mw = system.worlds[system.mainworld_index]
        uwp = mw.uwp
        remarks = system.remarks
    return {
        "Hex": system.hex_code,
        "Name": system.name,
        "UWP": uwp,
        "Remarks": remarks,
        "{Ix}": f"{{ {system.importance:+d} }}",
        "(Ex)": "(000+0)",
        "[Cx]": "[0000]",
        "Nobility": "-",
        "Bases": "S" if system.importance >= 1 else "-",
        "Zone": "A" if system.importance < 0 else "-",
        "PBG": system.pbg,
        "W": str(len(system.worlds)),
        "Allegiance": "Na",
        "Stars": stars_string(system.stars),
        "Seed": system.seed_used,
        "SystemSeed": system.system_seed,
        "SubsectorSeed": system.subsector_seed,
    }


def system_json(system: GeneratedSystem) -> Dict:
    return {
        "hex": system.hex_code,
        "name": system.name,
        "seed": system.seed_used,
        "system_seed": system.system_seed,
        "subsector_seed": system.subsector_seed,
        "empty_system": system.empty_system,
        "empty_reason": system.empty_reason,
        "detectability": system.detectability,
        "detectability_flux_at_1pc": system.detectability_flux_at_1pc,
        "habitable_zones": system.habitable_zones,
        "mainworld_index": system.mainworld_index,
        "pbg": system.pbg,
        "remarks": system.remarks,
        "importance": system.importance,
        "stars": [asdict(s) for s in system.stars],
        "worlds": [asdict(w) for w in system.worlds],
        "traveller_map_record": traveller_map_record(system),
    }


# =============================================================================
# Subsector generation
# =============================================================================
def subsector_hexes() -> List[str]:
    out = []
    for x in range(1, 9):
        for y in range(1, 11):
            out.append(f"{x:02d}{y:02d}")
    return out


def random_system_name(hex_code: str, prefix: str, seed: Optional[str] = None) -> str:
    syll_a = ["Ar", "Bel", "Cor", "Den", "Eri", "Fal", "Gor", "Hel", "Ira", "Jor", "Kel", "Lor", "Mor", "Nor", "Or", "Pra", "Qua", "Ryn", "Sel", "Tor", "Ula", "Vor", "Wen", "Xan", "Yor", "Zed"]
    syll_b = ["a", "e", "i", "o", "u", "ae", "ia", "or", "en", "ul"]
    syll_c = ["dor", "mar", "nus", "tis", "vek", "lon", "bar", "phi", "rax", "th", "mir", "zen"]
    if seed is None:
        return f"{prefix}{random.choice(syll_a)}{random.choice(syll_b)}{random.choice(syll_c)}-{hex_code}"
    rng = random.Random(seed)
    return f"{prefix}{rng.choice(syll_a)}{rng.choice(syll_b)}{rng.choice(syll_c)}-{hex_code}"


def generate_subsector(prefix: str, density_pct: int, allow_dim_primaries: bool, with_variance: bool, allow_special_cases: bool, seed_used: str, empty_system: bool) -> List[GeneratedSystem]:
    systems: List[GeneratedSystem] = []
    subsector_rng = random.Random(seed_used)
    for hex_code in subsector_hexes():
        if subsector_rng.randint(1, 100) <= density_pct:
            system_seed = make_system_seed(seed_used, hex_code)
            name = random_system_name(hex_code, prefix, system_seed)
            systems.append(generate_single_system(hex_code, name, allow_dim_primaries, with_variance, allow_special_cases, seed_used, empty_system=empty_system, subsector_seed=seed_used, system_seed=system_seed))
    return systems


# =============================================================================
# UI tables and plotting
# =============================================================================
def stars_df(stars: List[Star]) -> pd.DataFrame:
    rows = []
    for s in stars:
        rows.append({
            "Role": s.role,
            "Code": s.code,
            "Mass (Sol)": s.mass_sol,
            "Temp (K)": s.temp_k,
            "Luminosity (Sol)": s.luminosity_sol,
            "Axis (AU)": s.semi_major_axis_au,
            "Ecc": s.eccentricity,
            "Parent": s.parent_role,
            "Notes": s.notes,
        })
    return pd.DataFrame(rows)


def worlds_df(worlds: List[OrbitingWorld], mainworld_index: Optional[int]) -> pd.DataFrame:
    rows = []
    for i, w in enumerate(worlds):
        rows.append({
            "#": i + 1,
            "Main": "Yes" if i == mainworld_index else "",
            "Role": w.primary_role,
            "Orbit (AU)": w.orbit_au,
            "Zone": w.zone,
            "Type": w.world_type,
            "Class": w.planet_class,
            "Size": w.size,
            "Atmo": w.atmosphere,
            "Hydro": w.hydrographics,
            "Pressure (bar)": w.atmosphere_pressure_bar,
            "Temp (K)": w.temperature_k,
            "Day/Night": f"{w.night_temp_k}-{w.day_temp_k}",
            "Thermal": w.thermal_band,
            "Season ΔK": w.seasonal_variation_k,
            "Tidal Lock": w.tidally_locked,
            "Habitability": w.habitability,
            "Pop": w.population,
            "Gov": w.government,
            "Law": w.law,
            "TL": w.tech_level,
            "Starport": w.starport,
            "UWP": w.uwp,
            "Trade": " ".join(w.trade_codes),
            "Economy": w.economics,
            "Moons": len(w.moons),
            "Moon MW": w.moon_mainworld_candidate or "",
            "History": w.history,
            "Notes": w.notes,
        })
    return pd.DataFrame(rows)


def moons_df(worlds: List[OrbitingWorld]) -> pd.DataFrame:
    rows = []
    for wi, w in enumerate(worlds):
        for moon in w.moons:
            rows.append({
                "Parent #": wi + 1,
                "Moon": moon.name,
                "Orbit (km)": moon.orbit_km,
                "Class": moon.planet_class,
                "Size": moon.size,
                "Atmo": moon.atmosphere,
                "Hydro": moon.hydrographics,
                "Temp (K)": moon.temperature_k,
                "Habitability": moon.habitability,
                "Habitable": moon.potentially_habitable,
                "Mainworld Candidate": moon.possible_mainworld,
                "Notes": moon.notes,
            })
    return pd.DataFrame(rows)


def hz_df(hz_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"Role": role, "Inner AU": hz["inner"], "Center AU": hz["center"], "Outer AU": hz["outer"]} for role, hz in hz_map.items()])


def subsector_df(systems: List[GeneratedSystem]) -> pd.DataFrame:
    return pd.DataFrame([traveller_map_record(s) for s in systems])


def plot_system_map(system: GeneratedSystem):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_title(f"Orbital Map - {system.name} {system.hex_code}")
    ax.set_xlabel("Orbit (AU)")
    ax.set_yticks([])
    ax.axvline(0, linestyle="--", linewidth=1)
    for s in system.stars:
        x = 0.0
        y = 0.0
        if s.role == "Secondary" and s.semi_major_axis_au:
            x = s.semi_major_axis_au
        elif s.role == "Tertiary" and s.semi_major_axis_au:
            x = s.semi_major_axis_au if s.parent_role == "Primary" else (next((st.semi_major_axis_au for st in system.stars if st.role == "Secondary"), 0) + s.semi_major_axis_au)
        ax.scatter([x], [y], s=120, marker="*")
        ax.text(x, 0.03, s.code, rotation=35, fontsize=8)
    for role, hz in system.habitable_zones.items():
        ax.axvspan(hz["inner"], hz["outer"], alpha=0.08)
        ax.text(hz["center"], -0.045, role, ha="center", fontsize=8)
    y_base = -0.15
    for idx, w in enumerate(system.worlds):
        y = y_base - (idx % 6) * 0.06
        ax.scatter([w.orbit_au], [y], s=50)
        label = f"{idx+1}"
        if system.mainworld_index == idx:
            label += " MW"
        if w.moon_mainworld_candidate:
            label += " m"
        ax.text(w.orbit_au, y + 0.015, label, fontsize=8, ha="center")
    xmax = max([1.0] + [w.orbit_au for w in system.worlds] + [s.semi_major_axis_au or 0 for s in system.stars])
    ax.set_xlim(-0.2, xmax * 1.15)
    ax.set_ylim(-0.6, 0.18)
    return fig


# =============================================================================
# Streamlit UI
# =============================================================================
st.title("Traveller Advanced System Generator")
st.caption("Gerador avançado com sistemas múltiplos, clima expandido, luas detalhadas, casos especiais, texto procedural e exportação.")

with st.sidebar:
    st.header("Configuração")
    seed_text = st.text_input("Seed opcional", value="")
    generated_seed = seed_text.strip() or f"seed-{random.randint(100000, 999999)}"
    random.seed(generated_seed)
    allow_dim_primaries = st.checkbox("Permitir primárias pouco luminosas / difíceis de detectar", value=True)
    with_variance = st.checkbox("Aplicar pequena variação estelar", value=True)
    allow_special_cases = st.checkbox("Permitir casos especiais", value=True)
    empty_system = st.checkbox("Sistema vazio", value=False, help="Gera sistemas quase sem alvos normais: estrelas pouco detectáveis, nenhum mundo útil ou apenas um errante/cinturão residual.")
    mode = st.radio("Modo", ["Sistema único", "Subsector inteiro"], index=0)

if mode == "Sistema único":
    col_a, col_b = st.columns([1, 1])
    with col_a:
        hex_code = st.text_input("Hex", value="0101")
    with col_b:
        system_name = st.text_input("Nome do sistema", value="")
    if st.button("Gerar sistema", type="primary"):
        system = generate_single_system(hex_code, system_name, allow_dim_primaries, with_variance, allow_special_cases, generated_seed, empty_system=empty_system)
        st.subheader(f"{system.name} {system.hex_code}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Estrelas", len(system.stars))
        m2.metric("Mundos", len(system.worlds))
        m3.metric("Detectabilidade", system.detectability)
        mw_text = system.worlds[system.mainworld_index].uwp if system.mainworld_index is not None else "sem mainworld"
        m4.metric("UWP", mw_text)
        m5.metric("Seed", system.system_seed)
        st.write(f"Seed do sistema: {system.system_seed} | Seed do subsetor: {system.subsector_seed}")
        st.write("Deixe o nome em branco para regeneração canônica pelo system seed.")
        st.write(f"Fluxo aparente relativo a 1 parsec: {system.detectability_flux_at_1pc:.8f}")
        if system.empty_reason:
            st.warning(system.empty_reason)
        st.markdown("#### Estrelas")
        st.dataframe(stars_df(system.stars), use_container_width=True)
        st.markdown("#### Zonas habitáveis")
        if system.habitable_zones:
            st.dataframe(hz_df(system.habitable_zones), use_container_width=True)
        else:
            st.write("Nenhuma zona habitável calculável neste sistema.")
        st.markdown("#### Mundos")
        st.dataframe(worlds_df(system.worlds, system.mainworld_index), use_container_width=True)
        st.markdown("#### Luas")
        moon_table = moons_df(system.worlds)
        if not moon_table.empty:
            st.dataframe(moon_table, use_container_width=True)
        else:
            st.write("Nenhuma lua gerada.")
        st.markdown("#### Mapa orbital")
        st.pyplot(plot_system_map(system))
        if system.mainworld_index is not None:
            mw = system.worlds[system.mainworld_index]
            st.markdown("#### Descrição do mainworld")
            st.write(mw.description)
        st.markdown("#### Traveller Map record")
        tmap = traveller_map_record(system)
        st.dataframe(pd.DataFrame([tmap]), use_container_width=True)
        st.markdown("#### JSON")
        payload = system_json(system)
        st.json(payload)
        json_text = json.dumps(payload, indent=2, ensure_ascii=False)
        st.download_button("Baixar JSON", data=json_text.encode("utf-8"), file_name=f"{system.hex_code}_{system.name.replace(' ', '_')}.json", mime="application/json")
        tsv = "\t".join(tmap.keys()) + "\n" + "\t".join(str(tmap[k]) for k in tmap.keys())
        st.download_button("Baixar linha Traveller Map/TSV", data=tsv.encode("utf-8"), file_name=f"{system.hex_code}_{system.name.replace(' ', '_')}_travellermap.tsv", mime="text/tab-separated-values")
    else:
        st.info("Configure e gere um sistema.")
else:
    prefix = st.text_input("Prefixo nominal", value="")
    density_pct = st.slider("Densidade estelar (%)", min_value=10, max_value=100, value=50, step=5)
    if st.button("Gerar subsector", type="primary"):
        systems = generate_subsector(prefix, density_pct, allow_dim_primaries, with_variance, allow_special_cases, generated_seed, empty_system)
        st.subheader(f"Subsector gerado: {len(systems)} sistemas")
        st.write(f"Seed do subsetor: {generated_seed}")
        df = subsector_df(systems)
        st.dataframe(df, use_container_width=True, height=600)
        st.markdown("#### Grade do subsector")
        grid_rows = []
        lookup = {s.hex_code: s for s in systems}
        for y in range(1, 11):
            row = []
            for x in range(1, 9):
                h = f"{x:02d}{y:02d}"
                if h in lookup:
                    mw = lookup[h].worlds[lookup[h].mainworld_index].uwp if lookup[h].mainworld_index is not None else "X000000-0"
                    row.append(f"{h}\n{lookup[h].name[:8]}\n{mw}\n{lookup[h].system_seed[:8]}")
                else:
                    row.append(h)
            grid_rows.append(row)
        grid_df = pd.DataFrame(grid_rows, columns=[f"{x:02d}" for x in range(1, 9)], index=[f"{y:02d}" for y in range(1, 11)])
        st.dataframe(grid_df, use_container_width=True, height=420)
        tsv_text = df.to_csv(sep="\t", index=False)
        json_text = json.dumps([system_json(s) for s in systems], indent=2, ensure_ascii=False)
        st.download_button("Baixar subsector TSV", data=tsv_text.encode("utf-8"), file_name="subsector_travellermap.tsv", mime="text/tab-separated-values")
        st.download_button("Baixar subsector JSON", data=json_text.encode("utf-8"), file_name="subsector_systems.json", mime="application/json")
    else:
        st.info("Configure e gere um subsector inteiro.")

st.divider()
st.markdown(
    """
### Implementado nesta versão
- seed por sistema derivada deterministicamente da seed do subsetor e do hex
- reprodução isolada de sistemas mantendo NAME, UWP, REMARKS, {Ix}, Stars e demais campos do registro
- geração de sistemas simples, binários e triplos
- cálculo de zona habitável por estrela e circumbinária quando aplicável
- detectabilidade revisada com fluxo relativo a 1 parsec
- geração de mundos por órbita
- geração completa de UWP para todos os mundos gerados
- clima expandido: thermal band, variação sazonal simples, pressão aproximada, travamento por maré, faixa dia/noite
- luas detalhadas com órbitas, tamanho, habitabilidade e candidatas a mainworld
- casos especiais: protostar, nebula, star cluster, neutron star, black hole, anomaly
- texto procedural expandido: atmosfera, céu, assentamentos, hooks e descrição planetária
- seed sempre registrada na interface e nas exportações
- opção de sistema vazio
- exportação Traveller Map/TSV e JSON
- geração automática de subsector inteiro

### Limites desta versão
- para reproduzir NAME exatamente a partir do system seed em modo de sistema único, deixe o campo de nome em branco; nomes digitados manualmente sobrescrevem o nome canônico
- estabilidade orbital rigorosa para binários/triplos continua simplificada
- luas candidatas a mainworld são sinalizadas, mas o mainworld sistêmico ainda prioriza corpos planetários
- casos especiais usam abstrações jogáveis, não modelagem astrofísica completa
"""
)
