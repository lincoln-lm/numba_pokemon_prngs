"""Linear Congruential Pseudo Random Number Generators"""

from .lcrng32 import (
    LCRNG32,
    LCRNG32RandomDistribution,
    lcrng32_init,
)
from .lcrng64 import (
    LCRNG64,
    LCRNG64RandomDistribution,
    lcrng64_init,
)

# pylint: disable=abstract-method


@lcrng32_init(
    add=0x6073, mult=0x41C64E6D, distribution=LCRNG32RandomDistribution.MODULO
)
class PokeRNGMod(LCRNG32):
    """Standard Pokemon LCRNG with modulo random distribution"""


@lcrng32_init(
    add=0x6073,
    mult=0x41C64E6D,
    distribution=LCRNG32RandomDistribution.RECIPROCAL_DIVISION,
)
class PokeRNGDiv(LCRNG32):
    """Standard Pokemon LCRNG with reciprocal division random distribution"""


@lcrng32_init(
    add=0x6073,
    mult=0x41C64E6D,
    distribution=LCRNG32RandomDistribution.MODULO,
    reverse=True,
)
class PokeRNGRMod(LCRNG32):
    """Reversed Standard Pokemon LCRNG with modulo random distribution"""


@lcrng32_init(
    add=0x6073,
    mult=0x41C64E6D,
    distribution=LCRNG32RandomDistribution.RECIPROCAL_DIVISION,
    reverse=True,
)
class PokeRNGRDiv(LCRNG32):
    """Reversed Standard Pokemon LCRNG with reciprocal division random distribution"""


@lcrng32_init(
    add=0x1,
    mult=0x6C078965,
)
class ARNG(LCRNG32):
    """Alternative Pokemon LCRNG with modulo random distribution"""


@lcrng32_init(
    add=0x1,
    mult=0x6C078965,
    reverse=True,
)
class ARNGR(LCRNG32):
    """Reversed Alternative Pokemon LCRNG with modulo random distribution"""


@lcrng32_init(
    add=0x269EC3,
    mult=0x343FD,
    distribution=LCRNG32RandomDistribution.MODULO,
)
class XDRNG(LCRNG32):
    """Pokemon Colosseum/XD/Channel LCRNG with modulo random distribution"""


@lcrng32_init(
    add=0x269EC3,
    mult=0x343FD,
    distribution=LCRNG32RandomDistribution.MODULO,
    reverse=True,
)
class XDRNGR(LCRNG32):
    """Reversed Pokemon Colosseum/XD/Channel LCRNG with modulo random distribution"""


@lcrng64_init(
    add=0x269EC3,
    mult=0x5D588B656C078965,
    distribution=LCRNG64RandomDistribution.MULTIPLICATION_SHIFT,
)
class BWRNG(LCRNG64):
    """Pokemon Gen 5 LCRNG with multiplication-shift random distribution"""


@lcrng64_init(
    add=0x269EC3,
    mult=0x5D588B656C078965,
    distribution=LCRNG64RandomDistribution.MULTIPLICATION_SHIFT,
    reverse=True,
)
class BWRNGR(LCRNG64):
    """Reversed Pokemon Gen 5 LCRNG with multiplication-shift random distribution"""


# pylint: enable=abstract-method
