"""Tests for Mersenne Twister classes"""

from hashlib import sha256
from numba_pokemon_prngs.mersenne_twister import (
    MersenneTwister,
    SIMDFastMersenneTwister,
    TinyMersenneTwister,
)


def test_init():
    """Test initialization re_init() functions"""
    test_mt = MersenneTwister(0)
    test_sfmt = SIMDFastMersenneTwister(0)
    test_tinymt = TinyMersenneTwister(0)

    assert tuple(
        sha256(test_mt.state.data.tobytes()).hexdigest()
        for _ in (
            test_mt.re_init(seed)
            for seed in (0x12345678, 0xDEADBEEF, 0x88776655, 0xCAFEBEEF)
        )
    ) == (
        "f75f425d88ef519f1b972ccc688c2659e699934204c3af00f8527d5aaec5db81",
        "abfdf49e18c3e57a2ae366c11f95252082872085e5048a8d475751bd09bea408",
        "af5ae4b6d7dce444709d7fe97f718c698f2d7076fcc8a62384a1fa2ee2246f17",
        "006d3165256376db0287398b900c8eb38828f2a23e298c93167597d61fca3729",
    )

    assert tuple(
        sha256(test_sfmt.state.tobytes()).hexdigest()
        for _ in (
            test_sfmt.re_init(seed)
            for seed in (0x12345678, 0xDEADBEEF, 0x88776655, 0xCAFEBEEF)
        )
    ) == (
        "ccc11319e23c70d1a94dd6d3fefa1086921674e59e72c531c2cc27544320b21d",
        "abfdf49e18c3e57a2ae366c11f95252082872085e5048a8d475751bd09bea408",
        "af5ae4b6d7dce444709d7fe97f718c698f2d7076fcc8a62384a1fa2ee2246f17",
        "006d3165256376db0287398b900c8eb38828f2a23e298c93167597d61fca3729",
    )

    assert tuple(
        sha256(test_tinymt.state.tobytes()).hexdigest()
        for _ in (
            test_tinymt.re_init(seed)
            for seed in (0x12345678, 0xDEADBEEF, 0x88776655, 0xCAFEBEEF)
        )
    ) == (
        "350d288590dfff8eca6d4706564c5f9ea8f5a91cd8c34ca2ea80403354f5f1ae",
        "de1246ade9d190ce834d1d8eca60dba6c1cca2ec1f1147a27aae02db69b75210",
        "f20117066fc37e31df5a1f69d28598dabb7cbe54c5c19e0d476d1cce270c2804",
        "1da1314cbd0d62a328c4f6dc2bc44bcea4cd828e8522e7713a1322f0696b4305",
    )


def test_shuffle():
    """Test state transition shuffle() functions"""
    test_mt = MersenneTwister(0x12345678)
    test_sfmt = SIMDFastMersenneTwister(0x12345678)
    test_tinymt = TinyMersenneTwister(0x12345678)

    assert tuple(
        sha256(test_mt.state.data.tobytes()).hexdigest()
        for _ in (test_mt.shuffle() for _ in range(4))
    ) == (
        "324aa1b936d575034bdac4a7496fff66ad97a712ff8245ef1e133556d214644d",
        "61512e29c9a22c48af8bd959f630af79e869829d6a9e2387e74ea44b2971ef87",
        "c88eaf942ad61179f1b8f11980ec989ab73f51c1d6d4e12494d1f1a37530dd02",
        "60ef09a8fbb7b2cc42c4ce34cb824ffedb4ca5495eb665d58cef5affbf191f22",
    )

    assert tuple(
        sha256(test_sfmt.state.tobytes()).hexdigest()
        for _ in (test_sfmt.shuffle() for _ in range(4))
    ) == (
        "d9ee5bdb83f403eb986e5a7631f24cc89b597d5934ef4b203b23657105cb6393",
        "da5879fbf2cd08855f7539d349c8df3c32c223482c28acd80b9e3dbde56b2cb9",
        "d64b5657cccee15f17bf49e1dd901eeb91db137d45ac02d0c8fd0525a060a914",
        "31b8d4454a387b720735f9d0ce6e7de582afdf2c49e1a2f1b5adaccc86d18773",
    )

    assert tuple(
        sha256(test_tinymt.state.tobytes()).hexdigest()
        for _ in (test_tinymt.shuffle() for _ in range(4))
    ) == (
        "37602281ccc1077e337dd26dfc3091a8246e7ae112b5ba7cf30f429434689698",
        "1599a6b80a09a3b96bbc3e0504aee7ec791f84f307cc9cff9830e17a6ef557c6",
        "afc1f50d08c709bd6b7ae61d1ce4994a33c2d30ded8bee4fb85a22c20bec8bf3",
        "50623afc3625b181897c68501f32b40cd604147e03bd3f6ee1386d4d49f6e472",
    )


def test_next():
    """Test state access/tempering of next() functions"""
    test_mt = MersenneTwister(0x12345678)
    test_sfmt = SIMDFastMersenneTwister(0x12345678)
    test_tinymt = TinyMersenneTwister(0x12345678)
    assert tuple(test_mt.next() for _ in range(5)) == (
        3331822403,
        157471482,
        2805605540,
        3776487808,
        3041352379,
    )
    assert tuple(test_sfmt.next() for _ in range(5)) == (
        4883196442416872734,
        15199411253426940632,
        10268751338100151066,
        602836503723090224,
        2976335783597749142,
    )
    assert tuple(test_tinymt.next() for _ in range(5)) == (
        2481148692,
        2185716838,
        3625480341,
        3369169125,
        3389594172,
    )


def test_next_rand():
    """Test next_rand()/next_rand_mod() bounded rand functions"""
    test_mt = MersenneTwister(0x12345678)
    test_sfmt = SIMDFastMersenneTwister(0x12345678)
    test_tinymt = TinyMersenneTwister(0x12345678)

    assert tuple(
        tuple(test_mt.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 0, 1, 1, 1),
        (1, 2, 4, 1, 2),
        (17, 12, 1, 22, 1),
        (54, 22, 61, 83, 69),
        (11, 140, 12, 76, 59),
    )
    assert tuple(
        tuple(test_mt.next_rand_mod(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 1, 1, 0, 0),
        (3, 1, 1, 0, 1),
        (4, 24, 19, 10, 20),
        (41, 18, 71, 28, 88),
        (154, 172, 65, 161, 125),
    )

    # next_rand is next_rand_mod
    assert tuple(
        tuple(test_sfmt.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 0, 0, 0, 0),
        (1, 3, 4, 2, 3),
        (1, 12, 1, 14, 1),
        (61, 24, 95, 71, 43),
        (241, 85, 35, 253, 83),
    )

    assert tuple(
        tuple(test_tinymt.next_rand(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (1, 1, 1, 1, 1),
        (1, 0, 2, 4, 0),
        (23, 0, 4, 20, 21),
        (59, 51, 5, 47, 77),
        (156, 84, 195, 5, 150),
    )
    assert tuple(
        tuple(test_tinymt.next_rand_mod(maximum) for _ in range(5))
        for maximum in (2, 5, 25, 100, 256)
    ) == (
        (0, 0, 1, 1, 0),
        (2, 2, 2, 0, 3),
        (20, 14, 8, 22, 2),
        (98, 10, 40, 11, 54),
        (30, 185, 57, 8, 182),
    )
