import colorsys
import math

from aptus.palettes import Palette

def test_unstretched():
    pal = Palette().spectrum(12)
    assert pal.colors == [
        [79, 21, 21],
        [232, 200, 168],
        [79, 79, 21],
        [200, 232, 168],
        [21, 79, 21],
        [168, 232, 200],
        [21, 79, 79],
        [168, 200, 232],
        [21, 21, 79],
        [200, 168, 232],
        [79, 21, 79],
        [232, 168, 200],
    ]

def test_stretched():
    pal = Palette().spectrum(12).stretch(3, hsl=True)
    assert pal.colors == [
        [79, 21, 21],
        [159, 61, 41],
        [212, 129, 88],
        [232, 200, 168],
        [212, 171, 88],
        [159, 139, 41],
        [79, 79, 21],
        [139, 159, 41],
        [171, 212, 88],
        [200, 232, 168],
        [129, 212, 88],
        [61, 159, 41],
        [21, 79, 21],
        [41, 159, 61],
        [88, 212, 129],
        [168, 232, 200],
        [88, 212, 171],
        [41, 159, 139],
        [21, 79, 79],
        [41, 139, 159],
        [88, 171, 212],
        [168, 200, 232],
        [88, 129, 212],
        [41, 61, 159],
        [21, 21, 79],
        [61, 41, 159],
        [129, 88, 212],
        [200, 168, 232],
        [171, 88, 212],
        [139, 41, 159],
        [79, 21, 79],
        [159, 41, 139],
        [212, 88, 171],
        [232, 168, 200],
        [212, 88, 129],
        [159, 41, 61],
    ]

def assert_is_hue(hue, r, g, b):
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    # black or white are ok.
    assert math.isclose(h, hue/360.0) or l in [0, 1]

def test_one_hue():
    pal = Palette().spectrum(12, h=245, l=(0, 255)).stretch(5, hsl=True)
    for rgb in pal.colors:
        assert_is_hue(245, *rgb)
