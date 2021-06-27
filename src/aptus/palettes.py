""" Palettes for Aptus.
    http://nedbatchelder.com/code/aptus
"""

import colorsys
import math

from aptus import data_file

# Pure data-munging functions
def _255(*vals):
    """ Convert all arguments from 0-1.0 to 0-255.
    """
    return [int(round(x * 255)) for x in vals]

def _1(*vals):
    """ Convert all arguments from 0-255 to 0-1.0.
    """
    return [x/255.0 for x in vals]

def _clip(val, lo, hi):
    """ Clip a val to staying between lo and hi.
    """
    if val < lo:
        val = lo
    if val > hi:
        val = hi
    return val


class Palette:
    """ A palette is a list of colors for coloring the successive bands of the
        Mandelbrot set.

        colors is a list of RGB triples, 0-255, for display.
        fcolors is a list of RGB triples, 0.0-1.0, for computation.
        incolor is the RGB255 color for the interior of the set.
        _spec is a value that can be passed to from_spec to reconstitute the
            palette. It's returned by the spec property.
    """

    default_adjusts = {'hue': 0, 'saturation': 0}

    def __init__(self):
        self.incolor = (0,0,0)
        self.fcolors = [(0.0,0.0,0.0), (1.0,1.0,1.0)]
        self._spec = []
        self.adjusts = dict(self.default_adjusts)
        self.wrap = True

        self._colors_from_fcolors()

    def __len__(self):
        return len(self.fcolors)

    def __eq__(self, other):
        return self.colors == other.colors

    def __ne__(self, other):
        return not self.__eq__(other)

    def _colors_from_fcolors(self):
        """ Set self.colors from self.fcolors, adjusting them for hue, etc,
            in the process.
        """
        self.colors = []

        hue_adj = self.adjusts['hue']/360.0
        sat_adj = self.adjusts['saturation']/255.0

        for r, g, b in self.fcolors:
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            h = (h + hue_adj) % 1.0
            s = _clip(s + sat_adj, 0.0, 1.0)
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            self.colors.append(_255(r, g, b))
        self._colorbytes = None

    def color_bytes(self):
        """ Compute a string of RGB bytes for use in the engine.
        """
        if not self._colorbytes:
            colbytes = b"".join([ bytes([r, g, b]) for r,g,b in self.colors ])
            self._colorbytes = colbytes
        return self._colorbytes

    def spec(self):
        """ Create a textual description of the palette, for later reconstitution
            with from_spec().
        """
        s = self._spec[:]
        if self.adjusts != self.default_adjusts:
            s.append(['adjust', self.adjusts])
        if self.incolor != (0,0,0):
            s.append(['rgb_incolor', {'color': self.incolor}])
        if not self.wrap:
            s.append(['wrapping', {'wrap': 0}])
        return s

    def rgb_colors(self, colors):
        """ Use an explicit list of RGB colors as the palette.
        """
        self.colors = colors[:]
        self.fcolors = [ _1(*rgb255) for rgb255 in self.colors ]
        self._colorbytes = None
        self._spec.append(['rgb_colors', {'colors':colors}])
        return self

    def spectrum(self, ncolors, h=(0,360), l=(50,200), s=150):
        if isinstance(h, (int, float)):
            h = (int(h), int(h))
        if isinstance(l, (int, float)):
            l = (int(l), int(l))
        if isinstance(s, (int, float)):
            s = (int(s), int(s))

        hlo, hhi = h
        llo, lhi = l
        slo, shi = s

        fcolors = []
        for pt in range(ncolors//2):
            hfrac = (pt*1.0/(ncolors/2))
            hue = hlo + (hhi-hlo)*hfrac
            fcolors.append(colorsys.hls_to_rgb(hue/360.0, llo/255.0, slo/255.0))

            hfrac = (pt*1.0+0.5)/(ncolors/2)
            hue = hlo + (hhi-hlo)*hfrac
            fcolors.append(colorsys.hls_to_rgb(hue/360.0, lhi/255.0, shi/255.0))
        self.fcolors = fcolors
        self._colors_from_fcolors()

        args = {'ncolors':ncolors}
        if h != (0,360):
            if hlo == hhi:
                args['h'] = hlo
            else:
                args['h'] = h
        if l != (50,200):
            if llo == lhi:
                args['l'] = llo
            else:
                args['l'] = l
        if s != (150,150):
            if slo == shi:
                args['s'] = slo
            else:
                args['s'] = s
        self._spec.append(['spectrum', args])
        return self

    def stretch(self, steps, hsl=False, ease=None):
        """ Interpolate between colors in the palette, stretching it out.
            Works in either RGB or HSL space.
        """
        fcolors = [None]*(len(self.fcolors)*steps)
        for i in range(len(fcolors)):
            color_index = i//steps
            a0, b0, c0 = self.fcolors[color_index]
            a1, b1, c1 = self.fcolors[(color_index + 1) % len(self.fcolors)]
            if hsl:
                a0, b0, c0 = colorsys.rgb_to_hls(a0, b0, c0)
                a1, b1, c1 = colorsys.rgb_to_hls(a1, b1, c1)
                if a1 < a0 and a0-a1 > 0.01:
                    a1 += 1
            step = i % steps / steps

            if ease == "sine":
                step = -(math.cos(math.pi * step) - 1) / 2;
            elif isinstance(ease, (int, float)):
                if step < 0.5:
                    step = math.pow(2 * step, ease) / 2
                else:
                    step = 1 - math.pow(-2 * step + 2, ease) / 2

            ax, bx, cx = (
                a0 + (a1 - a0) * step,
                b0 + (b1 - b0) * step,
                c0 + (c1 - c0) * step,
                )
            if hsl:
                ax, bx, cx = colorsys.hls_to_rgb(ax, bx, cx)
            fcolors[i] = (ax, bx, cx)
        self.fcolors = fcolors
        self._colors_from_fcolors()
        self._spec.append(['stretch', {'steps':steps, 'hsl':hsl, 'ease':ease}])
        return self

    def adjust(self, hue=0, saturation=0):
        """ Make adjustments to various aspects of the display of the palette.
            0 <= hue <= 360
            0 <= saturation <= 255
        """
        adj = self.adjusts
        adj['hue'] = (adj['hue'] + hue) % 360
        adj['saturation'] = _clip(adj['saturation'] + saturation, -255, 255)
        self._colors_from_fcolors()
        return self

    def reset(self):
        """ Reset all palette adjustments.
        """
        self.adjusts = {'hue': 0, 'saturation': 0}
        self._colors_from_fcolors()
        return self

    def rgb_incolor(self, color):
        """ Set the color for the interior of the Mandelbrot set.
        """
        self.incolor = color
        return self

    def wrapping(self, wrap):
        """ Set the wrap boolean on or off.
        """
        self.wrap = wrap
        return self

    def gradient(self, ggr_file, ncolors):
        """ Create the palette from a GIMP .ggr gradient file.
        """
        from aptus.ggr import GimpGradient
        ggr = GimpGradient()
        try:
            ggr.read(ggr_file)
            self.fcolors = [ ggr.color(float(c)/ncolors) for c in range(ncolors) ]
        except IOError:
            self.fcolors = [ (0.0,0.0,0.0), (1.0,0.0,0.0), (1.0,1.0,1.0) ]
        self._colors_from_fcolors()
        self._spec.append(['gradient', {'ggr_file':ggr_file, 'ncolors':ncolors}])
        return self

    def xaos(self):
        # Colors taken from Xaos, to get the same rendering.
        xaos_colors = [
            (0, 0, 0),
            (120, 119, 238),
            (24, 7, 25),
            (197, 66, 28),
            (29, 18, 11),
            (135, 46, 71),
            (24, 27, 13),
            (241, 230, 128),
            (17, 31, 24),
            (240, 162, 139),
            (11, 4, 30),
            (106, 87, 189),
            (29, 21, 14),
            (12, 140, 118),
            (10, 6, 29),
            (50, 144, 77),
            (22, 0, 24),
            (148, 188, 243),
            (4, 32, 7),
            (231, 146, 14),
            (10, 13, 20),
            (184, 147, 68),
            (13, 28, 3),
            (169, 248, 152),
            (4, 0, 34),
            (62, 83, 48),
            (7, 21, 22),
            (152, 97, 184),
            (8, 3, 12),
            (247, 92, 235),
            (31, 32, 16)
        ]

        self.rgb_colors(xaos_colors)
        del self._spec[-1]
        self.stretch(8)
        del self._spec[-1]
        self._spec.append(['xaos', {}])
        return self

    def from_spec(self, spec):
        for op, args in spec:
            getattr(self, op)(**args)
        return self

all_palettes = [
    Palette().spectrum(12).stretch(10, hsl=True),
    Palette().spectrum(12).stretch(10, hsl=True, ease="sine"),
    Palette().spectrum(12, l=(50,150), s=150).stretch(25, hsl=True),
    Palette().spectrum(12, l=(50,150), s=150).stretch(25, hsl=True, ease="sine"),
    Palette().spectrum(64, l=125, s=175),
    Palette().spectrum(48, l=(100,150), s=175).stretch(5),
    Palette().spectrum(2, h=250, l=(100,150), s=175).stretch(10, hsl=True),
    Palette().spectrum(2, h=290, l=(75,175), s=(230,25)).stretch(10, hsl=True),
    Palette().spectrum(16, l=125, s=175),
    Palette().xaos(),
    Palette().rgb_colors([(255,192,192), (255,255,255)]).rgb_incolor((192,192,255)),
    Palette().rgb_colors([(255,255,255), (0,0,0), (0,0,0), (0,0,0)]),
    Palette().rgb_colors([(255,255,255)]),
    Palette().spectrum(2, h=120, l=(50,200), s=125).stretch(128, hsl=True),
    Palette().gradient(data_file('palettes/bluefly.ggr'), 50),
    Palette().gradient(data_file('palettes/ib18.ggr'), 50),
    Palette().gradient(data_file('palettes/redblue.ggr'), 50),
    Palette().gradient(data_file('palettes/DEM_screen.ggr'), 50),
    Palette().rgb_colors([(0,0,0), (255,255,255)]).wrapping(False),
    ]
