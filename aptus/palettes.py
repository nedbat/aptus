""" Palettes for Mand.
"""

import colorsys

class Palette:
    def __init__(self):
        self.incolor = (0,0,0)
        self.colors = [(0,0,0),(255,255,255)]
        self.phase = 0
    
    def _255(self, *vals):
        return map(lambda x:int(x*255), vals)
    
    def set_colors(self, colors):
        self.colors = colors[:]
        return self

    def set_rainbow(self, ncolors, s, v):
        colors = []
        for h in xrange(ncolors):
            colors.append(self._255(*colorsys.hsv_to_rgb(h*1.0/ncolors,s,v)))
        self.colors = colors
        return self
    
    def set_rainbow_hls(self, ncolors, l, s):
        colors = []
        for h in xrange(ncolors):
            colors.append(self._255(*colorsys.hls_to_rgb(h*1.0/ncolors,l,s)))
        self.colors = colors
        return self
    
    def set_rainbow_ramps(self, npts, nsteps, hrange=(0.0,1.0), lrange=(.2,.8), srange=.6):
        if isinstance(hrange, (int, float)):
            hrange = (float(hrange), float(hrange))
        if isinstance(lrange, (int, float)):
            lrange = (float(lrange), float(lrange))
        if isinstance(srange, (int, float)):
            srange = (float(srange), float(srange))
        
        hlo, hhi = hrange
        llo, lhi = lrange
        slo, shi = srange
        
        colors = []
        for pt in range(npts):
            for step in range(nsteps):
                hfrac = (pt*1.0/npts) + (step)*1.0/(npts * 2 * nsteps)
                h = hlo + (hhi-hlo)*hfrac
                l = llo + (lhi-llo)*step/nsteps
                s = slo + (shi-slo)*step/nsteps
                colors.append(self._255(*colorsys.hls_to_rgb(h, l, s)))
            for step in range(nsteps):
                hfrac = (pt*1.0/npts) + (step+nsteps)*1.0/(npts * 2 * nsteps)
                h = hlo + (hhi-hlo)*hfrac
                l = lhi - (lhi-llo)*step/nsteps
                s = shi - (shi-slo)*step/nsteps
                colors.append(self._255(*colorsys.hls_to_rgb(h, l, s)))
        self.colors = colors
        return self
    
    def stretch(self, steps):
        colors = [None]*(len(self.colors)*steps)
        for i in range(len(colors)):
            color_index = i//steps
            r0, g0, b0 = self.colors[color_index]
            r1, g1, b1 = self.colors[(color_index + 1) % len(self.colors)]
            step = float(i % steps)/steps
            colors[i] = (
                int(r0 + (r1 - r0) * step),
                int(g0 + (g1 - g0) * step),
                int(b0 + (b1 - b0) * step),
                )
        self.colors = colors    
        return self
    
    def set_incolor(self, color):
        self.incolor = color
        return self

    def from_ggr(self, ggr_file, ncolors):
        from ggr import GimpGradient
        ggr = GimpGradient()
        try:
            ggr.read(ggr_file)
            self.colors = [ self._255(*ggr.color(float(c)/ncolors)) for c in range(ncolors) ]
        except:
            self.colors = [ (0,0,0), (255,0,0), (255,255,255) ]
        return self
    
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

xaos_palette = Palette().set_colors(xaos_colors).stretch(8)

all_palettes = [
    xaos_palette,
    Palette().set_rainbow_ramps(6, nsteps=10),
    Palette().set_rainbow_ramps(6, nsteps=25, lrange=(.2,.6), srange=.6),
    Palette().set_rainbow_ramps(24, nsteps=5, lrange=(.4,.6), srange=.7),
    Palette().set_rainbow_ramps(1, nsteps=10, hrange=.7, lrange=(.4,.6), srange=.7),
    Palette().set_rainbow_ramps(1, nsteps=10, hrange=.8, lrange=(.3,.7), srange=(.9,.1)),
    Palette().set_rainbow(16, .7, .9),
    Palette().set_rainbow_hls(16, .5, .7),
    Palette().set_rainbow_ramps(4, nsteps=4, lrange=.5, srange=.7),
    Palette().set_colors([(255,192,192), (255,255,255)]).set_incolor((192,192,255)),
    Palette().set_colors([(255,255,255), (0,0,0), (0,0,0), (0,0,0)]),
    Palette().from_ggr('palettes/bluefly.ggr', 20),
    Palette().from_ggr('palettes/ib18.ggr', 20),
    Palette().from_ggr('palettes/ib18.ggr', 50),
    ]
