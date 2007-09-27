""" Palettes for Aptus.
"""

import colorsys

class Palette:
    def __init__(self):
        self.incolor = (0,0,0)
        self.colors = [(0,0,0),(255,255,255)]
        self.phase = 0
    
    def _255(self, *vals):
        return map(lambda x:int(x*255), vals)
    
    def _1(self, *vals):
        return map(lambda x:x/255.0, vals)
    
    def set_colors(self, colors):
        self.colors = colors[:]
        return self

    def set_rainbow(self, ncolors, l, s):
        colors = []
        for h in xrange(ncolors):
            colors.append(self._255(*colorsys.hls_to_rgb(h*1.0/ncolors,l,s)))
        self.colors = colors
        return self
    
    def set_rainbow_ramps(self, npts, nsteps=1, hrange=(0.0,1.0), lrange=(.2,.8), srange=.6):
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
    
    def stretch(self, steps, hsl=False):
        colors = [None]*(len(self.colors)*steps)
        for i in range(len(colors)):
            color_index = i//steps
            a0, b0, c0 = self.colors[color_index]
            a1, b1, c1 = self.colors[(color_index + 1) % len(self.colors)]
            if hsl:
                a0, b0, c0 = colorsys.rgb_to_hls(*self._1(a0, b0, c0))
                a1, b1, c1 = colorsys.rgb_to_hls(*self._1(a1, b1, c1))
                if a1 < a0:
                    a1 += 1
            step = float(i % steps)/steps
            ax, bx, cx = (
                a0 + (a1 - a0) * step,
                b0 + (b1 - b0) * step,
                c0 + (c1 - c0) * step,
                )
            if hsl:
                ax, bx, cx = self._255(*colorsys.hls_to_rgb(ax, bx, cx))
            colors[i] = (int(ax), int(bx), int(cx))
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

all_palettes = [
    Palette().set_colors(xaos_colors).stretch(8),
    Palette().set_rainbow_ramps(6, nsteps=1).stretch(10, hsl=True),
    Palette().set_rainbow_ramps(6, nsteps=10),
    Palette().set_rainbow_ramps(6, nsteps=1).stretch(10),
    Palette().set_rainbow_ramps(6, nsteps=25, lrange=(.2,.6), srange=.6),
    Palette().set_rainbow_ramps(24, nsteps=5, lrange=(.4,.6), srange=.7),
    Palette().set_rainbow_ramps(1, nsteps=10, hrange=.7, lrange=(.4,.6), srange=.7),
    Palette().set_rainbow_ramps(1, nsteps=10, hrange=.8, lrange=(.3,.7), srange=(.9,.1)),
    Palette().set_rainbow(16, .5, .7),
    Palette().set_rainbow_ramps(4, nsteps=2, lrange=.5, srange=.7),
    Palette().set_colors([(255,192,192), (255,255,255)]).set_incolor((192,192,255)),
    Palette().set_colors([(255,255,255), (0,0,0), (0,0,0), (0,0,0)]),
    Palette().set_rainbow_ramps(1, nsteps=128, hrange=.333, lrange=(0.2,0.8), srange=.5),
    Palette().from_ggr('palettes/bluefly.ggr', 20),
    Palette().from_ggr('palettes/ib18.ggr', 20),
    Palette().from_ggr('palettes/ib18.ggr', 50),
    ]

# A simple viewer to see the palettes.
if __name__ == '__main__':
    import sys, wx

    class PalettesView(wx.Frame):
        def __init__(self, palettes):
            super(PalettesView, self).__init__(None, -1, 'All palettes')
            self.palettes = palettes
            self.SetSize((800, 300))
            self.panel = wx.Panel(self)
            self.panel.Bind(wx.EVT_PAINT, self.on_paint)
            self.panel.Bind(wx.EVT_SIZE, self.on_size)

        def on_paint(self, event):
            dc = wx.PaintDC(self.panel)
            cw, ch = self.GetClientSize()
            stripe_height = ch/len(self.palettes)
            for y, pal in enumerate(self.palettes):
                self.paint_palette(dc, pal, y*stripe_height, stripe_height)
                
        def paint_palette(self, dc, pal, y0, height):
            cw, ch = self.GetClientSize()
            ncolors = len(pal.colors)
            width = float(cw)/ncolors
            for c in range(0, ncolors):
                dc.SetPen(wx.Pen(wx.Colour(*pal.colors[c]), 1))
                dc.SetBrush(wx.Brush(wx.Colour(*pal.colors[c]), wx.SOLID))
                #dc.DrawRectangle(chunkw*c, y0, chunkw, height)
                dc.DrawRectangle(int(c*width), y0, int(width+1), height)
        
        def on_size(self, event):
            self.Refresh()

    app = wx.PySimpleApp()
    f = PalettesView(all_palettes)
    f.Show()
    app.MainLoop()
