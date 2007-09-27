""" Palettes for Aptus.
"""

import colorsys

class Palette:
    def __init__(self):
        self.incolor = (0,0,0)
        self.colors = [(0,0,0),(255,255,255)]
        self.spec = []
        
    def _255(self, *vals):
        return map(lambda x:int(round(x*255)), vals)
    
    def _1(self, *vals):
        return map(lambda x:x/255.0, vals)
    
    def rgb_colors(self, colors):
        """ Use an explicit list of RGB colors as the palette.
        """
        self.colors = colors[:]
        self.spec.append(['rgb_colors', {'colors':colors}])
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
        
        colors = []
        for pt in range(ncolors//2):
            hfrac = (pt*1.0/(ncolors/2))
            hue = hlo + (hhi-hlo)*hfrac
            colors.append(self._255(*colorsys.hls_to_rgb(hue/360.0, llo/255.0, slo/255.0)))

            hfrac = (pt*1.0+0.5)/(ncolors/2)
            hue = hlo + (hhi-hlo)*hfrac
            colors.append(self._255(*colorsys.hls_to_rgb(hue/360.0, lhi/255.0, shi/255.0)))
        self.colors = colors
        
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
        self.spec.append(['spectrum', args])
        return self
    
    def stretch(self, steps, hsl=False):
        """ Interpolate between colors in the palette, stretching it out.
            Works in either RGB or HSL space.
        """
        colors = [None]*(len(self.colors)*steps)
        for i in range(len(colors)):
            color_index = i//steps
            a0, b0, c0 = self.colors[color_index]
            a1, b1, c1 = self.colors[(color_index + 1) % len(self.colors)]
            if hsl:
                a0, b0, c0 = colorsys.rgb_to_hls(*self._1(a0, b0, c0))
                a1, b1, c1 = colorsys.rgb_to_hls(*self._1(a1, b1, c1))
                if a1 < a0 and a0-a1 > 0.01:
                    a1 += 1
            step = float(i % steps)/steps
            ax, bx, cx = (
                a0 + (a1 - a0) * step,
                b0 + (b1 - b0) * step,
                c0 + (c1 - c0) * step,
                )
            if hsl:
                ax, bx, cx = self._255(*colorsys.hls_to_rgb(ax, bx, cx))
            colors[i] = map(lambda x:int(round(x)), (ax, bx,cx))
        self.colors = colors
        self.spec.append(['stretch', {'steps':steps, 'hsl':hsl}])
        return self
    
    def rgb_incolor(self, color):
        """ Set the color for the interior of the Mandelbrot set.
        """
        self.incolor = color
        self.spec.append(['rgb_incolor', {'color':color}])
        return self

    def gradient(self, ggr_file, ncolors):
        """ Create the palette from a GIMP .ggr gradient file.
        """
        from ggr import GimpGradient
        ggr = GimpGradient()
        try:
            ggr.read(ggr_file)
            self.colors = [ self._255(*ggr.color(float(c)/ncolors)) for c in range(ncolors) ]
        except:
            self.colors = [ (0,0,0), (255,0,0), (255,255,255) ]
        self.spec.append(['gradient', {'ggr_file':ggr_file, 'ncolors':ncolors}])
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
        del self.spec[-1]
        self.stretch(8)
        del self.spec[-1]
        self.spec.append(['xaos', {}])
        return self
    
    def from_spec(self, spec):
        for op, args in spec:
            getattr(self, op)(**args)
        return self
    
all_palettes = [
    Palette().xaos(),
    Palette().spectrum(12).stretch(10, hsl=True),
    Palette().spectrum(12, l=(50,150), s=150).stretch(25, hsl=True),
    Palette().spectrum(48, l=(100,150), s=175).stretch(5),
    Palette().spectrum(2, h=250, l=(100,150), s=175).stretch(10, hsl=True),
    Palette().spectrum(2, h=290, l=(75,175), s=(230,25)).stretch(10, hsl=True),
    Palette().spectrum(16, l=125, s=175),
    Palette().rgb_colors([(255,192,192), (255,255,255)]).rgb_incolor((192,192,255)),
    Palette().rgb_colors([(255,255,255), (0,0,0), (0,0,0), (0,0,0)]),
    Palette().spectrum(2, h=120, l=(50,200), s=125).stretch(128, hsl=True),
    Palette().gradient('palettes/bluefly.ggr', 20),
    Palette().gradient('palettes/ib18.ggr', 20),
    Palette().gradient('palettes/ib18.ggr', 50),
    ]

# A simple viewer to see the palettes.
if __name__ == '__main__':
    import sys, wx

    class PalettesView(wx.Frame):
        def __init__(self, palettes):
            super(PalettesView, self).__init__(None, -1, 'All palettes')
            self.palettes = palettes
            self.SetSize((1000, 500))
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
                dc.DrawRectangle(int(c*width), y0, int(width+1), height)
        
        def on_size(self, event):
            self.Refresh()

    app = wx.PySimpleApp()
    f = PalettesView(all_palettes)
    f.Show()
    app.MainLoop()
