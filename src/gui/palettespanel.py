""" Palette visualization for Aptus.
"""

from aptus.importer import importer
wx = importer("wx")
from wx.lib.scrolledpanel import ScrolledPanel

class PaletteWin(wx.Window):
    def __init__(self, parent, palette, size=wx.DefaultSize):
        wx.Window.__init__(self, parent, size=size)
        self.palette = palette

        self.Bind(wx.EVT_PAINT, self.on_paint)
        
    def on_paint(self, event_unused):
        dc = wx.PaintDC(self)
        cw, ch = self.GetClientSize()
        ncolors = len(self.palette.colors)
        width = float(cw)/ncolors
        for c in range(0, ncolors):
            dc.SetPen(wx.Pen(wx.Colour(*self.palette.colors[c]), 1))
            dc.SetBrush(wx.Brush(wx.Colour(*self.palette.colors[c]), wx.SOLID))
            dc.DrawRectangle(int(c*width), 0, int(width+1), ch)
        
class PalettesPanel(ScrolledPanel):
    def __init__(self, parent, palettes, size=wx.DefaultSize):
        ScrolledPanel.__init__(self, parent, size=size)
        
        self.palettes = palettes
        self.pal_height = 30
        
        self.sizer = wx.FlexGridSizer(len(self.palettes), 1)
        for pal in self.palettes:
            self.a_palette = PaletteWin(self, pal, size=(200, self.pal_height))
            self.sizer.Add(self.a_palette, wx.EXPAND)

        self.sizer.AddGrowableCol(0)
        self.sizer.SetFlexibleDirection(wx.HORIZONTAL)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(True)
        self.SetupScrolling()
        #self.Bind(wx.EVT_SIZE, self.on_size)
        
    def on_size(self, event_unused):
        h = len(self.palettes) * self.pal_height
        cw, _ = self.GetClientSize()
        self.SetVirtualSize((cw, h))
        self.sizer.SetMinSize((cw, h))
        print "Laying out: %r" % self.sizer.GetFlexibleDirection()
        self.Layout()
        self.sizer.Layout()


class PalettesFrame(wx.MiniFrame):
    def __init__(self, palettes):
        wx.MiniFrame.__init__(self, None, title='Palettes', size=(250, 350),
            style=wx.DEFAULT_FRAME_STYLE)
        self.panel = PalettesPanel(self, palettes)
