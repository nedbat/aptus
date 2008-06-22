""" Palette visualization for Aptus.
"""

from aptus.importer import importer
from aptus.gui.ids import *

wx = importer("wx")
from wx.lib.scrolledpanel import ScrolledPanel

class PaletteWin(wx.Window):
    def __init__(self, parent, palette, ipal, viewwin, size=wx.DefaultSize):
        wx.Window.__init__(self, parent, size=size)
        self.palette = palette
        self.ipal = ipal
        self.viewwin = viewwin
        
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)

    def on_paint(self, event_unused):
        dc = wx.PaintDC(self)
        cw, ch = self.GetClientSize()
        ncolors = len(self.palette.colors)
        width = float(cw)/ncolors
        for c in range(0, ncolors):
            dc.SetPen(wx.Pen(wx.Colour(*self.palette.colors[c]), 1))
            dc.SetBrush(wx.Brush(wx.Colour(*self.palette.colors[c]), wx.SOLID))
            dc.DrawRectangle(int(c*width), 0, int(width+1), ch)

    def on_left_up(self, event_unused):
        self.viewwin.fire_command(id_set_palette, self.ipal)

class PalettesPanel(ScrolledPanel):
    def __init__(self, parent, palettes, viewwin, size=wx.DefaultSize):
        ScrolledPanel.__init__(self, parent, size=size)
        
        self.viewwin = viewwin
        self.palettes = palettes
        self.pal_height = 30
        
        self.sizer = wx.FlexGridSizer(len(self.palettes), 1)
        for i, pal in enumerate(self.palettes):
            self.a_palette = PaletteWin(self, pal, i, viewwin, size=(200, self.pal_height))
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
    def __init__(self, palettes, viewwin):
        wx.MiniFrame.__init__(self, None, title='Palettes', size=(250, 350),
            style=wx.DEFAULT_FRAME_STYLE)
        self.panel = PalettesPanel(self, palettes, viewwin)
