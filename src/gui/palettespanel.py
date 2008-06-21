""" Palette visualization for Aptus.
"""

from aptus.importer import importer
wx = importer("wx")
from wx.lib.scrolledpanel import ScrolledPanel

class PalettesPanel(wx.Panel):
    def __init__(self, parent, palettes, size=wx.DefaultSize):
        wx.Panel.__init__(self, parent, size=size)
        
        self.palettes = palettes
        
        scrolledpanel = ScrolledPanel(self, -1, #size=(140, 300),
                    style = wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)

        self.panel = wx.Panel(scrolledpanel, size=(200,300))
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_SIZE, self.on_size)

        self.stripe_height = 50

        #box = wx.BoxSizer(wx.VERTICAL)
        #box.Add(self.panel)
        #scrolledpanel.SetSizer(box)
        scrolledpanel.SetAutoLayout(True)
        scrolledpanel.SetupScrolling()
        
    def on_paint(self, event_unused):
        dc = wx.PaintDC(self.panel)
        cw_unused, ch = self.GetClientSize()
        for y, pal in enumerate(self.palettes):
            self.paint_palette(dc, pal, y*self.stripe_height, self.stripe_height)
            
    def paint_palette(self, dc, pal, y0, height):
        cw, ch_unused = self.GetClientSize()
        ncolors = len(pal.colors)
        width = float(cw)/ncolors
        for c in range(0, ncolors):
            dc.SetPen(wx.Pen(wx.Colour(*pal.colors[c]), 1))
            dc.SetBrush(wx.Brush(wx.Colour(*pal.colors[c]), wx.SOLID))
            dc.DrawRectangle(int(c*width), y0, int(width+1), height)
    
    def on_size(self, event_unused):
        self.Refresh()


class PalettesFrame(wx.MiniFrame):
    def __init__(self, palettes):
        wx.MiniFrame.__init__(self, None, title='Palettes', size=(250, 350),
            style=wx.DEFAULT_MINIFRAME_STYLE|wx.CLOSE_BOX)
        self.panel = PalettesPanel(self, palettes)
