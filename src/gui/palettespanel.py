""" Palette visualization for Aptus.
"""

from aptus.importer import importer
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame

wx = importer("wx")
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.evtmgr import eventManager

class PaletteWin(wx.Window):
    """ A window for displaying a single palette.  Handles click events to
        change the palette in the view window.
    """
    def __init__(self, parent, palette, ipal, viewwin, size=wx.DefaultSize):
        wx.Window.__init__(self, parent, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.palette = palette
        self.ipal = ipal
        self.viewwin = viewwin
        self.selected = False
        
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)

    def on_paint(self, event_unused):
        # Geometry: client size and margin widths.
        cw, ch = self.GetClientSize()
        mt, mr, mb, ml = 3, 6, 3, 6

        dc = wx.AutoBufferedPaintDC(self)

        # Paint the background.
        if self.selected:
            color = wx.Colour(128, 128, 128)
        else:
            color = wx.Colour(255, 255, 255)
            
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush(color, wx.SOLID))
        dc.DrawRectangle(0, 0, cw, ch)

        # Paint the palette
        ncolors = len(self.palette.colors)
        width = float(cw-mr-ml-2)/ncolors
        for c in range(0, ncolors):
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.SetBrush(wx.Brush(wx.Colour(*self.palette.colors[c]), wx.SOLID))
            dc.DrawRectangle(int(c*width)+ml+1, mt+1, int(width+1), ch-mt-mb-2)

        # Paint the black outline
        dc.SetPen(wx.BLACK_PEN)
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(ml, mt, cw-ml-mr, ch-mt-mb)

    def on_size(self, event_unused):
        # Since the painting changes everywhere when the width changes, refresh
        # on size changes.
        self.Refresh()

    def on_left_up(self, event_unused):
        # Left click: tell the view window to switch to my palette.
        self.viewwin.fire_command(id_set_palette, self.ipal)


class PalettesPanel(ScrolledPanel):
    """ A panel displaying a number of palettes.
    """
    def __init__(self, parent, palettes, viewwin, size=wx.DefaultSize):
        ScrolledPanel.__init__(self, parent, size=size)
        
        self.viewwin = viewwin
        self.palettes = palettes
        self.pal_height = 30
        self.selected = -1
        
        self.palwins = []
        self.sizer = wx.FlexGridSizer(len(self.palettes), 1)
        for i, pal in enumerate(self.palettes):
            palwin = PaletteWin(self, pal, i, viewwin, size=(200, self.pal_height))
            self.sizer.Add(palwin, flag=wx.EXPAND)
            self.palwins.append(palwin)
            
        self.sizer.AddGrowableCol(0)
        self.sizer.SetFlexibleDirection(wx.HORIZONTAL)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(True)
        self.SetupScrolling()

        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

        eventManager.Register(self.on_coloring_changed, EVT_APTUS_COLORING_CHANGED, self.viewwin)
        self.on_coloring_changed(None)
        
    def on_destroy(self, event_unused):
        eventManager.DeregisterListener(self.on_coloring_changed)

    def on_coloring_changed(self, event_unused):
        # When the view window's coloring changes, see if the palette changed.
        if self.viewwin.palette_index != self.selected:
            # Change which of the palettes is selected.
            self.palwins[self.selected].selected = False
            self.selected = self.viewwin.palette_index
            self.palwins[self.selected].selected = True
            self.ScrollChildIntoView(self.palwins[self.selected])
            self.Refresh()

            
class PalettesFrame(AptusToolFrame):
    """ The top level frame for the palettes list.
    """
    def __init__(self, palettes, viewwin):
        AptusToolFrame.__init__(self, viewwin, title='Palettes', size=(250, 350))
        self.panel = PalettesPanel(self, palettes, viewwin)
