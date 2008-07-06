""" A panel to display computation statistics.
"""

from aptus.importer import importer
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame

wx = importer("wx")
from wx.lib.evtmgr import eventManager

import locale
locale.setlocale(locale.LC_ALL, "")


class StatsPanel(wx.Panel):
    """ A panel displaying the statistics from a view window.  It listens for
        recomputations, and updates automatically.
    """
    
    statmap = [
        { 'label': 'Min iteration', 'stat': 'miniter', },
        { 'label': 'Max iteration', 'stat': 'maxiter', },
        { 'label': 'Total iterations', 'stat': 'totaliter', },
        { 'label': 'Total cycles', 'stat': 'totalcycles', },
        { 'label': 'Shortest cycle', 'stat': 'minitercycle', },
        { 'label': 'Longest cycle', 'stat': 'maxitercycle', },
        { 'label': 'Maxed points', 'stat': 'maxedpoints', },
        { 'label': 'Computed points', 'stat': 'computedpoints', },
        { 'label': 'Boundaries traced', 'stat': 'boundaries', },
        { 'label': 'Boundaries filled', 'stat': 'boundariesfilled', },
        ]
        
    def __init__(self, parent, viewwin):
        """ Create a StatsPanel, with `parent` as its parent, and `viewwin` as
            the window to track.
        """
        wx.Panel.__init__(self, parent)
        self.viewwin = viewwin
        self.statwins = []
        
        grid = wx.FlexGridSizer(cols=2, vgap=1, hgap=3)
        for statd in self.statmap:
            label = wx.StaticText(self, -1, statd['label'] + ':')
            value = wx.StaticText(self, -1, style=wx.ALIGN_RIGHT)
            grid.Add(label)
            grid.Add(value)
            self.statwins.append((statd['stat'], value))
        
        sizer = wx.BoxSizer()
        sizer.Add(grid, flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.LEFT, border=3)
        self.SetSizer(sizer)
        sizer.Fit(self)
        eventManager.Register(self.set_values, EVT_APTUS_RECOMPUTED, self.viewwin)

        # Need to call set_values after the window appears, so that the widths of
        # the text controls can be set properly.  Else, it all appears left-aligned.
        wx.CallAfter(self.set_values)
        
    def set_values(self, event_unused=None):
        stats = self.viewwin.m.eng.get_stats()
        maxw = 50
        for statname, valwin in self.statwins:
            s = locale.format("%d", stats[statname], True)
            valwin.SetLabel(s)
            w = valwin.GetSizeTuple()[0]
            maxw = max(maxw, w)

        for statname, valwin in self.statwins:
            valwin.SetSize((maxw, -1))
            valwin.SetMinSize((maxw, -1))


class StatsFrame(AptusToolFrame):
    def __init__(self, mainwin):
        AptusToolFrame.__init__(self, mainwin, title='Statistics', size=(180,180))
        self.panel = StatsPanel(self, mainwin)
