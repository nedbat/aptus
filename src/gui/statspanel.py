""" A panel to display computation statistics.
"""

from aptus.compute import ComputeStats
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame, ListeningWindowMixin
from aptus.gui.dictpanel import DictPanel

import wx

import locale
locale.setlocale(locale.LC_ALL, "")


class StatsPanel(DictPanel, ListeningWindowMixin):
    """ A panel displaying the statistics from a view window.  It listens for
        recomputations, and updates automatically.
    """

    def __init__(self, parent, viewwin):
        """ Create a StatsPanel, with `parent` as its parent, and `viewwin` as
            the window to track.
        """
        DictPanel.__init__(self, parent, ComputeStats.statmap)
        ListeningWindowMixin.__init__(self)

        self.viewwin = viewwin
        self.register_listener(self.on_recomputed, EVT_APTUS_RECOMPUTED, self.viewwin)

        # Need to call on_recomputed after the window appears, so that the widths of
        # the text controls can be set properly.  Else, it all appears left-aligned.
        wx.CallAfter(self.on_recomputed)

    def on_recomputed(self, event_unused=None):
        stats = self.viewwin.get_stats()
        self.update(stats)


class StatsFrame(AptusToolFrame):
    def __init__(self, mainframe, viewwin):
        AptusToolFrame.__init__(self, mainframe, title='Statistics', size=(180,250))
        self.panel = StatsPanel(self, viewwin)
