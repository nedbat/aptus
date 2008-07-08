""" A panel to display information about the pointed-to point in the main
    window.
"""

from aptus.importer import importer
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame, ListeningWindowMixin
from aptus.gui.dictpanel import DictPanel

wx = importer("wx")

class PointInfoPanel(DictPanel, ListeningWindowMixin):
    """ A panel displaying information about the current point in the main window.
    """
    
    infomap = [
        { 'label': 'x', 'key': 'x', },
        { 'label': 'y', 'key': 'y', },
        { 'label': 'r', 'key': 'r', },
        { 'label': 'i', 'key': 'i', },
        { 'label': 'count', 'key': 'count', },
        { 'label': 'color', 'key': 'color', },
        ]
        
    def __init__(self, parent, viewwin):
        """ Create a PointInfoPanel, with `parent` as its parent, and `viewwin` as
            the window to track.
        """
        DictPanel.__init__(self, parent, self.infomap)
        ListeningWindowMixin.__init__(self)
        
        self.viewwin = viewwin
        
        self.viewwin.Bind(wx.EVT_MOTION, self.on_viewwin_motion)
        
        # Need to call set_values after the window appears, so that the widths of
        # the text controls can be set properly.  Else, it all appears left-aligned.
        #wx.CallAfter(self.on_viewwin_motion)
        
    def on_viewwin_motion(self, event):
        mx, my = event.GetPosition()
        info = self.viewwin.get_point_info((mx, my))
        self.update(info)
        event.Skip()    # Need to let the main window handle the event too.


class PointInfoFrame(AptusToolFrame):
    def __init__(self, mainwin):
        AptusToolFrame.__init__(self, mainwin, title='Point info', size=(180,180))
        self.panel = PointInfoPanel(self, mainwin)
