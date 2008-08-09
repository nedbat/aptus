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
        { 'label': 'count', 'key': 'count', 'fmt': '%.2f'},
        { 'label': 'color', 'key': 'color', },
        ]
        
    def __init__(self, parent, viewwin):
        """ Create a PointInfoPanel, with `parent` as its parent, and `viewwin` as
            the window to track.
        """
        DictPanel.__init__(self, parent, self.infomap)
        ListeningWindowMixin.__init__(self)
        
        self.viewwin = viewwin
        
        self.register_listener(self.update_info, EVT_APTUS_RECOMPUTED, self.viewwin)
        self.register_listener(self.update_info, EVT_APTUS_INDICATEPOINT, self.viewwin)

        # Need to call update_info after the window appears, so that the widths of
        # the text controls can be set properly.  Else, it all appears left-aligned.
        wx.CallAfter(self.update_info)
        
    def update_info(self, event=None):
        # Different events will trigger this, be flexible about how to get the
        # mouse position.
        if event and hasattr(event, 'point'):
            pt = event.point
        else:
            pt = self.viewwin.ScreenToClient(wx.GetMousePosition())
        info = self.viewwin.get_point_info(pt)
        if info:
            self.update(info)

        # Need to let the main window handle the event too.
        if event:
            event.Skip()    


class PointInfoFrame(AptusToolFrame):
    def __init__(self, mainframe, viewwin):
        AptusToolFrame.__init__(self, mainframe, title='Point info', size=(180,180))
        self.panel = PointInfoPanel(self, viewwin)
