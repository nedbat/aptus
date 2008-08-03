""" A panel to display the Julia set for the currently hovered point in the main
    window.
"""

from aptus.gui.computepanel import MiniComputePanel
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame, ListeningWindowMixin
from aptus.importer import importer

wx = importer("wx")

class JuliaPanel(MiniComputePanel, ListeningWindowMixin):
    """ A panel displaying the Julia set for the current point in another window.
    """
    
    def __init__(self, parent, viewwin, size=wx.DefaultSize):
        """ Create a JuliaPanel, with `parent` as its parent, and `viewwin` as
            the window to track.
        """
        MiniComputePanel.__init__(self, parent, size=size)
        ListeningWindowMixin.__init__(self)
        
        self.viewwin = viewwin
        
        self.register_listener(self.on_coloring_changed, EVT_APTUS_COLORING_CHANGED, self.viewwin)
        self.register_listener(self.draw_julia, EVT_APTUS_INDICATEPOINT, self.viewwin)

        self.compute.center, self.compute.diam = (0.0,0.0), (3.0,3.0)
        self.compute.mode = 'julia'

        self.on_coloring_changed(None)

        # Need to call update_info after the window appears, so that the widths of
        # the text controls can be set properly.  Else, it all appears left-aligned.
        wx.CallAfter(self.draw_julia)

    def draw_julia(self, event=None):
        # Different events will trigger this, be flexible about how to get the
        # mouse position.
        if event and hasattr(event, 'point'):
            pt = event.point
        else:
            pt = self.viewwin.ScreenToClient(wx.GetMousePosition())

        pt_info = self.viewwin.get_point_info(pt)
        if pt_info:
            self.compute.rijulia = pt_info['r'], pt_info['i']
            self.compute.iter_limit = 999
        else:
            self.compute.rijulia = 0,0
        self.compute.create_mandel()
        self.computation_changed()
        
        # Need to let the main window handle the event too.
        if event:
            event.Skip()    

    def on_coloring_changed(self, event_unused):
        if self.compute.copy_coloring(self.viewwin.compute):
            self.coloring_changed()


class JuliaFrame(AptusToolFrame):
    def __init__(self, mainframe, viewwin):
        AptusToolFrame.__init__(self, mainframe, title='Julia Set', size=(180,180))
        self.panel = JuliaPanel(self, viewwin)
