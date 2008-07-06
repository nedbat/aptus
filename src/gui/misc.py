""" Miscellaneous stuff for the Aptus GUI.
"""

from aptus.importer import importer
wx = importer("wx")
from wx.lib.evtmgr import eventManager


class AptusToolFrame(wx.MiniFrame):
    """ A frame for tool windows.
    """
    # This handles getting the styles right for miniframes.
    def __init__(self, parent_unused, title='', size=wx.DefaultSize):
        # If I pass parent into MiniFrame, the focus gets messed up, and keys don't work anymore!?
        wx.MiniFrame.__init__(self, None, title=title, size=size,
            style=wx.DEFAULT_FRAME_STYLE
            )


class ListeningWindowMixin:
    """ Adds event listening to a window, and deregisters automatically on
        destruction.
    """
    def __init__(self):
        self.listeners = set()
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

    def on_destroy(self, event_unused):
        for l in self.listeners:
            eventManager.DeregisterListener(l)

    def register_listener(self, fn, evt, sender):
        eventManager.Register(fn, evt, sender)
        self.listeners.add(fn)

    def deregister_listener(self, fn):
        eventManager.DeregisterListener(fn)
        
        if fn in self.listeners:
            self.listeners.remove(fn)
