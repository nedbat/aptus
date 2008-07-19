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
        # The eventManager listeners we've registered
        self.listeners = set()
        # The raw events we've bound to
        self.events = set()

        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

    def on_destroy(self, event):
        for l in self.listeners:
            eventManager.DeregisterListener(l)
        for other_win, evt in self.events:
            other_win.Unbind(evt)
        event.Skip()

    def register_listener(self, fn, evt, sender):
        """ Register a listener for an eventManager event. This will be automatically
            de-registered when self is destroyed.
        """
        eventManager.Register(fn, evt, sender)
        self.listeners.add(fn)

    def deregister_listener(self, fn):
        """ Deregister a previously registered listener.
        """
        eventManager.DeregisterListener(fn)
        
        if fn in self.listeners:
            self.listeners.remove(fn)

    def bind_to_other(self, other_win, evt, fn):
        """ Bind to a standard wxPython event on another window.  This will be
            automatically Unbind'ed when self is destroyed.
        """
        other_win.Bind(evt, fn)
        self.events.add((other_win, evt))
