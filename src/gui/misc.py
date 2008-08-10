""" Miscellaneous stuff for the Aptus GUI.
"""

from aptus.importer import importer
wx = importer("wx")
from wx.lib.evtmgr import eventManager


class AptusToolableFrameMixin:
    """ A mixin to add to a frame.  Tool windows can be attached to this, and
        will behave nicely (minimizing, etc).
    """
    def __init__(self):
        self.toolwins = []
        #self.Bind(wx.EVT_ACTIVATE, self.on_activate)
        self.Bind(wx.EVT_ICONIZE, self.on_iconize)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def add_toolwin(self, toolwin):
        self.toolwins.append(toolwin)

    def remove_toolwin(self, toolwin):
        self.toolwins.remove(toolwin)

    #def on_activate(self, event):
    #    print "on_activate:", event, event.GetActive(), wx.GetApp().IsActive()
    #    event.Skip()

    def on_iconize(self, event):
        bshow = not event.Iconized()
        for toolwin in self.toolwins:
            toolwin.Show(bshow)
        event.Skip()

    def on_close(self, event):
        for toolwin in self.toolwins:
            toolwin.Close()
        event.Skip()


class AptusToolFrame(wx.MiniFrame):
    """ A frame for tool windows.
    """
    # This handles getting the styles right for miniframes.
    def __init__(self, mainframe, title='', size=wx.DefaultSize):
        # If I pass mainframe into MiniFrame, the focus gets messed up, and keys don't work anymore!?  Really, where?
        wx.MiniFrame.__init__(self, mainframe, title=title, size=size,
            style=wx.DEFAULT_MINIFRAME_STYLE|wx.CLOSE_BOX
            )
        self.mainframe = mainframe
        self.mainframe.add_toolwin(self)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
    
    def on_destroy(self, event_unused):
        self.mainframe.remove_toolwin(self)


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
