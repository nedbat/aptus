""" Aptus GUI
    http://nedbatchelder.com/code/aptus
"""

import sys

from aptus import data_file
from aptus.gui.mainframe import AptusMainFrame

# Import third-party packages.
import wx
import wx.adv

class AptusGuiApp(wx.App):
    def __init__(self, args):
        self.args = args
        wx.App.__init__(self)

    def OnInit(self):
        frame = self.new_window(self.args)
        SplashScreen(frame).Show()
        return True

    def new_window(self, *args, **kwargs):
        frame = AptusMainFrame(*args, **kwargs)
        frame.Show()
        return frame


class SplashScreen(wx.adv.SplashScreen):
    """ A nice splash screen.
    """
    def __init__(self, parent=None):
        bitmap = wx.Image(name=data_file("splash.png")).ConvertToBitmap()
        splash_style = wx.adv.SPLASH_TIMEOUT | wx.adv.SPLASH_NO_CENTRE
        style = wx.FRAME_NO_TASKBAR | wx.STAY_ON_TOP | wx.NO_BORDER
        wx.adv.SplashScreen.__init__(self, bitmap, splash_style, 2000, parent, style=style)
        self.Move(parent.ClientToScreen((0, 0)) + (50, 50))
        self.Bind(wx.EVT_CLOSE, self.on_exit)

    def on_exit(self, evt_unused):
        self.alpha = 255
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.fade_some)
        self.timer.Start(25)

        # Don't actually destroy the window or skip the event, so the timer can
        # run, and fade the window out..

    def fade_some(self, evt_unused):
        self.alpha -= 16
        if self.alpha <= 0:
            self.timer.Stop()
            del self.timer
            self.Destroy()
        else:
            self.SetTransparent(self.alpha)


def main(argv=None):
    """ The main for the Aptus GUI.
    """
    if argv is None:
        argv = sys.argv[1:]
    AptusGuiApp(argv).MainLoop()
