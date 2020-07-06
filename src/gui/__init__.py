""" Aptus GUI
    http://nedbatchelder.com/code/aptus
"""

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
        return True

    def new_window(self, *args, **kwargs):
        frame = AptusMainFrame(*args, **kwargs)
        frame.Show()
        return frame


def main(args):
    """ The main for the Aptus GUI.
    """
    AptusGuiApp(args).MainLoop()
