""" Aptus GUI
    http://nedbatchelder.com/code/aptus
    Copyright 2007-2008, Ned Batchelder
"""

from aptus.importer import importer
from aptus.gui.mainframe import AptusMainFrame

# Import third-party packages.
wx = importer('wx')

class AptusGuiApp(wx.PySimpleApp):
    def __init__(self, args):
        wx.PySimpleApp.__init__(self)
        self.new_window(args)
            
    def new_window(self, *args, **kwargs):
        AptusMainFrame(*args, **kwargs).Show()


def main(args):
    """ The main for the Aptus GUI.
    """
    AptusGuiApp(args).MainLoop()
