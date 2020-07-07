import os
import os.path
import re

import wx
import wx.aui

from aptus import data_file
from aptus.gui.ids import *
from aptus.gui.viewpanel import AptusViewPanel
from aptus.options import AptusOptions


class AptusMainFrame(wx.Frame):
    """ The main window frame of the Aptus app.
    """
    def __init__(self, args=None, compute=None, size=None):
        """ Create an Aptus main GUI frame.  `args` is an argv-style list of
            command-line arguments. `compute` is an existing compute object to
            copy settings from.
        """
        wx.Frame.__init__(self, None, -1, 'Aptus')

        # Make the panel
        self.panel = AptusViewPanel(self)

        if args:
            opts = AptusOptions(self.panel.compute)
            opts.read_args(args)
        if compute:
            self.panel.compute.copy_all(compute)

        if size:
            self.panel.compute.size = size

        self.panel.compute.supersample = 1

    def Show(self, show=True):
        # Override Show so we can set the view properly.
        if show:
            self.SetClientSize(self.panel.compute.size)
            self.panel.set_view()
            wx.Frame.Show(self, True)
            self.panel.SetFocus()
        else:
            wx.Frame.Show(self, False)

    def message(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Aptus', wx.OK | wx.ICON_WARNING)
        dlg.ShowModal()
        dlg.Destroy()
