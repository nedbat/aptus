import os
import os.path
import re

import wx
import wx.aui

from aptus import data_file
from aptus.gui.ids import *
from aptus.gui.viewpanel import AptusViewPanel
from aptus.gui.misc import AptusToolableFrameMixin
from aptus.options import AptusOptions


class AptusMainFrame(wx.Frame, AptusToolableFrameMixin):
    """ The main window frame of the Aptus app.
    """
    def __init__(self, args=None, compute=None, size=None):
        """ Create an Aptus main GUI frame.  `args` is an argv-style list of
            command-line arguments. `compute` is an existing compute object to
            copy settings from.
        """
        wx.Frame.__init__(self, None, -1, 'Aptus')
        AptusToolableFrameMixin.__init__(self)

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

        if 0:
            # Experimental AUI support
            self.auimgr = wx.aui.AuiManager()
            self.auimgr.SetManagedWindow(self)

            self.auimgr.AddPane(self.panel, wx.aui.AuiPaneInfo().Name("grid_content").
                              PaneBorder(False).CenterPane())

            from aptus.gui import pointinfo
            self.pointinfo_tool = pointinfo.PointInfoPanel(self, self.panel)

            self.auimgr.AddPane(self.pointinfo_tool, wx.aui.AuiPaneInfo().
                              Name("pointinfo").Caption("Point info").
                              Right().Layer(1).Position(1).CloseButton(True))

            self.auimgr.Update()

        # Set the window icon
        ib = wx.IconBundle()
        ib.AddIcon(data_file("icon48.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIcon(data_file("icon32.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIcon(data_file("icon16.png"), wx.BITMAP_TYPE_ANY)
        self.SetIcons(ib)

        # Bind commands
        self.Bind(wx.EVT_MENU, self.cmd_new, id=id_new)
        self.Bind(wx.EVT_MENU, self.cmd_window_size, id=id_window_size)

        # Auxilliary frames.
        self.stats_tool = None
        self.pointinfo_tool = None

        # Files can be dropped here.
        self.SetDropTarget(MainFrameFileDropTarget(self))

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

    # Command handlers.

    def show_file_dialog(self, dlg):
        """ Show a file dialog, and do some post-processing on the result.
            Returns a pair: type, path.
            Type is one of the extensions from the wildcard choices.
        """
        if dlg.ShowModal() == wx.ID_OK:
            pth = dlg.Path
            ext = os.path.splitext(pth)[1].lower()
            idx = dlg.FilterIndex
            wildcards = dlg.Wildcard.split('|')
            wildcard = wildcards[2*idx+1]
            if wildcard == '*.*':
                if ext:
                    typ = ext[1:]
                else:
                    typ = ''
            elif '*'+ext in wildcards:
                # The extension of the file is a recognized extension:
                # Use it regardless of the file type chosen in the picker.
                typ = ext[1:]
            else:
                typ = wildcard.split('.')[-1].lower()
            if ext == '' and typ != '':
                pth += '.' + typ
            return typ, pth
        else:
            return None, None

    def cmd_new(self, event_unused):
        return wx.GetApp().new_window()

    def cmd_window_size(self, event_unused):
        cur_size = "%d x %d" % tuple(self.GetClientSize())
        dlg = wx.TextEntryDialog(self.GetTopLevelParent(), "Window size",
            "New window size?", cur_size)

        if dlg.ShowModal() == wx.ID_OK:
            new_size = dlg.GetValue().strip()
            m = re.match(r"(?P<w>\d+)\s*[x, ]\s*(?P<h>\d+)|s/(?P<mini>[\d.]+)", new_size)
            if m:
                if m.group('mini') is not None:
                    factor = float(m.group('mini'))
                    screen_w, screen_h = wx.GetDisplaySize()
                    w, h = screen_w/factor, screen_h/factor
                elif m.group('w') is not None:
                    w, h = int(m.group('w')), int(m.group('h'))
                self.SetClientSize((w,h))
        dlg.Destroy()


class MainFrameFileDropTarget(wx.FileDropTarget):
    """A drop target so files can be opened by dragging them to the Aptus window.

    The first file opens in the current window, the rest open new windows.

    """
    def __init__(self, frame):
        wx.FileDropTarget.__init__(self)
        self.frame = frame

    def OnDropFiles(self, x, y, filenames):
        self.frame.open_file(filenames[0])
        for filename in filenames[1:]:
            frame = self.frame.cmd_new(None)
            frame.open_file(filename)
