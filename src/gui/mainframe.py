from aptus import data_file

from aptus.importer import importer
from aptus.options import AptusOptions

from aptus.gui.ids import *
from aptus.gui.viewpanel import AptusViewPanel
from aptus.gui.misc import AptusToolableFrameMixin

wx = importer("wx")
import wx.aui

import os, os.path, re


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
        ib.AddIconFromFile(data_file("icon48.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIconFromFile(data_file("icon32.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIconFromFile(data_file("icon16.png"), wx.BITMAP_TYPE_ANY)
        self.SetIcons(ib)

        # Bind commands
        self.Bind(wx.EVT_MENU, self.cmd_new, id=id_new)
        self.Bind(wx.EVT_MENU, self.cmd_save, id=id_save)
        self.Bind(wx.EVT_MENU, self.cmd_help, id=id_help)
        self.Bind(wx.EVT_MENU, self.cmd_fullscreen, id=id_fullscreen)
        self.Bind(wx.EVT_MENU, self.cmd_window_size, id=id_window_size)
        self.Bind(wx.EVT_MENU, self.cmd_show_youarehere, id=id_show_youarehere)
        self.Bind(wx.EVT_MENU, self.cmd_show_palettes, id=id_show_palettes)
        self.Bind(wx.EVT_MENU, self.cmd_show_stats, id=id_show_stats)
        self.Bind(wx.EVT_MENU, self.cmd_show_pointinfo, id=id_show_pointinfo)
        self.Bind(wx.EVT_MENU, self.cmd_show_julia, id=id_show_julia)

        # Auxilliary frames.
        self.youarehere_tool = None
        self.palettes_tool = None
        self.stats_tool = None
        self.pointinfo_tool = None
        self.julia_tool = None

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
        wx.GetApp().new_window()
        
    def cmd_save(self, event_unused):
        wildcard = (
            "PNG image (*.png)|*.png|"     
            "Aptus state (*.aptus)|*.aptus|"
            "All files (*.*)|*.*"
            )

        dlg = wx.FileDialog(
            self, message="Save", defaultDir=os.getcwd(), 
            defaultFile="", style=wx.SAVE|wx.OVERWRITE_PROMPT, wildcard=wildcard, 
            )

        typ, pth = self.show_file_dialog(dlg)
        if typ:
            if typ == 'png':
                self.panel.write_png(pth)
            elif typ == 'aptus':
                self.panel.write_aptus(pth)
            else:
                self.message("Don't understand how to write file '%s'" % pth)
                
    def cmd_help(self, event_unused):
        from aptus.gui.help import HelpDlg
        dlg = HelpDlg(self)
        dlg.ShowModal()

    def cmd_fullscreen(self, event_unused):
        self.ShowFullScreen(not self.IsFullScreen())
    
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

    def cmd_show_youarehere(self, event_unused):
        """ Toggle the presence of the YouAreHere tool.
        """
        if self.youarehere_tool:
            self.youarehere_tool.Destroy()
        else:
            from aptus.gui import youarehere
            self.youarehere_tool = youarehere.YouAreHereFrame(self, self.panel)
            self.youarehere_tool.Show()

    def cmd_show_palettes(self, event_unused):
        """ Toggle the presence of the Palettes tool.
        """
        if self.palettes_tool:
            self.palettes_tool.Destroy()
        else:
            from aptus.gui import palettespanel
            from aptus.palettes import all_palettes
            self.palettes_tool = palettespanel.PalettesFrame(self, all_palettes, self.panel)
            self.palettes_tool.Show()

    def cmd_show_stats(self, event_unused):
        """ Toggle the presence of the Stats tool.
        """
        if self.stats_tool:
            self.stats_tool.Destroy()
        else:
            from aptus.gui import statspanel
            self.stats_tool = statspanel.StatsFrame(self, self.panel)
            self.stats_tool.Show()

    def cmd_show_pointinfo(self, event_unused):
        """ Toggle the presence of the PointInfo tool.
        """
        if self.pointinfo_tool:
            self.pointinfo_tool.Destroy()
        else:
            from aptus.gui import pointinfo
            self.pointinfo_tool = pointinfo.PointInfoFrame(self, self.panel)
            self.pointinfo_tool.Show()

    def cmd_show_julia(self, event_unused):
        """ Toggle the presence of the Julia tool.
        """
        if self.panel.compute.mode == 'mandelbrot':
            if self.julia_tool:
                self.julia_tool.Destroy()
            else:
                from aptus.gui import juliapanel
                self.julia_tool = juliapanel.JuliaFrame(self, self.panel)
                self.julia_tool.Show()
