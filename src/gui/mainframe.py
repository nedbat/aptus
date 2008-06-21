from aptus import data_file

from aptus.importer import importer
from aptus.options import AptusOptions

from aptus.gui.ids import *
from aptus.gui.viewpanel import AptusViewPanel

wx = importer("wx")

import os, os.path


class AptusMainFrame(wx.Frame):
    """ The main window frame of the Aptus app.
    """
    def __init__(self, args=None):
        wx.Frame.__init__(self, None, -1, 'Aptus')

        # Make the panel
        self.panel = AptusViewPanel(self)
        
        if args:
            opts = AptusOptions(self.panel.m)
            opts.read_args(args)
        self.panel.supersample = 1
        
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
        self.Bind(wx.EVT_MENU, self.cmd_show_youarehere, id=id_show_youarehere)
        self.Bind(wx.EVT_MENU, self.cmd_show_palettes, id=id_show_palettes)
        
    def Show(self, show=True):
        # Override Show so we can set the view properly.
        if show:
            self.SetClientSize(self.panel.m.size)
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

    def cmd_show_youarehere(self, event_unused):
        from aptus.gui import youarehere 
        youarehere.YouAreHereFrame(self.panel).Show()

    def cmd_show_palettes(self, event_unused):
        from aptus.gui import palettespanel
        from aptus.palettes import all_palettes
        palettespanel.PalettesFrame(all_palettes).Show()
