""" A panel for displaying information from a dictionary.
"""

from aptus.importer import importer
wx = importer("wx")

import locale
locale.setlocale(locale.LC_ALL, "")


class DictPanel(wx.Panel):
    """ A panel displaying the contents of a dictionary.
    """
    def __init__(self, parent, keymap):
        wx.Panel.__init__(self, parent)
        
        self.keymap = keymap
        self.keywins = []
        
        grid = wx.FlexGridSizer(cols=2, vgap=1, hgap=3)
        for keyd in self.keymap:
            label = wx.StaticText(self, -1, keyd['label'] + ':')
            value = wx.StaticText(self, -1, style=wx.ALIGN_RIGHT)
            grid.Add(label)
            grid.Add(value)
            self.keywins.append((keyd['key'], value))
        
        sizer = wx.BoxSizer()
        sizer.Add(grid, flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.LEFT, border=3)
        self.SetSizer(sizer)
        sizer.Fit(self)
        
    def update(self, d):
        """ Update the values in the panel, from the dictionary d.
        """
        maxw = 50
        for key, valwin in self.keywins:
            val = d[key]
            if isinstance(val, (int, long)):
                s = locale.format("%d", val, True)
            elif isinstance(val, float):
                s = "%.08f" % val
            else:
                s = str(val)
            valwin.SetLabel(s)
            w = valwin.GetSizeTuple()[0]
            maxw = max(maxw, w)

        for statname, valwin in self.keywins:
            valwin.SetSize((maxw, -1))
            valwin.SetMinSize((maxw, -1))
