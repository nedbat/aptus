""" A panel for displaying information from a dictionary.
"""

import wx

# Set the locale to the user's default.
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
            self.keywins.append((keyd, value))

        sizer = wx.BoxSizer()
        sizer.Add(grid, flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.LEFT, border=3)
        self.SetSizer(sizer)
        sizer.Fit(self)

    def update(self, dval):
        """ Update the values in the panel, from the dictionary `dval`.
        """
        maxw = 50
        for keyd, valwin in self.keywins:
            val = dval[keyd['key']]
            if isinstance(val, int):
                s = locale.format(keyd.get('fmt', "%d"), val, grouping=True)
            elif isinstance(val, float):
                s = locale.format(keyd.get('fmt', "%.10e"), val, grouping=True)
            elif val is None:
                s = u"\u2014"   # emdash
            else:
                s = str(val)
            valwin.SetLabel(s)
            w = valwin.GetSize()[0]
            maxw = max(maxw, w)

        for _, valwin in self.keywins:
            valwin.SetSize((maxw, -1))
            valwin.SetMinSize((maxw, -1))
