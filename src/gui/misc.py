""" Miscellaneous stuff for the Aptus GUI.
"""

from aptus.importer import importer
wx = importer("wx")

class AptusToolFrame(wx.MiniFrame):
    """ A frame for tool windows.
    """
    # This handles getting the styles right for miniframes.
    def __init__(self, parent, title='', size=wx.DefaultSize):
        # If I pass parent into MiniFrame, the focus gets messed up, and keys don't work anymore!?
        wx.MiniFrame.__init__(self, None, title=title, size=size,
            style=wx.DEFAULT_FRAME_STYLE
            )
