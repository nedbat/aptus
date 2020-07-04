""" Help dialog for Aptus.
"""

from aptus import data_file, __version__
from aptus.options import AptusOptions

import numpy
from PIL import Image
import wx
import wx.lib.layoutf
import wx.html 

import webbrowser, sys


class HtmlDialog(wx.Dialog):
    """ A simple dialog for displaying HTML, with clickable links that launch
        a web browser, or change the page displayed in the dialog.
    """
    def __init__(self, parent, caption, pages, subs=None,
                 pos=wx.DefaultPosition, size=(500,530),
                 style=wx.DEFAULT_DIALOG_STYLE):
        wx.Dialog.__init__(self, parent, -1, caption, pos, size, style)
        if pos == (-1, -1):
            self.CenterOnScreen(wx.BOTH)

        self.pages = pages
        self.subs = subs or {}
        self.html = wx.html.HtmlWindow(self, -1)
        self.html.Bind(wx.html.EVT_HTML_LINK_CLICKED, self.on_link_clicked)
        ok = wx.Button(self, wx.ID_OK, "OK")
        ok.SetDefault()
        
        lc = wx.lib.layoutf.Layoutf('t=t#1;b=t5#2;l=l#1;r=r#1', (self,ok))
        self.html.SetConstraints(lc)
        self.set_page('interactive')
        
        lc = wx.lib.layoutf.Layoutf('b=b5#1;r=r5#1;w!80;h*', (self,))
        ok.SetConstraints(lc)
        
        self.SetAutoLayout(1)
        self.Layout()

    def on_link_clicked(self, event):
        url = event.GetLinkInfo().GetHref()
        if url.startswith('http:'):
            webbrowser.open(url)
        elif url.startswith('internal:'):
            self.set_page(url.split(':')[1])

    def set_page(self, pagename):
        html = self.pages['head'] + self.pages[pagename]
        html = html % self.subs
        self.html.SetPage(html)


# The help text

is_mac = ('wxMac' in wx.PlatformInfo)

TERMS = {
    'ctrl': 'cmd' if is_mac else 'ctrl',
    'iconsrc': data_file('icon48.png'),
    'version': __version__,
    'python_version': sys.version,
    'wx_version': wx.__version__,
    'numpy_version': numpy.__version__,
    'pil_version': Image.__version__,
    }
    

HELP_PAGES = {
    'head': """\
        <table width='100%%'>
        <tr>
            <td width='50' valign='top'><img src='%(iconsrc)s'/></td>
            <td valign='top'>
                <b>Aptus %(version)s</b>, Mandelbrot set explorer.<br>
                Copyright 2007-2010, Ned Batchelder.<br>
                <a href='http://nedbatchelder.com/code/aptus'>http://nedbatchelder.com/code/aptus</a>
            </td>
        </tr>
        </table>
        
        <p>
            <a href='internal:interactive'>Interactive</a> |
            <a href='internal:command'>Command line</a> |
            <a href='internal:about'>About</a></p>
        <hr>
        """,

    'interactive': """
        <p><b>Interactive controls:</b></p>
        
        <blockquote>
        <b>a</b>: set the angle of rotation.<br>
        <b>c</b>: toggle continuous coloring.<br>
        <b>f</b>: toggle full-screen display.<br>
        <b>h</b> or <b>?</b>: show this help.<br>
        <b>i</b>: set the limit on iterations.<br>
        <b>j</b>: jump among a few pre-determined locations.<br>
        <b>n</b>: create a new window.<br>
        <b>o</b>: open a saved settings or image file.<br>
        <b>r</b>: redraw the current image.<br>
        <b>s</b>: save the current image or settings.<br>
        <b>w</b>: set the window size.<br>
        <b>&lt;</b> or <b>&gt;</b>: switch to the next palette.<br>
        <b>,</b> or <b>.</b>: cycle the current palette one color.<br>
        <b>;</b> or <b>'</b>: stretch the palette colors (+%(ctrl)s: just a little), if continuous.<br>
        <b>[</b> or <b>]</b>: adjust the hue of the palette (+%(ctrl)s: just a little).<br>
        <b>{</b> or <b>}</b>: adjust the saturation of the palette (+%(ctrl)s: just a little).<br>
        <b>0</b> (zero): reset all palette adjustments.<br>
        <b>space</b>: drag mode: click to drag the image to a new position.<br>
        <b>left-click</b>: zoom in (+%(ctrl)s: just a little).<br>
        <b>right-click</b>: zoom out (+%(ctrl)s: just a little).<br>
        <b>left-drag</b>: select a new rectangle to display.<br>
        <b>middle-drag</b>: drag the image to a new position.<br>
        <b>shift</b>: indicate a point of interest for Julia set and point info.
        </blockquote>
        
        <p><b>Tool windows: press a key to toggle on and off:</b></p>
        
        <blockquote>
        <b>shift-j</b>: Show a Julia set for the current (shift-hovered) point.<br>
        <b>l (ell)</b>: Show zoom snapshots indicating the current position.<br>
        <b>p</b>: Show a list of palettes that can be applied to the current view.<br>
        <b>q</b>: Show point info for the current (shift-hovered) point.<br>
        <b>v</b>: Show statistics for the latest calculation.
        </blockquote>
        """,

    'command': """
        <p>On the command line, use <tt><b>--help</b></tt> to see options:</p>
        <pre>""" + AptusOptions(None).options_help() + "</pre>",
        
    'about': """
        <p>Built with
        <a href='http://python.org'>Python</a>, <a href='http://wxpython.org'>wxPython</a>,
        <a href='http://numpy.scipy.org/'>numpy</a>, and
        <a href='http://www.pythonware.com/library/pil/handbook/index.htm'>PIL</a>.</p>
        
        <p>Thanks to Rob McMullen and Paul Ollis for help with the drawing code.</p>
        
        <hr>
        <p>Installed versions:</p>
        <p>
        Aptus: %(version)s<br>
        Python: %(python_version)s<br>
        wx: %(wx_version)s<br>
        numpy: %(numpy_version)s<br>
        PIL: %(pil_version)s
        </p>
        """,
    }


class HelpDlg(HtmlDialog):
    """ The help dialog for Aptus.
    """
    def __init__(self, parent):
        HtmlDialog.__init__(self, parent, "Aptus", HELP_PAGES, subs=TERMS, size=(650,530))
