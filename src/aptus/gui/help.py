""" Help dialog for Aptus.
"""

import webbrowser
import sys

import numpy
import wx
import wx.html2
import wx.lib.layoutf
from PIL import Image

from aptus import data_file, __version__
from aptus.options import AptusOptions


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
        self.html = wx.html2.WebView.New(self)
        self.html.Bind(wx.html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating)
        ok = wx.Button(self, wx.ID_OK, "OK")
        ok.SetDefault()

        lc = wx.lib.layoutf.Layoutf('t=t#1;b=t5#2;l=l#1;r=r#1', (self,ok))
        self.html.SetConstraints(lc)
        self.set_page('interactive')

        lc = wx.lib.layoutf.Layoutf('b=b5#1;r=r5#1;w!80;h*', (self,))
        ok.SetConstraints(lc)

        self.SetAutoLayout(1)
        self.Layout()

    def on_navigating(self, event):
        url = event.GetURL()
        if url == "":
            event.Veto()
        elif url.startswith(("http:", "https:")):
            webbrowser.open(url)
            event.Veto()
        elif url.startswith('internal:'):
            self.set_page(url.split(':')[1])

    def set_page(self, pagename):
        html = self.pages['head'] + self.pages[pagename]
        html = html % self.subs
        self.html.SetPage(html, "")


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
        <style>
        kbd {
            display: inline-block;
            background: #f0f0f0;
            border: 2px solid #888;
            border-color: #888 #333 #333 #888;
            border-radius: .25em;
            padding: .1em .25em;
            margin: .1em;
        }
        </style>

        <table width='100%%'>
        <tr>
            <td width='50' valign='top'><img src='%(iconsrc)s'/></td>
            <td valign='top'>
                <b>Aptus %(version)s</b>, Mandelbrot set explorer.<br>
                Copyright 2007-2020, Ned Batchelder.<br>
                <a href='https://nedbatchelder.com/code/aptus'>http://nedbatchelder.com/code/aptus</a>
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
        <kbd>a</kbd>: set the angle of rotation.<br>
        <kbd>c</kbd>: toggle continuous coloring.<br>
        <kbd>f</kbd>: toggle full-screen display.<br>
        <kbd>h</kbd> or <kbd>?</kbd>: show this help.<br>
        <kbd>i</kbd>: set the limit on iterations.<br>
        <kbd>j</kbd>: jump among a few pre-determined locations.<br>
        <kbd>n</kbd>: create a new window.<br>
        <kbd>o</kbd>: open a saved settings or image file.<br>
        <kbd>r</kbd>: redraw the current image.<br>
        <kbd>s</kbd>: save the current image or settings.<br>
        <kbd>w</kbd>: set the window size.<br>
        <kbd>&lt;</kbd> or <kbd>&gt;</kbd>: switch to the next palette.<br>
        <kbd>,</kbd> or <kbd>.</kbd>: cycle the current palette one color.<br>
        <kbd>;</kbd> or <kbd>'</kbd>: stretch the palette colors (+%(ctrl)s: just a little), if continuous.<br>
        <kbd>[</kbd> or <kbd>]</kbd>: adjust the hue of the palette (+%(ctrl)s: just a little).<br>
        <kbd>{</kbd> or <kbd>}</kbd>: adjust the saturation of the palette (+%(ctrl)s: just a little).<br>
        <kbd>0</kbd> (zero): reset all palette adjustments.<br>
        <kbd>space</kbd>: drag mode: click to drag the image to a new position.<br>
        <kbd>shift</kbd>: indicate a point of interest for Julia set and point info.<br>
        <b>left-click</b>: zoom in (+%(ctrl)s: just a little).<br>
        <b>right-click</b>: zoom out (+%(ctrl)s: just a little).<br>
        <b>left-drag</b>: select a new rectangle to display.<br>
        <b>middle-drag</b>: drag the image to a new position.<br>
        </blockquote>

        <p><b>Tool windows: press a key to toggle on and off:</b></p>

        <blockquote>
        <kbd>J</kbd> (shift-j): Show a Julia set for the current (shift-hovered) point.<br>
        <kbd>l</kbd> (ell): Show zoom snapshots indicating the current position.<br>
        <kbd>p</kbd>: Show a list of palettes that can be applied to the current view.<br>
        <kbd>q</kbd>: Show point info for the current (shift-hovered) point.<br>
        <kbd>v</kbd>: Show statistics for the latest calculation.
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
