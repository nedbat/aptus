from PIL import Image
import wx
import numpy

from aptus.compute import AptusCompute
from aptus.gui.ids import *


class ComputePanel(wx.Panel):
    """ A panel capable of drawing a Mandelbrot.
    """
    def __init__(self, parent, size=wx.DefaultSize):
        wx.Panel.__init__(self, parent, style=wx.NO_BORDER+wx.WANTS_CHARS, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.compute = AptusCompute()

        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)

    # GUI helpers

    def fire_command(self, cmdid, data=None):
        # I'm not entirely sure about why this is the right event type to use,
        # but it works...
        evt = wx.CommandEvent(wx.wxEVT_COMMAND_TOOL_CLICKED)
        evt.SetId(cmdid)
        evt.SetClientData(data)
        wx.PostEvent(self, evt)

    def fire_event(self, evclass, **kwargs):
        evt = evclass(**kwargs)
        self.GetEventHandler().ProcessEvent(evt)

    def message(self, msg):
        top = self.GetTopLevelParent()
        top.message(msg)

    def computation_changed(self):
        self.set_view()
        self.fire_event(AptusComputationChangedEvent)

    def geometry_changed(self):
        self.set_view()
        self.fire_event(AptusGeometryChangedEvent)

    # Event handlers

    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()

        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0, False)

    # Output methods

    def bitmap_from_compute(self):
        #pix = self.compute.color_mandel()
        w, h = 600, 600 #pix.shape[1], pix.shape[0]
        sq = 10
        c = numpy.fromfunction(lambda x,y: ((x//sq) + (y//sq)) % 2, (w,h))
        chex = numpy.empty((w,h,3), dtype=numpy.uint8)
        chex[c == 0] = (0xAA, 0x00, 0x00)
        chex[c == 1] = (0x99, 0x99, 0x00)
        bitmap = wx.Bitmap.FromBuffer(600, 600, chex)
        return bitmap

    def draw_bitmap(self):
        """ Return a bitmap with the image to display in the window.
        """
        wx.BeginBusyCursor()
        self.compute.progress = self.make_progress_reporter()
        self.compute.while_waiting = self.draw_progress
        self.compute.compute_pixels()
        wx.CallAfter(self.fire_event, AptusRecomputedEvent)
        self.Refresh()
        bitmap = self.bitmap_from_compute()
        wx.EndBusyCursor()
        return bitmap

    def draw_progress(self):
        """ Called from the GUI thread periodically during computation.

        Repaints the window.

        """
        print("into draw_progress")
        self.bitmap = self.bitmap_from_compute()
        self.Refresh()
        self.Update()
        wx.CallAfter(self.fire_event, AptusRecomputedEvent)
        wx.SafeYield(onlyIfNeeded=True)
        print("out of draw_progress")

    def set_view(self):
        self.bitmap = None
        self.compute.size = self.GetClientSize()
        self.compute.create_mandel()
        self.Refresh()
