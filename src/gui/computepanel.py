from PIL import Image
import wx
import numpy

from aptus import settings
from aptus.compute import AptusCompute
from aptus.gui.ids import *
from aptus.options import AptusState
from aptus.palettes import all_palettes
from aptus.progress import NullProgressReporter


class ComputePanel(wx.Panel):
    """ A panel capable of drawing a Mandelbrot.
    """
    def __init__(self, parent, size=wx.DefaultSize):
        wx.Panel.__init__(self, parent, style=wx.NO_BORDER+wx.WANTS_CHARS, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.compute = AptusCompute()
        self.compute.quiet = True     # default to quiet.

        # AptusCompute default values
        self.compute.palette = all_palettes[0]

        # Bind events
        self.Bind(wx.EVT_WINDOW_CREATE, self.on_window_create)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)

    def set_geometry(self, center=None, diam=None, corners=None):
        """ Change the panel to display a new place in the Set.
            `center` is the ri coords of the new center, `diam` is the r and i
            size of the view, `corners` is a 4-tuple (ulr, uli, lrr, lri) of the
            four corners of the view.  Only specify a subset of these.
        """
        compute = self.compute
        if corners:
            ulr, uli, lrr, lri = corners
            compute.center = ((ulr+lrr)/2, (uli+lri)/2)
            ulx, uly = compute.pixel_from_coords(ulr, uli)
            lrx, lry = compute.pixel_from_coords(lrr, lri)
            compute.diam = (abs(compute.pixsize*(lrx-ulx)), abs(compute.pixsize*(lry-uly)))
        if center:
            compute.center = center
        if diam:
            compute.diam = diam

        self.geometry_changed()

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

    def coloring_changed(self):
        self.bitmap = None
        self.Refresh()
        self.fire_event(AptusColoringChangedEvent)

    def computation_changed(self):
        self.set_view()
        self.fire_event(AptusComputationChangedEvent)

    def geometry_changed(self):
        self.set_view()
        self.fire_event(AptusGeometryChangedEvent)

    # Event handlers

    def on_window_create(self, event):
        self.on_idle(event)

    def on_size(self, event_unused):
        self.check_size = True

    def on_idle(self, event_unused):
        if self.check_size and self.GetClientSize() != self.compute.size:
            if self.GetClientSize() != (0,0):
                self.geometry_changed()

    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()

        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0, False)

    # Output methods

    def make_progress_reporter(self):
        """ Create a progress reporter for use when this panel computes.
        """
        return NullProgressReporter()

    def bitmap_from_compute(self):
        print("bitmap_from_compute")
        pix = self.compute.color_mandel()
        w, h = pix.shape[1], pix.shape[0]
        sq = 10
        c = numpy.fromfunction(lambda x,y: ((x//sq) + (y//sq)) % 2, (w,h))
        chex = numpy.empty((w,h,3), dtype=numpy.uint8)
        chex[c == 0] = (0xAA, 0x00, 0x00)
        chex[c == 1] = (0x99, 0x99, 0x00)
        bitmap = wx.Bitmap.FromBuffer(pix.shape[1], pix.shape[0], chex)

        print("bitmap_from_compute done")
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
        #print("Parent is active: %r" % self.GetParent().IsActive())
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
        self.check_size = False
        self.Refresh()
