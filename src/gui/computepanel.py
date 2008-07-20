from aptus.compute import AptusCompute
from aptus.importer import importer
from aptus.options import AptusState
from aptus.palettes import all_palettes
from aptus.progress import ConsoleProgressReporter, IntervalProgressReporter, AggregateProgressReporter

from aptus.gui.ids import *

wx = importer('wx')
Image = importer('Image')

class GuiProgressReporter:
    """ A progress reporter tied into the GUI.
    """
    def __init__(self, aptview):
        self.aptview = aptview
        
    def begin(self):
        wx.BeginBusyCursor()
        
    def progress(self, frac_done_unused, info_unused=''):
        self.aptview.draw_progress()
        # Yield so that repaints of the screen will happen.
        wx.SafeYield()

    def end(self):
        wx.EndBusyCursor()


class ComputePanel(wx.Panel):
    """ A panel capable of drawing a Mandelbrot.
    """
    def __init__(self, parent, size=wx.DefaultSize):
        wx.Panel.__init__(self, parent, style=wx.NO_BORDER+wx.WANTS_CHARS, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.m = AptusCompute()
        
        # AptusCompute default values        
        self.m.palette = all_palettes[0]

        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)

    def set_geometry(self, center=None, diam=None, corners=None):
        """ Change the panel to display a new place in the Set.
            `center` is the ri coords of the new center, `diam` is the r and i
            size of the view, `corners` is a 4-tuple (ulr, uli, lrr, lri) of the
            four corners of the view.  Only specify a subset of these.
        """
        if corners:
            ulr, uli, lrr, lri = corners
            self.m.center = ((ulr+lrr)/2, (uli+lri)/2)
            ulx, uly = self.m.pixel_from_coords(ulr, uli)
            lrx, lry = self.m.pixel_from_coords(lrr, lri)
            self.m.diam = (abs(self.m.pixsize*(lrx-ulx)), abs(self.m.pixsize*(lry-uly)))
        if center:
            self.m.center = center
        if diam:
            self.m.diam = diam
            
        self.geometry_changed()

    # GUI helpers
    
    def fire_command(self, cmdid, data=None):
        # I'm not entirely sure about why this is the right event type to use,
        # but it works...
        evt = wx.CommandEvent(wx.wxEVT_COMMAND_TOOL_CLICKED)
        evt.SetId(cmdid)
        evt.SetClientData(data)
        wx.PostEvent(self, evt)
    
    def fire_event(self, evclass):
        self.GetEventHandler().ProcessEvent(evclass())
        
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
    
    def on_size(self, event_unused):
        self.check_size = True
        
    def on_idle(self, event_unused):
        if self.check_size and self.GetClientSize() != self.m.size:
            if self.GetClientSize() != (0,0):
                self.geometry_changed()

    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()

        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0, False)
        self.on_paint_extras(dc)
        
    def on_paint_extras(self, dc):
        """ An overridable method so that derived classes can paint extra stuff
            on top of the fractal.
        """
        pass

    # Information methods
    
    def get_stats(self):
        """ Return a dictionary full of statistics about the latest computation.
        """
        return self.m.eng.get_stats()

    def get_point_info(self, pt):
        """ Return a dictionary of information about the specified point (in client pixels).
            If the point is outside the window, None is returned.
        """
        client_rect = self.GetRect()
        if not client_rect.Contains(pt):
            return None
        
        x, y = pt
        r, i = self.m.coords_from_pixel(x, y)

        if self.m.pix is not None:
            rgb = self.m.pix[y, x]
            color = "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])
        else:
            color = None
        
        count = self.m.counts[y, x]
        if self.m.eng.cont_levels != 1:
            count /= self.m.eng.cont_levels
        
        point_info = {
            'x': x, 'y': y,
            'r': r, 'i': i,
            'count': count,
            'color': color,
            }
        
        return point_info
    
    # Output methods
    
    def make_progress_reporter(self):
        # Construct a progress reporter that suits us.  Write to the console,
        # and keep the GUI updated, but only once a second.
        prorep = AggregateProgressReporter()
        prorep.add(ConsoleProgressReporter())
        prorep.add(GuiProgressReporter(self))
        return IntervalProgressReporter(1, prorep)
    
    def bitmap_from_compute(self):
        pix = self.m.color_mandel()
        bitmap = wx.BitmapFromBuffer(pix.shape[1], pix.shape[0], pix)
        return bitmap

    def draw_bitmap(self):
        """ Return a bitmap with the image to display in the window.
        """
        self.m.progress = self.make_progress_reporter()
        self.m.compute_pixels()
        wx.CallAfter(self.fire_event, AptusRecomputedEvent)
        self.Refresh()
        return self.bitmap_from_compute()

    def draw_progress(self):
        self.bitmap = self.bitmap_from_compute()
        self.Refresh()
        self.Update()
        
    def set_view(self):
        self.bitmap = None
        self.m.size = self.GetClientSize()
        self.m.create_mandel()
        self.check_size = False
        self.Refresh()

    # Output-writing methods
    
    def write_png(self, pth):
        """ Write the current image as a PNG to the path `pth`.
        """
        image = wx.ImageFromBitmap(self.bitmap)
        im = Image.new('RGB', (image.GetWidth(), image.GetHeight()))
        im.fromstring(image.GetData())
        self.m.write_image(im, pth)

    def write_aptus(self, pth):
        """ Write the current Aptus state of the panel to the path `pth`.
        """
        aptst = AptusState(self.m)
        aptst.write(pth)
