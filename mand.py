#!/usr/bin/env python
# Started from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

from timeutil import duration, future
from options import MandOptions

import wx
import numpy
import os, re, sys, time, traceback, zlib

# We use PIL, and need 1.1.6.
import Image

if not hasattr(Image, 'fromarray'):
    raise Exception("Need PIL 1.1.6 or greater.")

try:
    from mandext import *
except:
    # Pure python (slow!) implementation of mandext interface.
    MAXITER = 999
    X0, Y0 = 0, 0
    XD, YD = 0, 0
    
    def mandelbrot_count(xi, yi):
        p = complex(X0+xi*XD, Y0+yi*YD)
        i = 0
        z = 0+0j
        while abs(z) < 2:
            if i >= MAXITER:
                return 0
            z = z*z+p
            i += 1
        return i

    def set_params(x0, y0, xd, yd, maxiter):
        global X0, Y0, XD, YD, MAXITER
        X0, Y0 = x0, y0
        XD, YD = xd, yd
        MAXITER = maxiter

    def clear_stats():
        pass

    def get_stats():
        return {}

# Colors taken from Xaos, to get the same rendering.
colors = [
    (0, 0, 0),
    (120, 119, 238),
    (24, 7, 25),
    (197, 66, 28),
    (29, 18, 11),
    (135, 46, 71),
    (24, 27, 13),
    (241, 230, 128),
    (17, 31, 24),
    (240, 162, 139),
    (11, 4, 30),
    (106, 87, 189),
    (29, 21, 14),
    (12, 140, 118),
    (10, 6, 29),
    (50, 144, 77),
    (22, 0, 24),
    (148, 188, 243),
    (4, 32, 7),
    (231, 146, 14),
    (10, 13, 20),
    (184, 147, 68),
    (13, 28, 3),
    (169, 248, 152),
    (4, 0, 34),
    (62, 83, 48),
    (7, 21, 22),
    (152, 97, 184),
    (8, 3, 12),
    (247, 92, 235),
    (31, 32, 16)
]

the_palette = [None]*(len(colors)*8)
for i in range(len(the_palette)):
    color_index = i//8
    r0, g0, b0 = colors[color_index]
    r1, g1, b1 = colors[(color_index + 1) % len(colors)]
    step = float(i % 8)/8
    the_palette[i] = (
        int(r0 + (r1 - r0) * step),
        int(g0 + (g1 - g0) * step),
        int(b0 + (b1 - b0) * step),
        )
    
class NullProgressReporter:
    def begin(self):
        pass
    
    def progress(self, frac_done):
        pass
    
    def end(self):
        pass
    
class MandelbrotSet:
    def __init__(self, x0, y0, x1, y1, w, h, maxiter=999):
        self.x0, self.y0 = x0, y0
        self.rx, self.ry = (x1-x0)/w, (y0-y1)/h
        self.w, self.h = w, h
 
        self.maxiter = maxiter
        self.progress = NullProgressReporter()
        
    def from_pixel(self, x, y):
        return self.x0+self.rx*x, self.y0-self.ry*y
 
    def compute(self):
        counts = numpy.zeros((self.h, self.w), dtype=numpy.uint32)
        for yi in xrange(self.h):
            for xi in xrange(self.w):
                c = mandelbrot_count(xi, -yi)
                counts[yi,xi] = c
            self.progress.progress(float(yi+1)/self.h)
        return counts
    
    def compute_trace(self):
        from boundary import trace_boundary
        return trace_boundary(mandelbrot_count, self.w, self.h, self.maxiter, progress_fn=self.progress.progress)
    
    def compute_pixels(self, palette, trace=False, keep=False):
        print "x, y %r step %r, maxiter %r, trace %r" % ((self.x0, self.y0), (self.rx, self.ry), self.maxiter, trace)
        clear_stats()
        set_params(self.x0, self.y0, self.rx, self.ry, self.maxiter)
        self.progress.begin()
        if trace:
            self.counts = self.compute_trace()
        else:
            self.counts = self.compute()
        self.progress.end()
        print get_stats()
        palarray = numpy.array(palette, dtype=numpy.uint8)
        pix = palarray[self.counts % len(palette)]
        return pix
        
class wxMandelbrotSetViewer(wx.Frame):
    def __init__(self, xcenter, ycenter, xdiam, ydiam, w, h, maxiter):
        super(wxMandelbrotSetViewer, self).__init__(None, -1, 'Mandelbrot Set')
 
        chromew, chromeh = 8, 28
        self.SetSize((w+chromew, h+chromeh))
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_LEFT_UP, self.on_zoom_in)
        self.panel.Bind(wx.EVT_RIGHT_UP, self.on_zoom_out)
        self.panel.Bind(wx.EVT_SIZE, self.on_size)
        self.panel.Bind(wx.EVT_IDLE, self.on_idle)
        self.panel.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        
        self.xcenter, self.ycenter = xcenter, ycenter
        self.xdiam, self.ydiam = xdiam, ydiam
        self.maxiter = maxiter
        self.set_view()

    def set_view(self):
        self.cw, self.ch = self.GetClientSize()
        self.bitmap = wx.EmptyBitmap(self.cw, self.ch)
        self.dc = None

        #self.rubberBand = rubberband.RubberBand(drawingSurface=self.panel)
        #self.rubberBand.reset(aspectRatio=1.0*self.cw/self.ch)

        self.m = self.choose_mandel(self.cw, self.ch)
        self.check_size = False
        self.Refresh()

    def choose_mandel(self, w, h):
        scale = max(self.xdiam / w, self.ydiam / h)
        xradius, yradius = scale * w / 2, scale * h / 2
        
        x0, y0 = self.xcenter - xradius, self.ycenter - yradius
        x1, y1 = self.xcenter + xradius, self.ycenter + yradius
        
        return MandelbrotSet(x0, y0, x1, y1, w, h, self.maxiter)
        
    def message(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Mand', wx.OK | wx.ICON_WARNING)
        dlg.ShowModal()
        dlg.Destroy()
        
    def on_zoom_in(self, event):
        self.xcenter, self.ycenter = self.m.from_pixel(event.GetX(), event.GetY())
        self.xdiam /= 2.0
        self.ydiam /= 2.0
        self.set_view()
 
    def on_zoom_out(self, event):
        self.xcenter, self.ycenter = self.m.from_pixel(event.GetX(), event.GetY())
        self.xdiam *= 2.0
        self.ydiam *= 2.0
        self.set_view()
 
    def on_size(self, event):
        self.check_size = True
        
    def on_idle(self, event):
        if self.check_size and self.GetClientSize() != (self.cw, self.ch):
            self.set_view()

    def on_key_down(self, event):
        shift = event.ShiftDown()
        if event.KeyCode == ord('S'):
            if shift:
                self.cmd_save_big()
            else:
                self.cmd_save()
        elif event.KeyCode == ord('I'):
            self.cmd_set_maxiter()
        elif event.KeyCode == ord('R'):
            self.cmd_redraw()
            
    def on_paint(self, event):
        if not self.dc:
            self.dc = self.draw()
        dc = wx.PaintDC(self.panel)
        dc.Blit(0, 0, self.cw, self.ch, self.dc, 0, 0)
 
    def draw(self):
        wx.BeginBusyCursor()
        self.m.progress = ConsoleProgressReporter()
        pix = self.m.compute_pixels(the_palette, trace=self.trace, keep=True)
        pix2 = None
        if 0:
            pix2 = self.m.compute_pixels(the_palette, trace=not self.trace)
        if 0:
            set_check_cycles(0)
            pix2 = self.m.compute_pixels(the_palette, trace=self.trace)
            set_check_cycles(1)
        if pix2 is not None:
            Image.fromarray(pix).save('one.png')
            Image.fromarray(pix2).save('two.png')
            wrong_count = numpy.sum(numpy.logical_not(numpy.equal(pix, pix2)))
            print wrong_count
        img = wx.EmptyImage(self.cw, self.ch)
        img.SetData(pix.tostring())
        dc = wx.MemoryDC()
        dc.SelectObject(self.bitmap)
        dc.DrawBitmap(img.ConvertToBitmap(), 0, 0, False)
        wx.EndBusyCursor()
        return dc

    def cmd_save(self):
        wildcard = (
            "PNG image (*.png)|*.png|"     
            "Mand state (*.mand)|*.mand|"
            "All files (*.*)|*.*"
            )

        dlg = wx.FileDialog(
            self, message="Save", defaultDir=os.getcwd(), 
            defaultFile="", style=wx.SAVE, wildcard=wildcard, 
            )

        if dlg.ShowModal() == wx.ID_OK:
            ext = dlg.GetFilename().split('.')[-1].lower()
            if ext == 'png':
                image = wx.ImageFromBitmap(self.bitmap)
                im = Image.new('RGB', (image.GetWidth(), image.GetHeight()))
                im.fromstring(image.GetData())
                fout = open(dlg.GetPath(), 'wb')
                im.save(fout, 'PNG')
                fout.close()
            elif ext == 'mand':
                ms = MandState()
                ms.w = self.cw
                ms.h = self.ch
                ms.counts = self.counts.tostring()
                ms.write(dlg.GetPath())
            else:
                self.message("Don't understand how to write file '%s'" % dlg.GetFilename())
                
    def cmd_save_big(self):
        wildcard = (
            "PNG image (*.png)|*.png|"     
            "All files (*.*)|*.*"
            )

        dlg = wx.FileDialog(
            self, message="Save big image", defaultDir=os.getcwd(), 
            defaultFile="", style=wx.SAVE, wildcard=wildcard, 
            )

        if dlg.ShowModal() == wx.ID_OK:
            ext = dlg.GetFilename().split('.')[-1].lower()
            if ext == 'png':
                w, h = 1680, 1050
                m = self.choose_mandel(w*3, h*3)
                m.progress = ConsoleProgressReporter()
                pix = m.compute_pixels(the_palette, trace=self.trace)
                im = Image.fromarray(pix)
                im = im.resize((w,h), Image.ANTIALIAS)
                im.save(dlg.GetPath())

    def cmd_set_maxiter(self):
        dlg = wx.TextEntryDialog(
                self, 'Maximum iteration count:',
                'Maxiter', str(self.maxiter)
                )

        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.maxiter = int(dlg.GetValue())
            except Exception, e:
                self.message("Couldn't set maxiter: %s" % e)

        dlg.Destroy()

    def cmd_redraw(self):
        self.set_view()
        
class ConsoleProgressReporter:
    def begin(self):
        self.start = time.time()
        self.latest = self.start

    def progress(self, frac_done, info=''):
        now = time.time()
        if now - self.latest > 10:
            so_far = int(now - self.start)
            to_go = int(so_far / frac_done * (1-frac_done))
            if info:
                info = '  ' + info
            print "%.2f%%: %10s done, %10s to go, eta %10s%s" % (
                frac_done*100, duration(so_far), duration(to_go), future(to_go), info
                )
            self.latest = now
    
    def end(self):
        total = time.time() - self.start
        print "Total: %s (%.2fs)" % (duration(total), total)
        
class MandState:
    def write(self, f):
        if isinstance(f, basestring):
            f = open(f, 'wb')
        print >>f, '{'
        self._write_item(f, 'what_is_this', 'A Mand state file, version 1')
        self._write_item(f, 'w', self.w)
        self._write_item(f, 'h', self.h)
        self._write_item(f, 'counts', zlib.compress(self.counts).encode('base64').strip())
        print >>f, '}'
    
    def read(self, f):
        if isinstance(f, basestring):
            f = open(f, 'rb')
        # This is dangerous!
        d = eval(f.read())
        self.w = d['w']
        self.h = d['h']
        self.counts = zlib.decompress(d['counts'].decode('base64'))
        
    def _write_item(self, f, k, v):
        print >> f, ' "%s": %r,' % (k, v)
        
if __name__ == '__main__':
    opts = MandOptions()
    opts.read_args(sys.argv[1:])
    
    app = wx.PySimpleApp()
    f = wxMandelbrotSetViewer(
        opts.center[0], opts.center[1],
        opts.diam[0], opts.diam[1],
        opts.size[0], opts.size[1],
        opts.maxiter
        )
    f.trace = opts.trace
    f.Show()
    app.MainLoop()
