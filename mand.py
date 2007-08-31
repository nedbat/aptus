#!/usr/bin/env python
# Started from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

import wx
import numpy
import Image
import os, re, sys, time, traceback, zlib

if 0: 
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

else:
    import mandext
    mandelbrot_count = mandext.mandelbrot_count
    set_params = mandext.set_params
        

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
    
class MandelbrotSet:
    def __init__(self, x0, y0, x1, y1, w, h, maxiter=999):
        self.x0, self.y0 = x0, y0
        self.rx, self.ry = (x1-x0)/w, (y0-y1)/h
        self.w, self.h = w, h
 
        self.maxiter = maxiter
 
    def from_pixel(self, x, y):
        return self.x0+self.rx*x, self.y0-self.ry*y
 
    def compute(self):
        print "x, y %r step %r" % ((self.x0, self.y0), (self.rx, self.ry))
        
        set_params(self.x0, self.y0, self.rx, self.ry, self.maxiter)
        counts = numpy.zeros((self.h, self.w), dtype=numpy.uint32)
        for xi in xrange(self.w):
            for yi in xrange(self.h):
                c = mandelbrot_count(xi, -yi)
                counts[yi,xi] = c
        return counts
    
    def compute_trace(self):
        from boundary import trace_boundary
        set_params(self.x0, self.y0, self.rx, self.ry, self.maxiter)
        return trace_boundary(mandelbrot_count, self.w, self.h)
    
    def compute_pixels(self, compute_fn, palette, keep=False):
        start = time.clock()
        self.counts = compute_fn()
        print "Computation: %.2f sec" % (time.clock() - start)
        palarray = numpy.array(palette, dtype=numpy.uint8)
        pix = palarray[self.counts % len(palette)]
        return pix
        
class wxMandelbrotSetViewer(wx.Frame):
    def __init__(self, xcenter, ycenter, xdiam, ydiam, w, h, maxiter):
        super(wxMandelbrotSetViewer, self).__init__(None, -1, 'Mandelbrot Set')
 
        self.SetSize((w, h))
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
        
    def on_paint(self, event):
        if not self.dc:
            self.dc = self.draw()
        dc = wx.PaintDC(self.panel)
        dc.Blit(0, 0, self.cw, self.ch, self.dc, 0, 0)
 
    def draw(self):
        wx.BeginBusyCursor()
        pix = self.m.compute_pixels(self.m.compute, the_palette, keep=True)
        #Image.fromarray(pix).save('one.png')
        if 0:
            pixt = self.m.compute_pixels(self.m.compute_trace, the_palette)
            #Image.fromarray(pixt).save('two.png')
            wrong_count = numpy.sum(numpy.logical_not(numpy.equal(pixt, pix)))
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
                m = self.choose_mandel(w*2, h*2)
                pix = m.compute_pixels(m.compute, the_palette)
                im = Image.fromarray(pix)
                im = im.resize((w,h), Image.ANTIALIAS)
                im.save(dlg.GetPath())

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
        
class XaosState:
    """ The state of a Xaos rendering.
    """
    def __init__(self):
        self.maxiter = 170
        self.center = -0.75, 0.0
        self.diam = 2.55, 2.55
        
    def read(self, f):
        if isinstance(f, str):
            f = open(f)
        for l in f:
            if l.startswith('('):
                argv = l[1:-2].split()
                if hasattr(self, 'handle_'+argv[0]):
                    getattr(self, 'handle_'+argv[0])(*argv)
                    
    def handle_maxiter(self, op, maxiter):
        self.maxiter = int(maxiter)
        
    def handle_view(self, op, cx, cy, rx, ry):
        self.center = self.read_float(cx), self.read_float(cy)
        self.diam = self.read_float(rx), self.read_float(ry)
        
    def read_float(self, fstr):
        fstr = fstr.lower()
        if fstr.endswith('e'):
            fstr += '0'
        return float(fstr)

if __name__ == '__main__':
    app = wx.PySimpleApp()

    xcenter, ycenter = -0.5, 0.0
    xdiam, ydiam = 3.0, 3.0
    w, h = 600, 600
    maxiter = 999
    
    if len(sys.argv) > 1:
        xaos = XaosState()
        xaos.read(sys.argv[1])
        xcenter, ycenter = xaos.center
        xdiam, ydiam = xaos.diam
        maxiter = xaos.maxiter
        
    f = wxMandelbrotSetViewer(xcenter, ycenter, xdiam, ydiam, w, h, maxiter)
    f.Show()
    app.MainLoop()
