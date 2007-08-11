#!/usr/bin/env python
# Started from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

import wx
import numpy
import re, sys, time

if 0: 
    def is_mandelbrot(xi, yi):
        p = complex(XO+xi*XD, Y0+yi*YD)
        i = 0
        z = 0+0j
        while abs(z) < 2:
            if i >= MAXITER:
                return None
            z = z*z+p
            i += 1
        return i

    MAXITER = 999
    X0, Y0 = 0, 0
    XD, YD = 0, 0
    
    def set_params(x0, y0, xd, yd, maxiter):
        X0, Y0 = x0, y0
        XD, YD = xd, yd
        MAXITER = maxiter

elif 0:
    def is_mandelbrot(x, y):
        return 4
else:
    import mandext
    is_mandelbrot = mandext.mandelbrot
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
 
    def from_screen(self, x, y):
        return self.x0+self.rx*x, self.y0-self.ry*y
 
    def zoom_in(self, x, y):
        zx, zy = self.w/4, self.h/4
        return x-zx*self.rx, y+zy*self.ry, x+zx*self.rx, y-zy*self.ry, self.w, self.h, self.maxiter
 
    def zoom_out(self, x, y):
        zx, zy = self.w, self.h
        return x-zx*self.rx, y+zy*self.ry, x+zx*self.rx, y-zy*self.ry, self.w, self.h, self.maxiter
        
    def compute(self, palette):
        print "x, y %r step %r" % ((self.x0, self.y0), (self.rx, self.ry))
        
        set_params(self.x0, self.y0, self.rx, self.ry, self.maxiter)
        counts = numpy.zeros((self.h, self.w), dtype=numpy.uint16)
        for xi in range(self.w):
            for yi in range(self.h):
                c = is_mandelbrot(xi, -yi)
                counts[yi,xi] = c
        palarray = numpy.array(palette, dtype=numpy.uint8)
        pix = palarray[counts % len(palette)]
        return pix
    
class wxMandelbrotSetViewer(wx.Frame):
    def __init__(self, x0, y0, x1, y1, w, h, maxiter):
        super(wxMandelbrotSetViewer, self).__init__(None, -1, 'Mandelbrot Set')
        self.dc = None
 
        self.SetSize((w, h))
        self.bitmap = wx.EmptyBitmap(w, h)
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_LEFT_UP, self.on_zoom_in)
        self.panel.Bind(wx.EVT_RIGHT_UP, self.on_zoom_out)
 
        self.w, self.h = w, h
        self.m = MandelbrotSet(x0, y0, x1, y1, w, h, maxiter)
 
    def on_zoom_in(self, event):
        x,y = self.m.from_screen(event.GetX(), event.GetY())
        self.m = MandelbrotSet(*self.m.zoom_in(x, y))
        self.dc = None
        self.Refresh()
 
    def on_zoom_out(self, event):
        x,y = self.m.from_screen(event.GetX(), event.GetY())
        self.m = MandelbrotSet(*self.m.zoom_out(x, y))
        self.dc = None
        self.Refresh()
 
    def on_paint(self, event):
        if not self.dc:
            self.dc = self.draw()
        dc = wx.PaintDC(self.panel)
        dc.Blit(0, 0, self.w, self.h, self.dc, 0, 0)
 
    def palette(self, c):
        if c is None:
            return (0,0,0)
        return the_palette[c % len(the_palette)]
 
    def draw(self):
        wx.BeginBusyCursor()
        start = time.clock()
        img = wx.EmptyImage(self.w, self.h)
        pix = self.m.compute(the_palette)
        img.SetData(pix.tostring())
        dc = wx.MemoryDC()
        dc.SelectObject(self.bitmap)
        dc.DrawBitmap(img.ConvertToBitmap(), 0, 0, False)
        print "Computation: %.2f sec" % (time.clock() - start)
        wx.EndBusyCursor()
        return dc

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

    x0, y0, x1, y1 = -2.0, 1.5, 1.0, -1.5
    w, h = 600, 600
    maxiter = 999
    
    if len(sys.argv) > 1:
        xaos = XaosState()
        xaos.read(sys.argv[1])
        x0 = xaos.center[0] - xaos.diam[0]/2
        x1 = xaos.center[0] + xaos.diam[0]/2
        y0 = xaos.center[1] - xaos.diam[1]/2
        y1 = xaos.center[1] + xaos.diam[1]/2
        maxiter = xaos.maxiter
        
    f = wxMandelbrotSetViewer(x0, y0, x1, y1, w, h, maxiter)
    f.Show()
    app.MainLoop()
