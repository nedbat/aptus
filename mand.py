#!/usr/bin/env python
# Copied from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

import wx
import time

import mandext

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
    def __init__(self,x0,y0,x1,y1,w,h,limit=2,maxiter=999):
        self.x0,self.y0 = x0,y0
        self.rx,self.ry = (x1-x0)/w,(y0-y1)/h
        self.w,self.h = w,h
 
        self.limit = limit
        self.maxiter = maxiter
 
    def from_screen(self,x,y):
        return self.x0+self.rx*x,self.y0-self.ry*y
 
    def zoom_in(self,x,y):
        zx,zy = self.w/4,self.h/4
        return x-zx*self.rx,y+zy*self.ry,x+zx*self.rx,y-zy*self.ry,self.w,self.h
 
    def zoom_out(self,x,y):
        zx,zy = self.w,self.h
        return x-zx*self.rx,y+zy*self.ry,x+zx*self.rx,y-zy*self.ry,self.w,self.h

    if 0: 
        def is_mandelbrot(self,x,y):
            p = complex(x,y)
            i = 0
            z = 0+0j
            while abs(z) < self.limit:
                if i >= self.maxiter:
                    return None
                z = z*z+p
                i += 1
            return i
    elif 0:
        def is_mandelbrot(self, x, y):
            return 4
    else:
        def is_mandelbrot(self, x, y):
            return mandext.mandelbrot(x, y, self.maxiter)
        
    def compute(self,callback):
        x = self.x0
        for xi in range(self.w):
            y = self.y0
            for yi in range(self.h):
                c = self.is_mandelbrot(x,y)
                callback(xi,yi,c)
                y -= self.ry
            x += self.rx
 
class wxMandelbrotSetViewer(wx.Frame):
    def __init__(self,x0,y0,x1,y1,w,h):
        super(wxMandelbrotSetViewer,self).__init__(None,-1,'Mandelbrot Set')
        self.dc = None
 
        self.SetSize((w,h))
        self.bitmap = wx.EmptyBitmap(w,h)
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT,self.on_paint)
        self.panel.Bind(wx.EVT_LEFT_UP,self.on_zoom_in)
        self.panel.Bind(wx.EVT_RIGHT_UP,self.on_zoom_out)
 
        self.w,self.h = w,h
        self.m = MandelbrotSet(x0,y0,x1,y1,w,h)
 
    def on_zoom_in(self,event):
        x,y = self.m.from_screen(event.GetX(),event.GetY())
        self.m = MandelbrotSet(*self.m.zoom_in(x,y))
        self.dc = None
        self.Refresh()
 
    def on_zoom_out(self,event):
        x,y = self.m.from_screen(event.GetX(),event.GetY())
        self.m = MandelbrotSet(*self.m.zoom_out(x,y))
        self.dc = None
        self.Refresh()
 
    def on_paint(self,event):
        if not self.dc:
            self.dc = self.draw()
        dc = wx.PaintDC(self.panel)
        dc.Blit(0,0,self.w,self.h,self.dc,0,0)
 
    def palette(self,c):
        if c is None:
            return (0,0,0)
        return the_palette[c % len(the_palette)]
 
    def old_draw(self):
        wx.BeginBusyCursor()
        start = time.clock()
        dc = wx.MemoryDC()
        dc.SelectObject(self.bitmap)
        self.cur_c = -1
        def callback(x,y,c):
            if c != self.cur_c:
                rgb = self.palette(c)
                self.cur_c = c
                dc.SetPen(wx.Pen(wx.Colour(*rgb),1))
            dc.DrawPoint(x,y)
        self.m.compute(callback)
        print "Computation: %.2f sec" % (time.clock() - start)
        wx.EndBusyCursor()
        return dc
 
    def draw(self):
        wx.BeginBusyCursor()
        start = time.clock()
        img = wx.EmptyImage(self.w, self.h)
        self.cur_c = -1
        self.cur_rgb = None
        def callback(x,y,c):
            if c != self.cur_c:
                self.cur_rgb = self.palette(c)
                self.cur_c = c
            img.SetRGB(x,y,self.cur_rgb[0],self.cur_rgb[1],self.cur_rgb[2])
        self.m.compute(callback)

        dc = wx.MemoryDC()
        dc.SelectObject(self.bitmap)
        dc.DrawBitmap(img.ConvertToBitmap(), 0, 0, False)
        print "Computation: %.2f sec" % (time.clock() - start)
        wx.EndBusyCursor()
        return dc
 
if __name__ == '__main__':
    app = wx.PySimpleApp()
    f = wxMandelbrotSetViewer(-2.0,1.5,1.0,-1.5,600,600)
    f.Show()
    app.MainLoop()
