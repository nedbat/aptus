#!/usr/bin/env python
# from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

import wx
 
class MandelbrotSet:
    def __init__(self,x0,y0,x1,y1,w,h,limit=2,maxiter=30):
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
 
    def is_mandelbrot(self,x,y):
        p = complex(x,y)
        i = 0
        z = 0+0j
        while abs(z) < self.limit and i < self.maxiter:
            z = z*z+p
            i += 1
        return i
 
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
        c = c*255.0/self.m.maxiter
        return map(int,[c,(c+64)%256,(c+32)%256])
 
    def draw(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.bitmap)
        def callback(x,y,c):
            r,g,b = self.palette(c)
            dc.SetPen(wx.Pen(wx.Colour(r,g,b),1))
            dc.DrawPoint(x,y)
        self.m.compute(callback)
        return dc
 
if __name__ == '__main__':
    app = wx.PySimpleApp()
    f = wxMandelbrotSetViewer(-2.0,1.0,1.0,-1.0,200,200)
    f.Show()
    app.MainLoop()
