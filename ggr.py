""" Read Gimp .ggr gradient files.
"""

import colorsys

class GimpGradient:
    
    class segment:
        pass
    
    def read(self, f):
        if isinstance(f, basestring):
            f = file(f)
        if f.readline().strip() != "GIMP Gradient":
            raise Exception("Not a GIMP gradient file")
        line = f.readline().strip()
        if not line.startswith("Name: "):
            raise Exception("Not a GIMP gradient file")
        self.name = line.split(": ", 1)[1]
        nsegs = int(f.readline().strip())
        self.segs = []
        for i in range(nsegs):
            line = f.readline().strip()
            seg = self.segment()
            (seg.l, seg.m, seg.r,
                seg.rl, seg.gl, seg.bl, _,
                seg.rr, seg.gr, seg.br, _,
                seg.fn, seg.space) = map(float, line.split())
            self.segs.append(seg)
            
    def color(self, x):
        """ Get the color for the point x in the range [0..1).
        """
        # Find the segment.
        for seg in self.segs:
            if seg.l <= x <= seg.r:
                break
        else:
            # No segment applies! Return black I guess.
            return (0,0,0)

        # Normalize the geometry of the segment.
        #mid = (seg.m - seg.l)/(seg.r - seg.l)
        #pos = (x - seg.l)/(seg.r - seg.l)
        
        # Assume linear for now.
        if x <= seg.m:
            f = (x - seg.l)/(seg.m - seg.l)
            upper = False
        else:
            f = (x - seg.m)/(seg.r - seg.m)
            upper = True

        cl = (seg.rl, seg.gl, seg.bl)
        cr = (seg.rr, seg.gr, seg.br)
        
        # Interpolate the colors
        if seg.space == 0:
            cm = ((seg.rl+seg.rr)/2, (seg.gl+seg.gr)/2, (seg.bl+seg.br)/2)
            if upper:
                cl = cm
            else:
                cr = cm
            c = (
                cl[0] + (cr[0]-cl[0]) * f,
                cl[1] + (cr[1]-cl[1]) * f,
                cl[2] + (cr[2]-cl[2]) * f
                )
        elif seg.space in (1,2):
            hl = colorsys.rgb_to_hsv(*cl)
            hr = colorsys.rgb_to_hsv(*cr)

            huel, huer = hl[0], hr[0]
            if seg.space == 1 and huer < huel:
                huer += 1
            elif seg.space == 2 and huer > huel:
                huer -= 1

            huem = (huel+huer)/2
            hm = (huem, (hl[1]+hr[1])/2, (hl[2]+hr[2])/2)
            if upper:
                hl = hm
            else:
                hr = hm

            huel, huer = hl[0], hr[0]
            if seg.space == 1 and huer < huel:
                huer += 1
            elif seg.space == 2 and huer > huel:
                huer -= 1

            hf = (
                (huel + (huer-huel) * f) % 1.0,
                hl[1] + (hr[1]-hl[1]) * f,
                hl[2] + (hr[2]-hl[2]) * f
                )
            c = colorsys.hsv_to_rgb(*hf)
        return c
    
if __name__ == '__main__':
    import sys, wx
    ggr = GimpGradient()
    ggr.read(sys.argv[1])

    class GgrView(wx.Frame):
        def __init__(self, ggr):
            super(GgrView, self).__init__(None, -1, 'Ggr: %s' % ggr.name)
            self.ggr = ggr
            self.SetSize((300, 50))
            self.panel = wx.Panel(self)
            self.panel.Bind(wx.EVT_PAINT, self.on_paint)
            self.panel.Bind(wx.EVT_SIZE, self.on_size)

        def on_paint(self, event):
            dc = wx.PaintDC(self.panel)
            cw, ch = self.GetClientSize()
            for x in range(0, cw):
                c = map(lambda x:int(255*x), ggr.color(float(x)/cw))
                dc.SetPen(wx.Pen(wx.Colour(*c),1))
                dc.DrawLine(x, 0, x, ch)
        
        def on_size(self, event):
            self.Refresh()
            
    app = wx.PySimpleApp()
    f = GgrView(ggr)
    f.Show()
    app.MainLoop()
