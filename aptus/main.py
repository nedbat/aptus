# Started from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

from aptus.timeutil import duration, future
from aptus.options import AptusOptions
from aptus.palettes import all_palettes

# Import third-party packages.

try:
    import wx
except:
    raise Exception("Need wxPython, from http://www.wxpython.org/")

try:
    import numpy
except:
    raise Exception("Need numpy, from http://numpy.scipy.org/")

try:
    import Image
except:
    raise Exception("Need PIL, from http://www.pythonware.com/products/pil/")

if not hasattr(Image, 'fromarray'):
    raise Exception("Need PIL 1.1.6 or greater, from http://www.pythonware.com/products/pil/")

import os, re, sys, time, traceback, zlib

# Load our engine.

try:
    from aptus_engine import *
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

class NullProgressReporter:
    def begin(self):
        pass
    
    def progress(self, frac_done):
        pass
    
    def end(self):
        pass
    
jumps = [
    (-0.5,0.0,3.0,3.0),
    (-1.8605294939875601,1.0475516319329809e-005,2.288818359375e-005,2.288818359375e-005),
    (-1.8605327731370924,1.2700557708795141e-005,1.7881393432617188e-007,1.7881393432617188e-007),
    (0.45687170535326038,-0.34780396997928614,0.005859375,0.005859375),
    ]

class MandelbrotSet:
    def __init__(self, x0, y0, x1, y1, w, h, maxiter=999):
        self.x0, self.y0 = x0, y0
        self.rx, self.ry = (x1-x0)/w, (y0-y1)/h
        self.w, self.h = w, h
 
        self.maxiter = maxiter
        self.progress = NullProgressReporter()
        self.counts = None
        
    def from_pixel(self, x, y):
        return self.x0+self.rx*x, self.y0-self.ry*y

    def compute_pixels(self, trace=False):
        if self.counts is not None:
            return
        print "x, y %r step %r, maxiter %r, trace %r" % ((self.x0, self.y0), (self.rx, self.ry), self.maxiter, trace)

        clear_stats()
        set_params(self.x0, self.y0, self.rx, self.ry, self.maxiter)
        self.progress.begin()
        self.counts = numpy.zeros((self.h, self.w), dtype=numpy.uint32)
        mandelbrot_array(self.counts, self.progress.progress)
        self.progress.end()
        print get_stats()

    def color_pixels(self, palette):
        palarray = numpy.array(palette.colors, dtype=numpy.uint8)
        pix = palarray[(self.counts+palette.phase) % palarray.shape[0]]
        pix[self.counts == 0] = palette.incolor
        return pix
        
class wxMandelbrotSetViewer(wx.Frame):
    def __init__(self, xcenter, ycenter, xdiam, ydiam, w, h, maxiter):
        super(wxMandelbrotSetViewer, self).__init__(None, -1, 'Aptus')
 
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
        self.palette_index = 0
        self.palette = all_palettes[0]
        self.jump_index = 0
        
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
        #print "Coords: (%r,%r,%r,%r)" % (self.xcenter, self.ycenter, self.xdiam, self.ydiam)
        return MandelbrotSet(x0, y0, x1, y1, w, h, self.maxiter)
        
    def message(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Aptus', wx.OK | wx.ICON_WARNING)
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
        keycode = event.KeyCode
        if keycode == ord('S'):
            if shift:
                self.cmd_save_big()
            else:
                self.cmd_save()
        elif keycode == ord('I'):
            self.cmd_set_maxiter()
        elif keycode == ord('J'):
            self.jump_index += 1
            self.jump_index %= len(jumps)
            self.xcenter, self.ycenter, self.xdiam, self.ydiam = jumps[self.jump_index]
            self.set_view()
        elif keycode == ord('R'):
            self.cmd_redraw()
        elif keycode == ord(','):
            if shift:
                self.cmd_change_palette(-1)
            else:
                self.cmd_cycle_palette(-1)
        elif keycode == ord('.'):
            if shift:
                self.cmd_change_palette(1)
            else:
                self.cmd_cycle_palette(1)
        else:
            revmap = dict([(getattr(wx,n), n) for n in dir(wx) if n.startswith('WXK')])
            sym = revmap.get(keycode, "")
            if not sym:
                sym = "ord(%r)" % chr(keycode)
            print "Unmapped key: %r, %s, shift=%r" % (keycode, sym, shift)

    def on_paint(self, event):
        if not self.dc:
            self.dc = self.draw()
        dc = wx.PaintDC(self.panel)
        dc.Blit(0, 0, self.cw, self.ch, self.dc, 0, 0)
 
    def draw(self):
        wx.BeginBusyCursor()
        self.m.progress = ConsoleProgressReporter()
        self.m.compute_pixels(trace=self.trace)
        pix = self.m.color_pixels(self.palette)
        pix2 = None
        if 0:
            self.m.compute_pixels(trace=not self.trace)
            pix2 = self.m.color_pixels(self.palette)
        if 0:
            set_check_cycles(0)
            self.m.compute_pixels(trace=self.trace)
            pix2 = self.m.color_pixels(self.palette)
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
            "Aptus state (*.aptus)|*.aptus|"
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
            elif ext == 'aptus':
                ms = AptusState()
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
                m.compute_pixels(trace=self.trace)
                pix = m.color_pixels(self.palette)
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
        
    def cmd_cycle_palette(self, delta):
        self.palette.phase += delta
        self.palette.phase %= len(self.palette.colors)
        self.dc = None
        self.Refresh()
        
    def cmd_change_palette(self, delta):
        self.palette_index += delta
        self.palette_index %= len(all_palettes)
        self.palette = all_palettes[self.palette_index]
        self.dc = None
        self.Refresh()
        
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
        
class AptusState:
    def write(self, f):
        if isinstance(f, basestring):
            f = open(f, 'wb')
        print >>f, '{'
        self._write_item(f, 'what_is_this', 'An Aptus state file, version 1')
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

def main(args):        
    opts = AptusOptions()
    opts.read_args(args)
    
    app = wx.PySimpleApp()
    f = wxMandelbrotSetViewer(
        opts.center[0], opts.center[1],
        opts.diam[0], opts.diam[1],
        opts.size[0], opts.size[1],
        opts.maxiter
        )
    f.trace = opts.trace
    f.palette.phase = opts.palette_phase
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main(sys.argv[1:])
