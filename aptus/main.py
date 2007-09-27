# Started from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

from aptus.timeutil import duration, future
from aptus.options import AptusOptions, AptusState
from aptus.palettes import all_palettes
from aptus.importer import importer

# Import third-party packages.

wx = importer('wx')
numpy = importer('numpy')
Image = importer('Image')

# Import our extension engine.
AptEngine = importer('AptEngine')

import os, re, sys, time, traceback, zlib

class NullProgressReporter:
    def begin(self):
        pass
    
    def progress(self, frac_done):
        pass
    
    def end(self):
        pass
    
jumps = [
    ((-0.5,0.0), (3.0,3.0)),
    ((-1.8605294939875601,1.0475516319329809e-005), (2.288818359375e-005,2.288818359375e-005)),
    ((-1.8605327731370924,1.2700557708795141e-005), (1.7881393432617188e-007,1.7881393432617188e-007)),
    ((0.45687170535326038,-0.34780396997928614), (0.005859375,0.005859375)),
    ]

class AptusMandelbrot(AptEngine):
    def __init__(self, center, diam, size, iter_limit=999):
        self.center = center
        self.diam = diam
        self.size = size
        
        pixsize = max(self.diam[0] / size[0], self.diam[1] / size[1])
        diam = pixsize * size[0], pixsize * size[1]
        
        self.xy0 = (center[0] - diam[0]/2, center[1] - diam[1]/2)
        self.xyd = (pixsize, pixsize)
        #print "Coords: (%r,%r,%r,%r)" % (self.xcenter, self.ycenter, xdiam, ydiam)
 
        self.iter_limit = iter_limit
        self.progress = NullProgressReporter()
        self.counts = None
        
    def coords_from_pixel(self, x, y):
        return self.xy0[0]+self.xyd[0]*x, self.xy0[1]+self.xyd[1]*y

    def compute_pixels(self):
        if self.counts is not None:
            return
        print "x, y %r step %r, iter_limit %r, size %r" % (self.xy0, self.xyd, self.iter_limit, self.size)

        self.clear_stats()
        self.progress.begin()
        self.counts = numpy.zeros((self.size[1], self.size[0]), dtype=numpy.uint32)
        self.mandelbrot_array(self.counts, self.progress.progress)
        self.progress.end()
        print self.get_stats()

    def color_pixels(self, palette):
        palarray = numpy.array(palette.colors, dtype=numpy.uint8)
        pix = palarray[(self.counts+palette.phase) % palarray.shape[0]]
        pix[self.counts == 0] = palette.incolor
        return pix

    def write_state(self, aptus_state):
        aptus_state.center = self.center
        aptus_state.diam = self.diam
        aptus_state.iter_limit = self.iter_limit
        
class AptusView(wx.Frame):
    def __init__(self, center, diam, size, iter_limit):
        super(AptusView, self).__init__(None, -1, 'Aptus')
 
        chromew, chromeh = 8, 28
        self.SetSize((size[0]+chromew, size[1]+chromeh))
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_LEFT_UP, self.on_zoom_in)
        self.panel.Bind(wx.EVT_RIGHT_UP, self.on_zoom_out)
        self.panel.Bind(wx.EVT_SIZE, self.on_size)
        self.panel.Bind(wx.EVT_IDLE, self.on_idle)
        self.panel.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        
        self.center = center
        self.diam = diam
        self.iter_limit = iter_limit
        self.set_view()
        self.palette_index = 0
        self.palette = all_palettes[0]
        self.jump_index = 0
        
    def set_view(self):
        self.size = self.GetClientSize()
        self.bitmap = wx.EmptyBitmap(*self.size)
        self.dc = None

        self.m = self.create_mandel(self.size)
        self.check_size = False
        self.Refresh()

    def create_mandel(self, size):
        return AptusMandelbrot(self.center, self.diam, size, self.iter_limit)
        
    def message(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Aptus', wx.OK | wx.ICON_WARNING)
        dlg.ShowModal()
        dlg.Destroy()
        
    def on_zoom_in(self, event):
        self.center = self.m.coords_from_pixel(event.GetX(), event.GetY())
        self.diam = (self.diam[0]/2.0, self.diam[1]/2.0)
        self.set_view()
 
    def on_zoom_out(self, event):
        self.center = self.m.coords_from_pixel(event.GetX(), event.GetY())
        self.diam = (self.diam[0]*2.0, self.diam[1]*2.0)
        self.set_view()
 
    def on_size(self, event):
        self.check_size = True
        
    def on_idle(self, event):
        if self.check_size and self.GetClientSize() != self.size:
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
            self.cmd_set_iter_limit()
        elif keycode == ord('J'):
            self.jump_index += 1
            self.jump_index %= len(jumps)
            self.center, self.diam = jumps[self.jump_index]
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
        dc.Blit(0, 0, self.size[0], self.size[1], self.dc, 0, 0)
 
    def draw(self):
        wx.BeginBusyCursor()
        self.m.progress = ConsoleProgressReporter()
        self.m.compute_pixels()
        pix = self.m.color_pixels(self.palette)
        pix2 = None
        if 0:
            set_check_cycles(0)
            self.m.compute_pixels()
            pix2 = self.m.color_pixels(self.palette)
            set_check_cycles(1)
        if pix2 is not None:
            Image.fromarray(pix).save('one.png')
            Image.fromarray(pix2).save('two.png')
            wrong_count = numpy.sum(numpy.logical_not(numpy.equal(pix, pix2)))
            print wrong_count
        img = wx.EmptyImage(*self.size)
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
                self.m.write_state(ms)
                ms.size = self.size
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
                m = self.create_mandel((w*3, h*3))
                m.progress = ConsoleProgressReporter()
                m.compute_pixels()
                pix = m.color_pixels(self.palette)
                im = Image.fromarray(pix)
                im = im.resize((w,h), Image.ANTIALIAS)
                im.save(dlg.GetPath())

    def cmd_set_iter_limit(self):
        dlg = wx.TextEntryDialog(
                self, 'Iteration limit:',
                'Set the iteration limit', str(self.iter_limit)
                )

        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.iter_limit = int(dlg.GetValue())
            except Exception, e:
                self.message("Couldn't set iter_limit: %s" % e)

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
        
def main(args):        
    opts = AptusOptions()
    opts.read_args(args)
    
    app = wx.PySimpleApp()
    f = AptusView(
        opts.center,
        opts.diam,
        opts.size,
        opts.iter_limit
        )
    f.palette.phase = opts.palette_phase
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main(sys.argv[1:])
