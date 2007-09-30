# Started from http://www.howforge.com/mandelbrot-set-viewer-using-wxpython

from aptus.app import NullProgressReporter, ConsoleProgressReporter, AptusApp
from aptus.importer import importer
from aptus.options import AptusOptions, AptusState
from aptus.palettes import all_palettes

# Import third-party packages.
wx = importer('wx')
numpy = importer('numpy')
Image = importer('Image')

import os, re, sys, traceback, zlib

jumps = [
    ((-0.5,0.0), (3.0,3.0)),
    ((-1.8605294939875601,1.0475516319329809e-005), (2.288818359375e-005,2.288818359375e-005)),
    ((-1.8605327731370924,1.2700557708795141e-005), (1.7881393432617188e-007,1.7881393432617188e-007)),
    ((0.45687170535326038,-0.34780396997928614), (0.005859375,0.005859375)),
    ]

class AptusView(wx.Frame, AptusApp):
    def __init__(self, center, diam, size, iter_limit):
        wx.Frame.__init__(self, None, -1, 'Aptus')
        AptusApp.__init__(self)
 
        chromew, chromeh = 8, 28
        self.SetSize((size[0]+chromew, size[1]+chromeh))
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.panel.Bind(wx.EVT_RIGHT_UP, self.on_right_up)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.panel.Bind(wx.EVT_MOTION, self.on_motion)
        self.panel.Bind(wx.EVT_SIZE, self.on_size)
        self.panel.Bind(wx.EVT_IDLE, self.on_idle)
        self.panel.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

        # AptusApp values        
        self.center = center
        self.diam = diam
        self.iter_limit = iter_limit
        self.palette = all_palettes[0]
        self.palette_phase = 0
        
        # Gui values
        self.palette_index = 0
        self.jump_index = 0
        self.zoom = 2.0

        # Rubberbanding
        self.pt_down = None
        self.pt_other = None
        self.rectangle = False
        
        self.set_view()
        
    def set_view(self):
        self.size = self.GetClientSize()
        self.bitmap = None

        self.m = self.create_mandel()
        self.check_size = False
        self.Refresh()

    def dilate_view(self, center, scale):
        """ Change the view by a certain scale factor, keeping the center in the
            same spot.
        """
        # Refuse to zoom out so that the whole escape circle is visible: it makes
        # boundary tracing erase the entire thing!
        if self.diam[0] * scale >= 3.9:
            return
        cx = center[0] + (self.size[0]/2 - center[0]) * scale
        cy = center[1] + (self.size[1]/2 - center[1]) * scale
        self.center = self.m.coords_from_pixel(cx, cy)
        self.diam = (self.diam[0]*scale, self.diam[1]*scale)
        self.set_view()
        
    def message(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Aptus', wx.OK | wx.ICON_WARNING)
        dlg.ShowModal()
        dlg.Destroy()
        
    # Event handlers
    
    def on_left_down(self, event):
        self.pt_down = event.GetPosition()
        self.rectangle = False
        self.pt_other = None
        
    def on_motion(self, event):
        if not self.pt_down:
            return
        
        mx, my = event.GetPosition()
        if not self.rectangle:
            if abs(self.pt_down[0] - mx) > 10 or abs(self.pt_down[1] - my) > 10:
                self.rectangle = True
                self.SetCursor(wx.StockCursor(wx.CURSOR_CROSS))
           
        if self.rectangle:
            dc = wx.ClientDC(self.panel)
            dc.SetLogicalFunction(wx.XOR)
            dc.SetBrush(wx.Brush(wx.WHITE, wx.TRANSPARENT))
            dc.SetPen(wx.Pen(wx.WHITE, 1, wx.SOLID))
            
            if self.pt_other:
                # Erase the old rectangle.
                dc.DrawRectangle(self.pt_down[0], self.pt_down[1], self.pt_other[0]-self.pt_down[0], self.pt_other[1]-self.pt_down[1])
                
            dc.DrawRectangle(self.pt_down[0], self.pt_down[1], mx-self.pt_down[0], my-self.pt_down[1])
            self.pt_other = mx, my
            
    def on_left_up(self, event):
        if self.rectangle:
            ulx, uly = self.m.coords_from_pixel(*self.pt_down)
            lrx, lry = self.m.coords_from_pixel(*self.pt_other)
            self.center = ((ulx+lrx)/2, (uly+lry)/2)
            self.diam = (abs(ulx-lrx), abs(uly-lry))
            
            self.set_view()
        else:
            # Single-click: zoom in.
            scale = self.zoom
            if event.ControlDown():
                scale = (scale - 1.0)/10 + 1.0
            self.dilate_view(event.GetPosition(), 1.0/scale)
        
        self.pt_down = None
        self.rectangle = False
        self.pt_other = None
        self.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
         
    def on_right_up(self, event):
        scale = self.zoom
        if event.ControlDown():
            scale = (scale - 1.0)/10 + 1.0
        self.dilate_view(event.GetPosition(), scale)
 
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
        elif 0:
            revmap = dict([(getattr(wx,n), n) for n in dir(wx) if n.startswith('WXK')])
            sym = revmap.get(keycode, "")
            if not sym:
                sym = "ord(%r)" % chr(keycode)
            print "Unmapped key: %r, %s, shift=%r" % (keycode, sym, shift)

    def on_paint(self, event):
        if not self.bitmap:
            self.bitmap = self.draw()
        dc = wx.PaintDC(self.panel)
        dc.DrawBitmap(self.bitmap, 0, 0, False)
 
    def draw(self):
        wx.BeginBusyCursor()
        self.m.progress = ConsoleProgressReporter()
        self.m.compute_pixels()
        pix = self.m.color_pixels(self.palette, self.palette_phase)
        pix2 = None
        if 0:
            set_check_cycles(0)
            self.m.compute_pixels()
            pix2 = self.m.color_pixels(self.palette, self.palette_phase)
            set_check_cycles(1)
        if pix2 is not None:
            Image.fromarray(pix).save('one.png')
            Image.fromarray(pix2).save('two.png')
            wrong_count = numpy.sum(numpy.logical_not(numpy.equal(pix, pix2)))
            print wrong_count
        bmp = wx.BitmapFromBuffer(pix.shape[1], pix.shape[0], pix)
        wx.EndBusyCursor()
        return bmp

    # Command handlers.
    
    def cmd_save(self):
        wildcard = (
            "PNG image (*.png)|*.png|"     
            "Aptus state (*.aptus)|*.aptus|"
            "All files (*.*)|*.*"
            )

        dlg = wx.FileDialog(
            self, message="Save", defaultDir=os.getcwd(), 
            defaultFile="", style=wx.SAVE|wx.OVERWRITE_PROMPT, wildcard=wildcard, 
            )

        if dlg.ShowModal() == wx.ID_OK:
            ext = dlg.GetFilename().split('.')[-1].lower()
            if ext == 'png':
                image = wx.ImageFromBitmap(self.bitmap)
                im = Image.new('RGB', (image.GetWidth(), image.GetHeight()))
                im.fromstring(image.GetData())
                self.write_image(im, dlg.GetPath())
            elif ext == 'aptus':
                aptst = AptusState()
                self.write_state(aptst)
                aptst.write(dlg.GetPath())
            else:
                self.message("Don't understand how to write file '%s'" % dlg.GetFilename())
                
    def cmd_save_big(self):
        wildcard = (
            "PNG image (*.png)|*.png|"     
            "All files (*.*)|*.*"
            )

        dlg = wx.FileDialog(
            self, message="Save big image", defaultDir=os.getcwd(), 
            defaultFile="", style=wx.SAVE|wx.OVERWRITE_PROMPT, wildcard=wildcard, 
            )

        if dlg.ShowModal() == wx.ID_OK:
            ext = dlg.GetFilename().split('.')[-1].lower()
            if ext == 'png':
                w, h = 1680, 1050
                m = self.create_mandel((w*3, h*3))
                m.progress = ConsoleProgressReporter()
                m.compute_pixels()
                pix = m.color_pixels(self.palette, self.palette_phase)
                im = Image.fromarray(pix)
                im = im.resize((w,h), Image.ANTIALIAS)
                self.write_image(im, dlg.GetPath())

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
        self.cmd_redraw()
        
    def cmd_redraw(self):
        self.set_view()
        
    def cmd_cycle_palette(self, delta):
        self.palette_phase += delta
        self.bitmap = None
        self.Refresh()
        
    def cmd_change_palette(self, delta):
        self.palette_index += delta
        self.palette_index %= len(all_palettes)
        self.palette = all_palettes[self.palette_index]
        self.palette_phase = 0
        self.bitmap = None
        self.Refresh()
        
def main(args):
    """ The main for the Aptus GUI.
    """
    opts = AptusOptions()
    opts.read_args(args)
    
    app = wx.PySimpleApp()
    f = AptusView(
        opts.center,
        opts.diam,
        opts.size,
        opts.iter_limit
        )
    if opts.palette:
        f.palette = opts.palette
    f.palette_phase = opts.palette_phase
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main(sys.argv[1:])
