""" Aptus GUI
    http://nedbatchelder.com/code/aptus
    Copyright 2007-2008, Ned Batchelder
"""

from aptus import data_file, __version__
from aptus.compute import AptusCompute
from aptus.progress import ConsoleProgressReporter
from aptus.importer import importer
from aptus.options import AptusOptions, AptusState
from aptus.palettes import all_palettes

# Import third-party packages.
wx = importer('wx')
numpy = importer('numpy')
Image = importer('Image')

import os, os.path, sys, webbrowser
import wx.lib.layoutf  as layoutf
import wx.html 

# There are a few places we conditionalize on platform.
is_mac = ('wxMac' in wx.PlatformInfo)

# A pre-set list of places to visit, with the j command.
jumps = [
    ((-0.5,0.0), (3.0,3.0)),
    ((-1.8605294939875601,-1.0475516319329809e-005), (2.288818359375e-005,2.288818359375e-005)),
    ((-1.8605327731370924,-1.2700557708795141e-005), (1.7881393432617188e-007,1.7881393432617188e-007)),
    ((0.45687170535326038,0.34780396997928614), (0.005859375,0.005859375)),
    ]

class GuiProgressReporter(ConsoleProgressReporter):
    def begin(self):
        wx.BeginBusyCursor()
        ConsoleProgressReporter.begin(self)
        
    def end(self):
        ConsoleProgressReporter.end(self)
        wx.EndBusyCursor()
        

# Command ids
id_set_angle = wx.NewId()
id_save = wx.NewId()
id_set_iter_limit = wx.NewId()
id_set_bailout = wx.NewId()
id_toggle_continuous = wx.NewId()
id_toggle_julia = wx.NewId()
id_jump = wx.NewId()
id_redraw = wx.NewId()
id_change_palette = wx.NewId()
id_cycle_palette = wx.NewId()
id_scale_palette = wx.NewId()
id_adjust_palette = wx.NewId()
id_reset_palette = wx.NewId()
id_help = wx.NewId()


class AptusPanel(wx.Panel):
    """ A panel capable of drawing a Mandelbrot.
    """
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style=wx.NO_BORDER+wx.WANTS_CHARS)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.m = AptusCompute()
        
        # AptusCompute default values        
        self.m.palette = all_palettes[0]

        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)
                  
    # GUI helpers
    
    def fire_command(self, cmdid, data=None):
        # I'm not entirely sure about why this is the right event type to use,
        # but it works...
        evt = wx.CommandEvent(wx.wxEVT_COMMAND_TOOL_CLICKED)
        evt.SetId(cmdid)
        evt.SetClientData(data)
        if not self.ProcessEvent(evt):
            print "Whoa! Didn't handle %r" % cmdid
        
    def message(self, msg):
        top = self.GetTopLevelParent()
        top.message(msg)
        
    # Event handlers
    
    def on_size(self, event_unused):
        self.check_size = True
        
    def on_idle(self, event_unused):
        self.set_cursor()
        if self.check_size and self.GetClientSize() != self.m.size:
            if self.GetClientSize() != (0,0):
                self.set_view()

    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw()
        
        dc = wx.AutoBufferedPaintDC(self)
        if self.panning:
            dc.SetBrush(wx.Brush(wx.Colour(224,224,128), wx.SOLID))
            dc.SetPen(wx.Pen(wx.Colour(224,224,128), 1, wx.SOLID))
            dc.DrawRectangle(0, 0, self.m.size[0], self.m.size[1])
            dc.DrawBitmap(self.bitmap, self.pt_pan[0]-self.pt_down[0], self.pt_pan[1]-self.pt_down[1], False)
        else:
            dc.DrawBitmap(self.bitmap, 0, 0, False)

    # Ouptut methods
    
    def draw(self):
        """ Return a bitmap with the image to display in the window.
        """
        self.m.progress = GuiProgressReporter()
        self.m.compute_pixels()
        pix = self.m.color_mandel()
        return wx.BitmapFromBuffer(pix.shape[1], pix.shape[0], pix)

    def set_view(self):
        self.m.size = self.GetClientSize()
        self.bitmap = None

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


class AptusViewPanel(AptusPanel):
    """ A panel implementing the primary Aptus view and controller.
    """
    def __init__(self, parent):
        AptusPanel.__init__(self, parent)

        # Bind input events.
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
        self.Bind(wx.EVT_RIGHT_UP, self.on_right_up)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave_window)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_KEY_UP, self.on_key_up)

        self.Bind(wx.EVT_MENU, self.cmd_set_angle, id=id_set_angle)
        self.Bind(wx.EVT_MENU, self.cmd_set_iter_limit, id=id_set_iter_limit)
        self.Bind(wx.EVT_MENU, self.cmd_set_bailout, id=id_set_bailout)
        self.Bind(wx.EVT_MENU, self.cmd_toggle_continuous, id=id_toggle_continuous)
        self.Bind(wx.EVT_MENU, self.cmd_jump, id=id_jump)
        self.Bind(wx.EVT_MENU, self.cmd_redraw, id=id_redraw)
        self.Bind(wx.EVT_MENU, self.cmd_change_palette, id=id_change_palette)
        self.Bind(wx.EVT_MENU, self.cmd_cycle_palette, id=id_cycle_palette)
        self.Bind(wx.EVT_MENU, self.cmd_scale_palette, id=id_scale_palette)
        self.Bind(wx.EVT_MENU, self.cmd_adjust_palette, id=id_adjust_palette)
        self.Bind(wx.EVT_MENU, self.cmd_reset_palette, id=id_reset_palette)

        self.reset_mousing()

        # Gui state values
        self.palette_index = 0      # The index of the currently displayed palette
        self.jump_index = 0         # The index of the last jumped-to spot.
        self.zoom = 2.0             # A constant zoom amt per click.

    # Input methods
    
    def reset_mousing(self):
        """ Set all the mousing variables to turn off rubberbanding and panning.
        """
        self.pt_down = None
        self.rubberbanding = False
        self.rubberrect = None
        # Panning information.
        self.panning = False
        self.pt_pan = None
        self.pan_locked = False

    def finish_panning(self, mx, my):
        if not self.pt_down:
            return
        cx, cy = self.m.size[0]/2.0, self.m.size[1]/2.0
        cx -= mx - self.pt_down[0]
        cy -= my - self.pt_down[1]
        self.m.center = self.m.coords_from_pixel(cx, cy)
        self.set_view()
        
    def xor_rectangle(self, rect):
        dc = wx.ClientDC(self)
        dc.SetLogicalFunction(wx.XOR)
        dc.SetBrush(wx.Brush(wx.WHITE, wx.TRANSPARENT))
        dc.SetPen(wx.Pen(wx.WHITE, 1, wx.SOLID))
        dc.DrawRectangle(*rect)

    def set_cursor(self):
        # If we aren't taking input, then we shouldn't change the cursor.
        if not self.IsEnabled():
            return 
        # Set the proper cursor:
        if self.rubberbanding:
            self.SetCursor(wx.StockCursor(wx.CURSOR_MAGNIFIER))
        elif self.panning:
            self.SetCursor(wx.StockCursor(wx.CURSOR_SIZING))
        else:
            self.SetCursor(wx.StockCursor(wx.CURSOR_DEFAULT))

    def dilate_view(self, center, scale):
        """ Change the view by a certain scale factor, keeping the center in the
            same spot.
        """
        # Refuse to zoom out so that the whole escape circle is visible: it makes
        # boundary tracing erase the entire thing!
        if self.m.diam[0] * scale >= 3.9:
            return
        cx = center[0] + (self.m.size[0]/2 - center[0]) * scale
        cy = center[1] + (self.m.size[1]/2 - center[1]) * scale
        self.m.center = self.m.coords_from_pixel(cx, cy)
        self.m.diam = (self.m.diam[0]*scale, self.m.diam[1]*scale)
        self.set_view()
        
    # Event handlers
    
    def on_left_down(self, event):
        self.pt_down = event.GetPosition()
        self.rubberbanding = False
        if self.panning:
            self.pt_pan = self.pt_down
            self.pan_locked = False
            
    def on_middle_down(self, event):
        self.pt_down = event.GetPosition()
        self.rubberbanding = False
        self.panning = True
        self.pt_pan = self.pt_down
        self.pan_locked = False
        
    def on_motion(self, event):
        self.set_cursor()
        
        # We do nothing with mouse moves that aren't dragging.
        if not self.pt_down:
            return
        
        mx, my = event.GetPosition()
        
        if self.panning:
            if self.pt_pan != (mx, my):
                # We've moved the image: redraw it.
                self.pt_pan = (mx, my)
                self.pan_locked = True
                self.Refresh()
        else:
            if not self.rubberbanding:
                # Start rubberbanding when we have a 10-pixel rectangle at least.
                if abs(self.pt_down[0] - mx) > 10 or abs(self.pt_down[1] - my) > 10:
                    self.rubberbanding = True
    
            if self.rubberbanding:
                if self.rubberrect:
                    # Erase the old rectangle.
                    self.xor_rectangle(self.rubberrect)
                    
                self.rubberrect = (self.pt_down[0], self.pt_down[1], mx-self.pt_down[0], my-self.pt_down[1]) 
                self.xor_rectangle(self.rubberrect)
                
    def on_left_up(self, event):
        mx, my = event.GetPosition()
        if self.rubberbanding:
            # Set a new view that encloses the rectangle.
            px, py = self.pt_down
            ulx, uly = self.m.coords_from_pixel(px, py)
            lrx, lry = self.m.coords_from_pixel(mx, my)
            self.m.center = ((ulx+lrx)/2, (uly+lry)/2)
            self.m.diam = (abs(self.m.pixsize*(px-mx)), abs(self.m.pixsize*(py-my)))
            self.set_view()
        elif self.panning:
            self.finish_panning(mx, my)
        elif self.pt_down:
            # Single-click: zoom in.
            scale = self.zoom
            if event.CmdDown():
                scale = (scale - 1.0)/10 + 1.0
            self.dilate_view((mx, my), 1.0/scale)

        self.reset_mousing()        

    def on_middle_up(self, event):
        self.finish_panning(*event.GetPosition())
        self.reset_mousing()        

    def on_right_up(self, event):
        scale = self.zoom
        if event.CmdDown():
            scale = (scale - 1.0)/10 + 1.0
        self.dilate_view(event.GetPosition(), scale)
        self.reset_mousing()
        
    def on_leave_window(self, event_unused):
        if self.rubberrect:
            self.xor_rectangle(self.rubberrect)
        self.reset_mousing()
        
    def on_key_down(self, event):
        # Turn keystrokes into commands.
        shift = event.ShiftDown()
        cmd = event.CmdDown()
        keycode = event.KeyCode
        if keycode == ord('A'):
            self.fire_command(id_set_angle)
        elif keycode == ord('S'):
            self.fire_command(id_save)
        elif keycode == ord('I'):
            self.fire_command(id_set_iter_limit)
        elif keycode == ord('B'):
            self.fire_command(id_set_bailout)
        elif keycode == ord('C'):
            self.fire_command(id_toggle_continuous)
        elif keycode == ord('J'):
            if shift:
                if self.m.julia:
                    self.m.center = self.m.juliaxy
                else:
                    self.m.juliaxy = self.m.center
                    self.m.center, self.m.diam = (0.0,0.0), (3.0,3.0)
                self.m.julia = not self.m.julia
                self.set_view()
            else:
                self.fire_command(id_jump)
        elif keycode == ord('R'):
            self.fire_command(id_redraw)
        elif keycode in [ord(','), ord('<')]:
            if shift:
                self.fire_command(id_change_palette, -1)
            else:
                self.fire_command(id_cycle_palette, -1)
        elif keycode in [ord('.'), ord('>')]:
            if shift:
                self.fire_command(id_change_palette, 1)
            else:
                self.fire_command(id_cycle_palette, 1)
        elif keycode == ord(';'):
            self.fire_command(id_scale_palette, 1/(1.01 if cmd else 1.1))
        elif keycode == ord("'"):
            self.fire_command(id_scale_palette, 1.01 if cmd else 1.1)
        elif keycode in [ord('['), ord(']')]:
            kw = 'hue'
            delta = 1 if cmd else 10
            if keycode == ord('['):
                delta = -delta
            if shift:
                kw = 'saturation'
            self.fire_command(id_adjust_palette, {kw:delta})
        elif keycode == ord('0'):
            self.fire_command(id_reset_palette)
        elif keycode == ord(' '):
            self.panning = True
        elif keycode == ord('H'):
            self.fire_command(id_help)
        elif keycode == ord('/') and shift:
            self.fire_command(id_help)
        elif 0:
            # Debugging aid: find the symbol for the key we didn't handle.
            revmap = dict([(getattr(wx,n), n) for n in dir(wx) if n.startswith('WXK')])
            sym = revmap.get(keycode, "")
            if not sym:
                sym = "ord(%r)" % chr(keycode)
            print "Unmapped key: %r, %s, shift=%r, cmd=%r" % (keycode, sym, shift, cmd)

    def on_key_up(self, event):
        keycode = event.KeyCode
        if keycode == ord(' '):
            if not self.pan_locked:
                self.panning = False
            
    # Commands
    
    def cmd_set_angle(self, event_unused):
        dlg = wx.TextEntryDialog(
                self.GetTopLevelParent(), 'Angle:',
                'Set the angle of rotation', str(self.m.angle)
                )

        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.m.angle = float(dlg.GetValue())
                self.set_view()
            except ValueError, e:
                self.message("Couldn't set angle: %s" % e)

        dlg.Destroy()
        
    def cmd_set_iter_limit(self, event_unused):
        dlg = wx.TextEntryDialog(
                self.GetTopLevelParent(), 'Iteration limit:',
                'Set the iteration limit', str(self.m.iter_limit)
                )

        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.m.iter_limit = int(dlg.GetValue())
                self.set_view()
            except ValueError, e:
                self.message("Couldn't set iter_limit: %s" % e)

        dlg.Destroy()
        
    def cmd_set_bailout(self, event_unused):
        dlg = wx.TextEntryDialog(
                self.GetTopLevelParent(), 'Bailout:',
                'Set the radius of the escape circle', str(self.m.bailout)
                )

        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.m.bailout = float(dlg.GetValue())
                self.set_view()
            except ValueError, e:
                self.message("Couldn't set bailout: %s" % e)

        dlg.Destroy()
        
    def cmd_toggle_continuous(self, event_unused):
        self.m.continuous = not self.m.continuous
        self.set_view()

    def cmd_redraw(self, event_unused):
        self.set_view()
        
    def cmd_jump(self, event_unused):
        self.jump_index += 1
        self.jump_index %= len(jumps)
        self.m.center, self.m.diam = jumps[self.jump_index]
        self.set_view()
        
    def cmd_cycle_palette(self, event):
        delta = event.GetClientData()
        self.m.palette_phase += delta
        self.m.palette_phase %= len(self.m.palette)
        self.bitmap = None
        self.Refresh()
        
    def cmd_scale_palette(self, event):
        factor = event.GetClientData()
        if self.m.continuous:
            self.m.palette_scale *= factor
            self.bitmap = None
            self.Refresh()
        
    def cmd_change_palette(self, event):
        delta = event.GetClientData()
        self.palette_index += delta
        self.palette_index %= len(all_palettes)
        self.m.palette = all_palettes[self.palette_index]
        self.m.palette_phase = 0
        self.m.palette_scale = 1.0
        self.bitmap = None
        self.Refresh()
    
    def cmd_adjust_palette(self, event):
        self.m.palette.adjust(**event.GetClientData())
        self.bitmap = None
        self.Refresh()

    def cmd_reset_palette(self, event_unused):
        self.m.palette_phase = 0
        self.m.palette_scale = 1.0
        self.m.palette.reset()
        self.bitmap = None
        self.Refresh()
        

class AptusFrame(wx.Frame):
    """ The main window frame of the Aptus app.
    """
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Aptus')

        # Make the panel
        self.panel = AptusViewPanel(self)
        
        # Set the window icon
        ib = wx.IconBundle()
        ib.AddIconFromFile(data_file("icon48.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIconFromFile(data_file("icon32.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIconFromFile(data_file("icon16.png"), wx.BITMAP_TYPE_ANY)
        self.SetIcons(ib)

        # Bind commands
        self.Bind(wx.EVT_MENU, self.cmd_save, id=id_save)
        self.Bind(wx.EVT_MENU, self.cmd_help, id=id_help)

    def Show(self):
        # Override Show so we can set the view properly.
        self.SetClientSize(self.panel.m.size)
        self.panel.set_view()
        wx.Frame.Show(self)
        self.panel.SetFocus()
        
    def message(self, msg):
        dlg = wx.MessageDialog(self, msg, 'Aptus', wx.OK | wx.ICON_WARNING)
        dlg.ShowModal()
        dlg.Destroy()

    # Command handlers.
    
    def show_file_dialog(self, dlg):
        """ Show a file dialog, and do some post-processing on the result.
            Returns a pair: type, path.
            Type is one of the extensions from the wildcard choices.
        """
        if dlg.ShowModal() == wx.ID_OK:
            pth = dlg.Path
            ext = os.path.splitext(pth)[1].lower()
            idx = dlg.FilterIndex
            wildcards = dlg.Wildcard.split('|')
            wildcard = wildcards[2*idx+1]
            if wildcard == '*.*':
                if ext:
                    typ = ext[1:]
                else:
                    typ = ''
            elif '*'+ext in wildcards:
                # The extension of the file is a recognized extension:
                # Use it regardless of the file type chosen in the picker.
                typ = ext[1:]
            else:
                typ = wildcard.split('.')[-1].lower()
            if ext == '' and typ != '':
                pth += '.' + typ
            return typ, pth
        else:
            return None, None
        
    def cmd_save(self, event_unused):
        wildcard = (
            "PNG image (*.png)|*.png|"     
            "Aptus state (*.aptus)|*.aptus|"
            "All files (*.*)|*.*"
            )

        dlg = wx.FileDialog(
            self, message="Save", defaultDir=os.getcwd(), 
            defaultFile="", style=wx.SAVE|wx.OVERWRITE_PROMPT, wildcard=wildcard, 
            )

        typ, pth = self.show_file_dialog(dlg)
        if typ:
            if typ == 'png':
                self.panel.write_png(pth)
            elif typ == 'aptus':
                self.panel.write_aptus(pth)
            else:
                self.message("Don't understand how to write file '%s'" % pth)
                
    def cmd_help(self, event_unused):
        dlg = HtmlDialog(self, help_html, "Aptus")
        dlg.ShowModal()


class HtmlDialog(wx.Dialog):
    def __init__(self, parent, html_text, caption,
                 pos=wx.DefaultPosition, size=(500,530),
                 style=wx.DEFAULT_DIALOG_STYLE):
        wx.Dialog.__init__(self, parent, -1, caption, pos, size, style)
        x, y = pos
        if x == -1 and y == -1:
            self.CenterOnScreen(wx.BOTH)

        self.html = wx.html.HtmlWindow(self, -1)
        self.html.Bind(wx.html.EVT_HTML_LINK_CLICKED, self.on_link_clicked)
        ok = wx.Button(self, wx.ID_OK, "OK")
        ok.SetDefault()
        
        lc = layoutf.Layoutf('t=t#1;b=t5#2;l=l#1;r=r#1', (self,ok))
        self.html.SetConstraints(lc)
        self.html.SetPage(html_text)
        
        lc = layoutf.Layoutf('b=b5#1;r=r5#1;w!80;h*', (self,))
        ok.SetConstraints(lc)
        
        self.SetAutoLayout(1)
        self.Layout()

    def on_link_clicked(self, event):
        url = event.GetLinkInfo().GetHref()
        webbrowser.open(url)
        
# The help text

terms = {
    'ctrl': 'cmd' if is_mac else 'ctrl',
    'iconsrc': data_file('icon48.png'),
    'version': __version__,
    }
    
help_html = """\
<table width='100%%'>
<tr>
    <td width='50' valign='top'><img src='%(iconsrc)s'/></td>
    <td valign='top'>
        <b>Aptus %(version)s</b>, Mandelbrot set explorer.<br>
        Copyright 2007-2008, Ned Batchelder.<br>
        <a href='http://nedbatchelder.com/code/aptus'>http://nedbatchelder.com/code/aptus</a>
    </td>
</tr>
</table>

<p><b>Controls:</b></p>

<blockquote>
<b>b</b>: set the radius of the bailout circle.<br>
<b>c</b>: toggle continuous coloring.<br>
<b>h</b> or <b>?</b>: show this help.<br>
<b>i</b>: set the limit on iterations.<br>
<b>j</b>: jump among a few pre-determined locations.<br>
<b>r</b>: redraw the current image.<br>
<b>s</b>: save the current image or settings.<br>
<b>&lt;</b> or <b>&gt;</b>: switch to the next palette.<br>
<b>,</b> or <b>.</b>: cycle the current palette one color.<br>
<b>;</b> or <b>'</b>: stretch the palette colors (+%(ctrl)s: just a little), if continuous.<br>
<b>[</b> or <b>]</b>: adjust the hue of the palette (+%(ctrl)s: just a little).<br>
<b>{</b> or <b>}</b>: adjust the saturation of the palette (+%(ctrl)s: just a little).<br>
<b>0</b> (zero): reset all palette adjustments.<br>
<b>space</b>: drag mode: click to drag the image to a new position.<br>
<b>left-click</b>: zoom in (+%(ctrl)s: just a little).<br>
<b>right-click</b>: zoom out (+%(ctrl)s: just a little).<br>
<b>left-drag</b>: select a new rectangle to display.<br>
<b>middle-drag</b>: drag the image to a new position.
</blockquote>

<p>Thanks to Rob McMullen and Paul Ollis for help with the drawing code.</p>
""" % terms

class AptusGuiApp(wx.PySimpleApp):
    def __init__(self, args):
        wx.PySimpleApp.__init__(self)
        f = AptusFrame()
        opts = AptusOptions(f.panel.m)
        opts.read_args(args)
        f.panel.supersample = 1
        f.Show()
        
def main(args):
    """ The main for the Aptus GUI.
    """
    AptusGuiApp(args).MainLoop()

if __name__ == '__main__':
    main(sys.argv[1:])
