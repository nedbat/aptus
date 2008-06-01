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

import math, os, os.path, sys, webbrowser
import wx.lib.layoutf
import wx.html 
import wx.lib.newevent
from wx.lib.evtmgr import eventManager

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
        
# Custom events
AptusColoringChangedEvent, EVT_APTUS_COLORING_CHANGED = wx.lib.newevent.NewEvent()
AptusComputationChangedEvent, EVT_APTUS_COMPUTATION_CHANGED = wx.lib.newevent.NewEvent()
AptusGeometryChangedEvent, EVT_APTUS_GEOMETRY_CHANGED = wx.lib.newevent.NewEvent()

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
id_new = wx.NewId()
id_show_youarehere = wx.NewId()


class AptusPanel(wx.Panel):
    """ A panel capable of drawing a Mandelbrot.
    """
    def __init__(self, parent, size=wx.DefaultSize):
        wx.Panel.__init__(self, parent, style=wx.NO_BORDER+wx.WANTS_CHARS, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.m = AptusCompute()
        
        # AptusCompute default values        
        self.m.palette = all_palettes[0]

        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)

    def recenter(self, center):
        """ Change the panel to display a new point on the Set.
        """
        self.m.center = center
        self.geometry_changed()

    # GUI helpers
    
    def fire_command(self, cmdid, data=None):
        # I'm not entirely sure about why this is the right event type to use,
        # but it works...
        evt = wx.CommandEvent(wx.wxEVT_COMMAND_TOOL_CLICKED)
        evt.SetId(cmdid)
        evt.SetClientData(data)
        wx.PostEvent(self, evt)
    
    def fire_event(self, evclass):
        self.GetEventHandler().ProcessEvent(evclass())
        
    def message(self, msg):
        top = self.GetTopLevelParent()
        top.message(msg)
        
    def coloring_changed(self):
        self.bitmap = None
        self.Refresh()
        self.fire_event(AptusColoringChangedEvent)

    def computation_changed(self):
        self.set_view()
        self.fire_event(AptusComputationChangedEvent)
        
    def geometry_changed(self):
        self.set_view()
        self.fire_event(AptusGeometryChangedEvent)
        
    # Event handlers
    
    def on_size(self, event_unused):
        self.check_size = True
        
    def on_idle(self, event_unused):
        if self.check_size and self.GetClientSize() != self.m.size:
            if self.GetClientSize() != (0,0):
                self.geometry_changed()

    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()
        
        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0, False)
        self.on_paint_extras(dc)
        
    def on_paint_extras(self, dc):
        pass
    
    # Output methods
    
    def draw_bitmap(self):
        """ Return a bitmap with the image to display in the window.
        """
        self.m.progress = GuiProgressReporter()
        self.m.compute_pixels()
        pix = self.m.color_mandel()
        return wx.BitmapFromBuffer(pix.shape[1], pix.shape[0], pix)

    def set_view(self):
        self.bitmap = None
        self.m.size = self.GetClientSize()
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
        self.geometry_changed()
        
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
        self.geometry_changed()
        
    # Event handlers
    
    def on_idle(self, event):
        self.set_cursor()
        AptusPanel.on_idle(self, event)
        
    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()
        
        dc = wx.AutoBufferedPaintDC(self)
        if self.panning:
            dc.SetBrush(wx.Brush(wx.Colour(224,224,128), wx.SOLID))
            dc.SetPen(wx.Pen(wx.Colour(224,224,128), 1, wx.SOLID))
            dc.DrawRectangle(0, 0, self.m.size[0], self.m.size[1])
            dc.DrawBitmap(self.bitmap, self.pt_pan[0]-self.pt_down[0], self.pt_pan[1]-self.pt_down[1], False)
        else:
            dc.DrawBitmap(self.bitmap, 0, 0, False)

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
            self.geometry_changed()
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
        
    def on_leave_window(self, event):
        if self.rubberrect:
            self.xor_rectangle(self.rubberrect)
        if self.panning:
            self.finish_panning(*event.GetPosition())
        self.reset_mousing()
        
    def on_key_down(self, event):
        # Turn keystrokes into commands.
        shift = event.ShiftDown()
        cmd = event.CmdDown()
        keycode = event.KeyCode
        if keycode == ord('A'):
            self.fire_command(id_set_angle)
        elif keycode == ord('B'):
            self.fire_command(id_set_bailout)
        elif keycode == ord('C'):
            self.fire_command(id_toggle_continuous)
        elif keycode == ord('H'):
            self.fire_command(id_help)
        elif keycode == ord('I'):
            self.fire_command(id_set_iter_limit)
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
        elif keycode == ord('L'):
            self.fire_command(id_show_youarehere)
        elif keycode == ord('N'):
            self.fire_command(id_new)
        elif keycode == ord('R'):
            self.fire_command(id_redraw)
        elif keycode == ord('S'):
            self.fire_command(id_save)
        elif keycode == ord('0'):
            self.fire_command(id_reset_palette)
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
        elif keycode == ord(' '):
            self.panning = True
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
            
    # Command helpers

    def set_value(self, dtitle, dprompt, attr, caster, when_done):
        cur_val = getattr(self.m, attr)
        dlg = wx.TextEntryDialog(self.GetTopLevelParent(), dtitle, dprompt, str(cur_val))

        if dlg.ShowModal() == wx.ID_OK:
            try:
                setattr(self.m, attr, caster(dlg.GetValue()))
                when_done()
            except ValueError, e:
                self.message("Couldn't set %s: %s" % (attr, e))

        dlg.Destroy()

    # Commands
    
    def cmd_set_angle(self, event_unused):
        self.set_value('Angle:', 'Set the angle of rotation', 'angle', float, self.geometry_changed)
        
    def cmd_set_iter_limit(self, event_unused):
        self.set_value('Iteration limit:', 'Set the iteration limit', 'iter_limit', int, self.computation_changed)
        
    def cmd_set_bailout(self, event_unused):
        self.set_value('Bailout:', 'Set the radius of the escape circle', 'bailout', float, self.computation_changed)

    def cmd_toggle_continuous(self, event_unused):
        self.m.continuous = not self.m.continuous
        self.computation_changed()

    def cmd_redraw(self, event_unused):
        self.set_view()
        
    def cmd_jump(self, event_unused):
        self.jump_index += 1
        self.jump_index %= len(jumps)
        self.m.center, self.m.diam = jumps[self.jump_index]
        self.geometry_changed()
        
    def cmd_cycle_palette(self, event):
        delta = event.GetClientData()
        self.m.palette_phase += delta
        self.m.palette_phase %= len(self.m.palette)
        self.coloring_changed()
        
    def cmd_scale_palette(self, event):
        factor = event.GetClientData()
        if self.m.continuous:
            self.m.palette_scale *= factor
            self.coloring_changed()
        
    def cmd_change_palette(self, event):
        delta = event.GetClientData()
        self.palette_index += delta
        self.palette_index %= len(all_palettes)
        self.m.palette = all_palettes[self.palette_index]
        self.m.palette_phase = 0
        self.m.palette_scale = 1.0
        self.coloring_changed()
    
    def cmd_adjust_palette(self, event):
        self.m.palette.adjust(**event.GetClientData())
        self.coloring_changed()

    def cmd_reset_palette(self, event_unused):
        self.m.palette_phase = 0
        self.m.palette_scale = 1.0
        self.m.palette.reset()
        self.coloring_changed()
        

class YouAreHerePanel(AptusPanel):
    """ A panel slaved to another AptusPanel to show where the master panel is
        on the Set.
    """
    def __init__(self, parent, mainwin):
        AptusPanel.__init__(self, parent)
        self.mainwin = mainwin
        self.hererect = None
        
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        
        eventManager.Register(self.on_coloring_changed, EVT_APTUS_COLORING_CHANGED, self.mainwin)
        eventManager.Register(self.on_computation_changed, EVT_APTUS_COMPUTATION_CHANGED, self.mainwin)
        eventManager.Register(self.on_geometry_changed, EVT_APTUS_GEOMETRY_CHANGED, self.mainwin)

        self.set_view()

    def on_destroy(self, event_unused):
        eventManager.DeregisterListener(self.on_coloring_changed)
        eventManager.DeregisterListener(self.on_computation_changed)
        eventManager.DeregisterListener(self.on_geometry_changed)

    def on_size(self, event):
        # Need to recalc our rectangle.
        self.hererect = None
        AptusPanel.on_size(self, event)

    def on_idle(self, event):
        # Let the AptusPanel resize.
        AptusPanel.on_idle(self, event)
        # Then we can recalc our rectangle.
        if not self.hererect:
            self.calc_rectangle()

    def on_left_up(self, event):
        mx, my = event.GetPosition()
        self.mainwin.recenter(self.m.coords_from_pixel(mx, my))
        
    def on_coloring_changed(self, event_unused):
        if self.m.copy_coloring(self.mainwin.m):
            self.coloring_changed()

    def on_computation_changed(self, event_unused):
        if self.m.copy_computation(self.mainwin.m):
            self.computation_changed()

    def on_geometry_changed(self, event_unused):
        # When a geometry_changed event comes in, copy the pertinent info from
        # the master window, then compute the window visible in our coordinates
        if self.m.angle != self.mainwin.m.angle:
            self.m.angle = self.mainwin.m.angle
            self.geometry_changed()
        self.calc_rectangle()

    def calc_rectangle(self):
        # Compute the master rectangle in our coords.
        ux, uy = self.m.pixel_from_coords(*self.mainwin.m.coords_from_pixel(0,0))
        lx, ly = self.m.pixel_from_coords(*self.mainwin.m.coords_from_pixel(*self.mainwin.m.size))
        ux = int(math.floor(ux))
        uy = int(math.floor(uy))
        lx = int(math.ceil(lx))+1
        ly = int(math.ceil(ly))+1
        w, h = lx-ux, ly-uy
        # Never draw the box smaller than 3 pixels
        if w < 3:
            w = 3
            ux -= 1     # Scooch back to adjust to the wider window.
        if h < 3:
            h = 3
            uy -= 1
        self.hererect = (ux, uy, w, h)
        self.Refresh()
        
    def on_paint_extras(self, dc):
        # Draw the mainwin view window.
        if self.hererect:
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.Pen(wx.Colour(255,255,255), 1, wx.SOLID))
            dc.DrawRectangle(*self.hererect)
        
class YouAreHereFrame(wx.Frame):
    def __init__(self, mainwin):
        wx.Frame.__init__(self, None, name='You are here', size=(250,250),
            style=wx.DEFAULT_FRAME_STYLE|wx.FRAME_TOOL_WINDOW)
        self.panel = YouAreHerePanel(self, mainwin)
        

class AptusMainFrame(wx.Frame):
    """ The main window frame of the Aptus app.
    """
    def __init__(self, args=None):
        wx.Frame.__init__(self, None, -1, 'Aptus')

        # Make the panel
        self.panel = AptusViewPanel(self)
        
        if args:
            opts = AptusOptions(self.panel.m)
            opts.read_args(args)
        self.panel.supersample = 1
        
        # Set the window icon
        ib = wx.IconBundle()
        ib.AddIconFromFile(data_file("icon48.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIconFromFile(data_file("icon32.png"), wx.BITMAP_TYPE_ANY)
        ib.AddIconFromFile(data_file("icon16.png"), wx.BITMAP_TYPE_ANY)
        self.SetIcons(ib)

        # Bind commands
        self.Bind(wx.EVT_MENU, self.cmd_new, id=id_new)
        self.Bind(wx.EVT_MENU, self.cmd_save, id=id_save)
        self.Bind(wx.EVT_MENU, self.cmd_help, id=id_help)
        self.Bind(wx.EVT_MENU, self.cmd_show_youarehere, id=id_show_youarehere)
        
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

    def cmd_new(self, event_unused):
        wx.GetApp().new_window()
        
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

    def cmd_show_youarehere(self, event_unused):
        YouAreHereFrame(self.panel).Show()


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
        
        lc = wx.lib.layoutf.Layoutf('t=t#1;b=t5#2;l=l#1;r=r#1', (self,ok))
        self.html.SetConstraints(lc)
        self.html.SetPage(html_text)
        
        lc = wx.lib.layoutf.Layoutf('b=b5#1;r=r5#1;w!80;h*', (self,))
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
<b>a</b>: set the angle of rotation.<br>
<b>b</b>: set the radius of the bailout circle.<br>
<b>c</b>: toggle continuous coloring.<br>
<b>h</b> or <b>?</b>: show this help.<br>
<b>i</b>: set the limit on iterations.<br>
<b>j</b>: jump among a few pre-determined locations.<br>
<b>n</b>: open a new window.<br>
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

<p>Built with
<a href='http://python.org'>Python</a>, <a href='http://wxpython.org'>wxPython</a>,
<a href='http://numpy.scipy.org/'>numpy</a>, and
<a href='http://www.pythonware.com/library/pil/handbook/index.htm'>PIL</a>.
Thanks to Rob McMullen and Paul Ollis for help with the wxPython drawing code.</p>
""" % terms


class AptusGuiApp(wx.PySimpleApp):
    def __init__(self, args):
        wx.PySimpleApp.__init__(self)
        self.new_window(args)
            
    def new_window(self, args=None):
        AptusMainFrame(args).Show()
        

def main(args):
    """ The main for the Aptus GUI.
    """
    AptusGuiApp(args).MainLoop()

if __name__ == '__main__':
    main(sys.argv[1:])
