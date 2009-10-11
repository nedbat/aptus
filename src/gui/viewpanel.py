from aptus.importer import importer
from aptus.palettes import all_palettes
from aptus.progress import ConsoleProgressReporter, IntervalProgressReporter, AggregateProgressReporter
from aptus import settings

from aptus.gui.computepanel import ComputePanel
from aptus.gui.ids import *

# Import third-party packages.
wx = importer('wx')


# A pre-set list of places to visit, with the j command.
JUMPS = [
    (settings.center(), settings.diam()),
    ((-1.8605294939875601,-1.0475516319329809e-005), (2.288818359375e-005,2.288818359375e-005)),
    ((-1.8605327731370924,-1.2700557708795141e-005), (1.7881393432617188e-007,1.7881393432617188e-007)),
    ((0.45687170535326038,0.34780396997928614), (0.005859375,0.005859375)),
    ]


class GuiProgressReporter:
    """ A progress reporter tied into the GUI.
    """
    def __init__(self, aptview):
        self.aptview = aptview
        
    def begin(self):
        wx.BeginBusyCursor()
        
    def progress(self, frac_done_unused, info_unused=''):
        self.aptview.draw_progress()
        # Yield so that repaints of the screen will happen.
        wx.SafeYield()

    def end(self):
        wx.EndBusyCursor()


class AptusViewPanel(ComputePanel):
    """ A panel implementing the primary Aptus view and controller.
    """
    def __init__(self, parent):
        ComputePanel.__init__(self, parent)

        self.compute.quiet = False

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
        self.Bind(wx.EVT_MENU, self.cmd_toggle_continuous, id=id_toggle_continuous)
        self.Bind(wx.EVT_MENU, self.cmd_jump, id=id_jump)
        self.Bind(wx.EVT_MENU, self.cmd_redraw, id=id_redraw)
        self.Bind(wx.EVT_MENU, self.cmd_change_palette, id=id_change_palette)
        self.Bind(wx.EVT_MENU, self.cmd_set_palette, id=id_set_palette)
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
        
        # When shift is down, then we're indicating points.
        self.indicating_pt = False
        self.indicated_pt = (-1, -1)
 
    def finish_panning(self, mx, my):
        if not self.pt_down:
            return
        cx, cy = self.compute.size[0]/2.0, self.compute.size[1]/2.0
        cx -= mx - self.pt_down[0]
        cy -= my - self.pt_down[1]
        self.compute.center = self.compute.coords_from_pixel(cx, cy)
        self.geometry_changed()
        
    def xor_rectangle(self, rect):
        dc = wx.ClientDC(self)
        dc.SetLogicalFunction(wx.XOR)
        dc.SetBrush(wx.Brush(wx.WHITE, wx.TRANSPARENT))
        dc.SetPen(wx.Pen(wx.WHITE, 1, wx.SOLID))
        dc.DrawRectangle(*rect)

    def set_cursor(self, event_unused):
        # If we aren't taking input, then we shouldn't change the cursor.
        if not self.GetTopLevelParent().IsActive():
            return

        # Set the proper cursor:
        if self.rubberbanding:
            self.SetCursor(wx.StockCursor(wx.CURSOR_MAGNIFIER))
        elif self.panning:
            self.SetCursor(wx.StockCursor(wx.CURSOR_SIZING))
        elif self.indicating_pt:
            import aptus.gui.resources
            curimg = aptus.gui.resources.getCrosshairImage()
            curimg.SetOptionInt(wx.IMAGE_OPTION_CUR_HOTSPOT_X, 7)
            curimg.SetOptionInt(wx.IMAGE_OPTION_CUR_HOTSPOT_Y, 7)
            cur = wx.CursorFromImage(curimg)
            self.SetCursor(cur)
            #self.SetCursor(wx.StockCursor(wx.CURSOR_CROSS))
        else:
            self.SetCursor(wx.StockCursor(wx.CURSOR_DEFAULT))

    def indicate_point(self, event):
        """ Use the given event to indicate a point, maybe.
        """
        if hasattr(event, 'ShiftDown'):
            self.indicating_pt = event.ShiftDown()
        else:
            self.indicating_pt = wx.GetMouseState().shiftDown

        if self.indicating_pt:
            if hasattr(event, 'GetPosition'):
                pt = event.GetPosition()
            else:
                ms = wx.GetMouseState()
                pt = self.ScreenToClient((ms.x, ms.y))
            if self.GetRect().Contains(pt) and pt != self.indicated_pt:
                self.indicated_pt = pt
                self.fire_event(AptusIndicatePointEvent, point=pt)

    def dilate_view(self, center, scale):
        """ Change the view by a certain scale factor, keeping the center in the
            same spot.
        """
        # Refuse to zoom out so that the whole escape circle is visible: it makes
        # boundary tracing erase the entire thing!
        if self.compute.diam[0] * scale >= 3.9:
            return
        cx = center[0] + (self.compute.size[0]/2 - center[0]) * scale
        cy = center[1] + (self.compute.size[1]/2 - center[1]) * scale
        self.compute.center = self.compute.coords_from_pixel(cx, cy)
        self.compute.diam = (self.compute.diam[0]*scale, self.compute.diam[1]*scale)
        self.geometry_changed()
        
    def make_progress_reporter(self):
        # Construct a progress reporter that suits us.  Write to the console,
        # and keep the GUI updated, but only once a second.
        prorep = AggregateProgressReporter()
        prorep.add(ConsoleProgressReporter())
        prorep.add(GuiProgressReporter(self))
        return IntervalProgressReporter(1, prorep)
    
    # Event handlers
    
    def on_idle(self, event):
        self.indicate_point(event)
        self.set_cursor(event)
        ComputePanel.on_idle(self, event)
        
    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()
        
        dc = wx.AutoBufferedPaintDC(self)
        if self.panning:
            dc.SetBrush(wx.Brush(wx.Colour(224,224,128), wx.SOLID))
            dc.SetPen(wx.Pen(wx.Colour(224,224,128), 1, wx.SOLID))
            dc.DrawRectangle(0, 0, self.compute.size[0], self.compute.size[1])
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
        self.indicate_point(event)
        self.set_cursor(event)
        
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
            ulr, uli = self.compute.coords_from_pixel(px, py)
            lrr, lri = self.compute.coords_from_pixel(mx, my)
            self.set_geometry(corners=(ulr, uli, lrr, lri))
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
        elif keycode == ord('C'):
            self.fire_command(id_toggle_continuous)
        elif keycode == ord('F'):
            self.fire_command(id_fullscreen)
        elif keycode == ord('H'):
            self.fire_command(id_help)
        elif keycode == ord('I'):
            self.fire_command(id_set_iter_limit)
        elif keycode == ord('J'):
            if shift:
                self.fire_command(id_show_julia)
            else:
                self.fire_command(id_jump)
        elif keycode == ord('L'):
            self.fire_command(id_show_youarehere)
        elif keycode == ord('N'):
            self.fire_command(id_new)
        elif keycode == ord('P'):
            self.fire_command(id_show_palettes)
        elif keycode == ord('Q'):
            self.fire_command(id_show_pointinfo)
        elif keycode == ord('R'):
            self.fire_command(id_redraw)
        elif keycode == ord('S'):
            self.fire_command(id_save)
        elif keycode == ord('V'):
            self.fire_command(id_show_stats)
        elif keycode == ord('W'):
            self.fire_command(id_window_size)

        elif keycode == ord('0'):       # zero
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
        cur_val = getattr(self.compute, attr)
        dlg = wx.TextEntryDialog(self.GetTopLevelParent(), dtitle, dprompt, str(cur_val))

        if dlg.ShowModal() == wx.ID_OK:
            try:
                setattr(self.compute, attr, caster(dlg.GetValue()))
                when_done()
            except ValueError, e:
                self.message("Couldn't set %s: %s" % (attr, e))

        dlg.Destroy()

    def palette_changed(self):
        """ Use the self.palette_index to set a new palette.
        """
        self.compute.palette = all_palettes[self.palette_index]
        self.compute.palette_phase = 0
        self.compute.palette_scale = 1.0
        self.coloring_changed()

    # Commands
    
    def cmd_set_angle(self, event_unused):
        self.set_value('Angle:', 'Set the angle of rotation', 'angle', float, self.geometry_changed)
        
    def cmd_set_iter_limit(self, event_unused):
        self.set_value('Iteration limit:', 'Set the iteration limit', 'iter_limit', int, self.computation_changed)
        
    def cmd_toggle_continuous(self, event_unused):
        self.compute.continuous = not self.compute.continuous
        self.computation_changed()

    def cmd_redraw(self, event_unused):
        self.compute.clear_results()
        #cp = list(self.compute.eng.cycle_params)
        #cp[1] += 1
        #self.compute.eng.cycle_params = tuple(cp)
        #print cp
        self.set_view()
        
    def cmd_jump(self, event_unused):
        self.jump_index += 1
        self.jump_index %= len(JUMPS)
        self.compute.center, self.compute.diam = JUMPS[self.jump_index]
        self.geometry_changed()
        
    def cmd_cycle_palette(self, event):
        delta = event.GetClientData()
        self.compute.palette_phase += delta
        self.coloring_changed()
        
    def cmd_scale_palette(self, event):
        factor = event.GetClientData()
        if self.compute.continuous:
            self.compute.palette_scale *= factor
            self.coloring_changed()
        
    def cmd_change_palette(self, event):
        delta = event.GetClientData()
        self.palette_index += delta
        self.palette_index %= len(all_palettes)
        self.palette_changed()
        
    def cmd_set_palette(self, event):
        self.palette_index = event.GetClientData()
        self.palette_changed()
        
    def cmd_adjust_palette(self, event):
        self.compute.palette.adjust(**event.GetClientData())
        self.coloring_changed()

    def cmd_reset_palette(self, event_unused):
        self.compute.palette_phase = 0
        self.compute.palette_scale = 1.0
        self.compute.palette.reset()
        self.coloring_changed()
