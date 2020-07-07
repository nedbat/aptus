import wx

from aptus import settings
from aptus.gui.computepanel import ComputePanel
from aptus.gui.ids import *
from aptus.palettes import all_palettes
from aptus.progress import ConsoleProgressReporter, IntervalProgressReporter


class AptusViewPanel(ComputePanel):
    """ A panel implementing the primary Aptus view and controller.
    """
    def __init__(self, parent):
        ComputePanel.__init__(self, parent)

        self.compute.quiet = False

        # Bind input events.
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)

        self.Bind(wx.EVT_MENU, self.cmd_set_iter_limit, id=id_set_iter_limit)
        self.Bind(wx.EVT_MENU, self.cmd_redraw, id=id_redraw)

    # Input methods

    def make_progress_reporter(self):
        # Construct a progress reporter that suits us.  Write to the console,
        # but only once a second.
        return IntervalProgressReporter(1, ConsoleProgressReporter())

    # Event handlers

    def on_paint(self, event_unused):
        if not self.bitmap:
            self.bitmap = self.draw_bitmap()

        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0, False)

    def on_key_down(self, event):
        # Turn keystrokes into commands.
        shift = event.ShiftDown()
        cmd = event.CmdDown()
        keycode = event.KeyCode
        #print("Look:", keycode)
        if keycode == ord('I'):
            self.fire_command(id_set_iter_limit)
        elif keycode == ord('N'):
            self.fire_command(id_new)
        elif keycode == ord('O'):
            self.fire_command(id_open)
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
        elif 0:
            # Debugging aid: find the symbol for the key we didn't handle.
            revmap = dict([(getattr(wx,n), n) for n in dir(wx) if n.startswith('WXK')])
            sym = revmap.get(keycode, "")
            if not sym:
                sym = "ord(%r)" % chr(keycode)
            #print("Unmapped key: %r, %s, shift=%r, cmd=%r" % (keycode, sym, shift, cmd))

    # Command helpers

    def set_value(self, dtitle, dprompt, attr, caster, when_done):
        cur_val = getattr(self.compute, attr)
        dlg = wx.TextEntryDialog(self.GetTopLevelParent(), dtitle, dprompt, str(cur_val))

        if dlg.ShowModal() == wx.ID_OK:
            try:
                setattr(self.compute, attr, caster(dlg.GetValue()))
                when_done()
            except ValueError as e:
                self.message("Couldn't set %s: %s" % (attr, e))

        dlg.Destroy()

    # Commands

    def cmd_set_iter_limit(self, event_unused):
        self.set_value('Iteration limit:', 'Set the iteration limit', 'iter_limit', int, self.computation_changed)

    def cmd_redraw(self, event_unused):
        self.compute.clear_results()
        self.set_view()
