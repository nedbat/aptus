""" YouAreHere stuff for Aptus.
"""

from aptus.gui.computepanel import ComputePanel
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame, ListeningWindowMixin
from aptus import settings

import wx
from wx.lib.scrolledpanel import ScrolledPanel

import math


MIN_RECT = 20
ParentComputePanel = ComputePanel

class YouAreHereWin(ParentComputePanel, ListeningWindowMixin):
    """ A panel slaved to another ComputePanel to show where the master panel is
        on the Set.  These are designed to be stacked in a YouAreHereStack to
        show successive magnifications.
        
        Two windows are referenced: the main view window (so that we can change
        the view), and the window our rectangle represents.  This can be either
        the next YouAreHereWin in the stack, or the main view window in the case
        of the last window in the stack.
    """
    def __init__(self, parent, mainwin, center, diam, size=wx.DefaultSize):
        ParentComputePanel.__init__(self, parent, size=size)
        ListeningWindowMixin.__init__(self)
        
        self.mainwin = mainwin
        self.hererect = None
        self.diam = diam

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        
        self.register_listener(self.on_coloring_changed, EVT_APTUS_COLORING_CHANGED, self.mainwin)
        self.register_listener(self.on_computation_changed, EVT_APTUS_COMPUTATION_CHANGED, self.mainwin)

        self.set_ref_window(mainwin)
        
        self.set_geometry(center=center, diam=diam)
        self.on_coloring_changed(None)
        self.on_computation_changed(None)
        self.on_geometry_changed(None)

        self.dragging = False
        self.drag_pt = None

    def set_ref_window(self, refwin):
        """ Set the other window that our rectangle models.
        """
        # Deregister the old geometry listener
        self.deregister_listener(self.on_geometry_changed)

        self.rectwin = refwin
        
        # Register the new listener and calc the rectangle.
        self.register_listener(self.on_geometry_changed, EVT_APTUS_GEOMETRY_CHANGED, self.rectwin)
        self.calc_rectangle()
        
    def on_size(self, event):
        # Need to recalc our rectangle.
        self.hererect = None
        ParentComputePanel.on_size(self, event)

    def on_idle(self, event):
        # Let the ComputePanel resize.
        ParentComputePanel.on_idle(self, event)
        # Then we can recalc our rectangle.
        if not self.hererect:
            self.calc_rectangle()

    def on_left_down(self, event):
        mouse_pt = event.GetPosition()
        if self.hererect.Contains(mouse_pt):
            self.dragging = True
            self.drag_pt = mouse_pt

    def on_left_up(self, event):
        # Reposition the main window.
        if self.dragging:
            if self.mainwin == self.rectwin:
                # We already show the actual view, so just recenter on the center
                # of the rectangle.
                mx = self.hererect.x + self.hererect.width/2
                my = self.hererect.y + self.hererect.height/2
                self.mainwin.set_geometry(center=self.compute.coords_from_pixel(mx, my))
            else:
                # Dragging the rect: set the view to invlude the four corners of
                # the rectangle.
                ulr, uli = self.compute.coords_from_pixel(*self.hererect.TopLeft)
                lrr, lri = self.compute.coords_from_pixel(*self.hererect.BottomRight)
                self.mainwin.set_geometry(corners=(ulr, uli, lrr, lri))
            self.dragging = False
        else:
            # Clicking outside the rect: recenter there.
            mx, my = event.GetPosition()
            self.mainwin.set_geometry(center=self.compute.coords_from_pixel(mx, my), diam=self.diam)

    def on_motion(self, event):
        self.set_cursor(event)
        if self.dragging:
            mouse_pt = event.GetPosition()
            self.hererect.Offset((mouse_pt.x - self.drag_pt.x, mouse_pt.y - self.drag_pt.y))
            self.drag_pt = mouse_pt
            self.Refresh()

    def set_cursor(self, event):
        # Set the proper cursor:
        mouse_pt = event.GetPosition()
        if self.dragging or (self.hererect and self.hererect.Contains(mouse_pt)):
            self.SetCursor(wx.Cursor(wx.CURSOR_SIZING))
        else:
            self.SetCursor(wx.Cursor(wx.CURSOR_DEFAULT))

    def on_coloring_changed(self, event_unused):
        if self.compute.copy_coloring(self.mainwin.compute):
            self.coloring_changed()

    def on_computation_changed(self, event_unused):
        if self.compute.copy_computation(self.mainwin.compute):
            self.computation_changed()

    def on_geometry_changed(self, event_unused):
        # When a geometry_changed event comes in, copy the pertinent info from
        # the master window, then compute the window visible in our coordinates
        if self.compute.angle != self.mainwin.compute.angle:
            self.compute.angle = self.mainwin.compute.angle
            self.geometry_changed()
        self.calc_rectangle()

    def calc_rectangle(self):
        # Compute the master rectangle in our coords.
        ulx, uly = self.compute.pixel_from_coords(*self.rectwin.compute.coords_from_pixel(0,0))
        lrx, lry = self.compute.pixel_from_coords(*self.rectwin.compute.coords_from_pixel(*self.rectwin.compute.size))
        ulx = int(math.floor(ulx))
        uly = int(math.floor(uly))
        lrx = int(math.ceil(lrx))+1
        lry = int(math.ceil(lry))+1
        w, h = lrx-ulx, lry-uly
        # Never draw the box smaller than 3 pixels
        if w < 3:
            w = 3
            ulx -= 1     # Scooch back to adjust to the wider window.
        if h < 3:
            h = 3
            uly -= 1
        self.hererect = wx.Rect(ulx, uly, w, h)
        self.Refresh()
        
    def on_paint_extras(self, dc):
        # Draw the mainwin view window.
        if self.hererect:
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.Pen(wx.Colour(255,255,255), 1, wx.SOLID))
            dc.DrawRectangle(*self.hererect)


class YouAreHereStack(ScrolledPanel, ListeningWindowMixin):
    """ A scrolled panel with a stack of YouAreHereWin's, each at a successive
        magnification.
    """
    def __init__(self, parent, viewwin, size=wx.DefaultSize):
        ScrolledPanel.__init__(self, parent, size=size)
        ListeningWindowMixin.__init__(self)
        
        self.winsize = 250
        self.minrect = MIN_RECT
        self.stepfactor = float(self.winsize)/self.minrect
        
        self.viewwin = viewwin
        self.sizer = wx.FlexGridSizer(cols=1, vgap=2, hgap=0)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()
        
        self.register_listener(self.on_geometry_changed, EVT_APTUS_GEOMETRY_CHANGED, self.viewwin)

        self.on_geometry_changed()

    def on_geometry_changed(self, event_unused=None):
        mode = self.viewwin.compute.mode
        diam = min(settings.diam(mode))
        
        # How many YouAreHereWin's will we need?
        targetdiam = min(self.viewwin.compute.diam)
        num_wins = int(math.ceil((math.log(diam)-math.log(targetdiam))/math.log(self.stepfactor)))
        num_wins = num_wins or 1
        
        cur_wins = list(self.sizer.Children)
        last = None
        for i in range(num_wins):
            if i == 0:
                # Don't recenter the topmost YouAreHere.
                center = settings.center(mode)
            else:
                center = self.viewwin.compute.center
            if i < len(cur_wins):
                # Re-using an existing window in the stack.
                win = cur_wins[i].Window
                win.set_geometry(center=center, diam=(diam,diam))
            else:
                # Going deeper: have to make a new window.
                win = YouAreHereWin(
                        self, self.viewwin, center=center,
                        diam=(diam,diam), size=(self.winsize, self.winsize)
                        )
                self.sizer.Add(win)
            if last:
                last.set_ref_window(win)
            last = win
            diam /= self.stepfactor

        # The last window needs to draw a rectangle for the view window.
        last.set_ref_window(self.viewwin)

        # Remove windows we no longer need.
        if 0:
            for child in cur_wins[num_wins:]:
                self.sizer.Remove(child.Window)
                child.Window.Destroy()

        for i in reversed(range(num_wins, len(cur_wins))):
            print("Thing to delete:", cur_wins[i])
            print("the window:", cur_wins[i].Window)
            win = cur_wins[i].Window
            self.sizer.Remove(i)
            win.Destroy()

        self.sizer.Layout()
        self.SetupScrolling()


class YouAreHereFrame(AptusToolFrame):
    def __init__(self, mainframe, viewwin):
        AptusToolFrame.__init__(self, mainframe, title='You are here', size=(250,550))
        self.stack = YouAreHereStack(self, viewwin)
