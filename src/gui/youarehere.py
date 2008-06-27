""" YouAreHere stuff for Aptus.
"""

from aptus.importer import importer
from aptus.gui.computepanel import ComputePanel
from aptus.gui.ids import *
from aptus.gui.misc import AptusToolFrame

wx = importer("wx")
from wx.lib.evtmgr import eventManager
from wx.lib.scrolledpanel import ScrolledPanel

import math


class YouAreHerePanel(ComputePanel):
    """ A panel slaved to another ComputePanel to show where the master panel is
        on the Set.
    """
    def __init__(self, parent, mainwin, size=wx.DefaultSize):
        ComputePanel.__init__(self, parent, size=size)
        self.mainwin = mainwin
        self.hererect = None
        
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_IDLE, self.on_idle)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        
        eventManager.Register(self.on_coloring_changed, EVT_APTUS_COLORING_CHANGED, self.mainwin)
        eventManager.Register(self.on_computation_changed, EVT_APTUS_COMPUTATION_CHANGED, self.mainwin)
        eventManager.Register(self.on_geometry_changed, EVT_APTUS_GEOMETRY_CHANGED, self.mainwin)

        self.set_view()
        self.on_coloring_changed(None)
        self.on_computation_changed(None)
        self.on_geometry_changed(None)
        
        self.dragging = False
        self.drag_pt = None

    def on_destroy(self, event_unused):
        eventManager.DeregisterListener(self.on_coloring_changed)
        eventManager.DeregisterListener(self.on_computation_changed)
        eventManager.DeregisterListener(self.on_geometry_changed)

    def on_size(self, event):
        # Need to recalc our rectangle.
        self.hererect = None
        ComputePanel.on_size(self, event)

    def on_idle(self, event):
        # Let the ComputePanel resize.
        ComputePanel.on_idle(self, event)
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
            # Dragging the rect: recenter on its center.
            mx = self.hererect.x + self.hererect.width/2
            my = self.hererect.y + self.hererect.height/2
        else:
            # Clicking outside the rect: recenter there.
            mx, my = event.GetPosition()

        self.mainwin.recenter(self.m.coords_from_pixel(mx, my))
        self.dragging = False

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
        if self.dragging or self.hererect.Contains(mouse_pt):
            self.SetCursor(wx.StockCursor(wx.CURSOR_SIZING))
        else:
            self.SetCursor(wx.StockCursor(wx.CURSOR_DEFAULT))

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
        self.hererect = wx.Rect(ux, uy, w, h)
        self.Refresh()
        
    def on_paint_extras(self, dc):
        # Draw the mainwin view window.
        if self.hererect:
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.Pen(wx.Colour(255,255,255), 1, wx.SOLID))
            dc.DrawRectangle(*self.hererect)


class YouAreHereFrame(AptusToolFrame):
    def __init__(self, mainwin):
        AptusToolFrame.__init__(self, mainwin, title='You are here', size=(250,250))
        self.panel = YouAreHerePanel(self, mainwin)
        

class FancyYouAreHereFrame(wx.Frame):
    def __init__(self, mainwin):
        wx.Frame.__init__(self, None, name='You are here', size=(250,250),
            style=wx.DEFAULT_FRAME_STYLE|wx.FRAME_TOOL_WINDOW)

        scrolledpanel = ScrolledPanel(self, -1, size=(140, 300),
                                 style = wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER, name="panel1")

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(YouAreHerePanel(scrolledpanel, mainwin, size=(250,250)))
        box.Add(YouAreHerePanel(scrolledpanel, mainwin, size=(250,250)))
        box.Add(YouAreHerePanel(scrolledpanel, mainwin, size=(250,250)))
        
        scrolledpanel.SetSizer(box)
        scrolledpanel.SetAutoLayout(1)
        scrolledpanel.SetupScrolling()
