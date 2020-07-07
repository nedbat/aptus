""" Ids and events for Aptus.
"""

import wx
import wx.lib.newevent

## Custom events

# The computation parameters of a view window changed: iterlimit, continuous, etc.
AptusComputationChangedEvent, EVT_APTUS_COMPUTATION_CHANGED = wx.lib.newevent.NewEvent()

# The geometry of a view window changed: position, angle, size.
AptusGeometryChangedEvent, EVT_APTUS_GEOMETRY_CHANGED = wx.lib.newevent.NewEvent()

# A view window just finished recomputing.
AptusRecomputedEvent, EVT_APTUS_RECOMPUTED = wx.lib.newevent.NewEvent()

# User indicated a new point in a view window, point= is client point coords.
AptusIndicatePointEvent, EVT_APTUS_INDICATEPOINT = wx.lib.newevent.NewEvent()

## Command ids

id_save = wx.NewId()
id_set_iter_limit = wx.NewId()
id_redraw = wx.NewId()
id_new = wx.NewId()
id_window_size = wx.NewId()
id_open = wx.NewId()
