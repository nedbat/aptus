""" Ids and events for Aptus.
"""

from aptus.importer import importer
wx = importer('wx')
import wx.lib.newevent

## Custom events

# The coloring of a view window changed.
AptusColoringChangedEvent, EVT_APTUS_COLORING_CHANGED = wx.lib.newevent.NewEvent()

# The computation parameters of a view window changed: iterlimit, continuous, etc.
AptusComputationChangedEvent, EVT_APTUS_COMPUTATION_CHANGED = wx.lib.newevent.NewEvent()

# The geometry of a view window changed: position, angle, size.
AptusGeometryChangedEvent, EVT_APTUS_GEOMETRY_CHANGED = wx.lib.newevent.NewEvent()

# A view window just finished recomputing.
AptusRecomputedEvent, EVT_APTUS_RECOMPUTED = wx.lib.newevent.NewEvent()


## Command ids

id_set_angle = wx.NewId()
id_save = wx.NewId()
id_set_iter_limit = wx.NewId()
id_set_bailout = wx.NewId()
id_toggle_continuous = wx.NewId()
id_toggle_julia = wx.NewId()
id_jump = wx.NewId()
id_redraw = wx.NewId()
id_change_palette = wx.NewId()      # data: palette index delta
id_set_palette = wx.NewId()         # data: palette index
id_cycle_palette = wx.NewId()
id_scale_palette = wx.NewId()
id_adjust_palette = wx.NewId()
id_reset_palette = wx.NewId()
id_help = wx.NewId()
id_new = wx.NewId()
id_show_youarehere = wx.NewId()
id_show_palettes = wx.NewId()
id_show_stats = wx.NewId()
id_show_pointinfo = wx.NewId()
id_show_julia = wx.NewId()
