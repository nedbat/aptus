import wx

class MainFrame(wx.Frame):
    def __init__(self, parent, ID, title):
        wx.Frame.__init__(self, parent, ID, title,
                          wx.DefaultPosition, wx.Size(200, 100))

        Panel = wx.Panel(self, -1)
        #TopSizer = wx.BoxSizer(wx.VERTICAL)
        #Panel.SetSizer(TopSizer)

        #Text = wx.TextCtrl(Panel, -1, "Type text here")
        #TopSizer.Add(Text, 1, wx.EXPAND)

        #Text.Bind(wx.EVT_KEY_DOWN, self.OnKeyText)
        Panel.Bind(wx.EVT_KEY_DOWN, self.OnKeyPanel)
        Panel.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        #self.Bind(wx.EVT_KEY_DOWN, self.OnKeyFrame)
        Panel.SetFocus()

    def OnKeyText(self, event):
        print "OnKeyText"
        print "\tShould Propagate %i" % event.ShouldPropagate()
        Level = event.StopPropagation()
        print "\tPropagate level %i" % Level
        # Try: event.ResumePropagation(x), x=1,2,3,...
        event.ResumePropagation(Level)
        event.Skip()

    def OnLeftDown(self, event):
        print "OnLeftDown"
        event.GetEventObject().SetFocus()

    def OnKeyPanel(self, event):
        print "OnKeyPanel"
        print "\tShould Propagate %i" % event.ShouldPropagate()
        Level = event.StopPropagation()
        print "\tPropagate level %i" % Level
        event.ResumePropagation(Level)
        event.Skip()

    def OnKeyFrame(self, event):
        print "OnKeyFrame"
        print "\tShould Propagate %i" % event.ShouldPropagate()
        Level = event.StopPropagation()
        print "\tPropagate level %i" % Level
        event.ResumePropagation(Level)
        event.Skip()

class MyApp(wx.App):
    def OnInit(self):
        Frame = MainFrame(None, -1, "Event Propagation Demo")
        Frame.Show(True)
        #self.SetTopWindow(Frame)
        return True

if __name__ == '__main__':
    App = MyApp(0)
    App.MainLoop()

