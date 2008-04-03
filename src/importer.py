# Manage import dependencies.

def importer(name):
    if name == 'wx':
        try:
            import wx
        except:
            raise Exception("Need wxPython, from http://wxpython.org/")
        if not hasattr(wx, 'BitmapFromBuffer'):
            raise Exception("Need wxPython 2.8 or greater, from http://wxpython.org/")
        return wx
    
    elif name == 'numpy':
        try:
            import numpy
        except:
            raise Exception("Need numpy, from http://numpy.scipy.org/")
        return numpy
    
    elif name == 'Image':
        try:
            import Image
        except:
            raise Exception("Need PIL, from http://pythonware.com/products/pil/")
        if not hasattr(Image, 'fromarray'):
            raise Exception("Need PIL 1.1.6 or greater, from http://pythonware.com/products/pil/")
        return Image

    elif name == 'AptEngine':
        try:
            from aptus.engine import AptEngine
        except:
            print "*** Warning: using slow engine ***"
            from slow_engine import AptEngine
        return AptEngine
