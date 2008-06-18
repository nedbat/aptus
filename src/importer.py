# Manage import dependencies.

def importer(name):
    if name == 'wx':
        url = "http://wxpython.org/"
        try:
            import wx
        except ImportError:
            raise Exception("Need wxPython, from " + url)
        if not hasattr(wx, 'BitmapFromBuffer'):
            raise Exception("Need wxPython 2.8 or greater, from " + url)
        return wx
    
    elif name == 'numpy':
        url = "http://numpy.scipy.org/"
        try:
            import numpy
        except ImportError:
            raise Exception("Need numpy, from " + url)
        return numpy
    
    elif name == 'Image':
        url = "http://pythonware.com/products/pil/"
        try:
            import Image
        except ImportError:
            raise Exception("Need PIL, from " + url)
        if not hasattr(Image, 'fromarray'):
            raise Exception("Need PIL 1.1.6 or greater, from " + url)
        return Image

    elif name == 'AptEngine':
        try:
            from aptus.engine import AptEngine
        except:
            raise Exception("There is no Python implementation of the compute engine!")
        return AptEngine
