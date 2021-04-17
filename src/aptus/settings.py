""" Defaults and settings for Aptus.
"""

# For now, simple constants.  Someday, maybe preferences.

mandelbrot_center = -0.6, 0.0
mandelbrot_diam = 3.0
explorer_size = 600, 600

julia_center = 0.0, 0.0
julia_diam = 3.0

def center(mode='mandelbrot'):
    """ What's the default center for this computation mode?
    """
    if mode == 'mandelbrot':
        c = mandelbrot_center
    elif mode == 'julia':
        c = julia_center
    else:
        raise Exception("Unknown mode: %r" % mode)
    return c

def diam(mode='mandelbrot'):
    """ What's the default diameter for this computation mode?
    """
    if mode == 'mandelbrot':
        d = mandelbrot_diam
    elif mode == 'julia':
        d = julia_diam
    else:
        raise Exception("Unknown mode: %r" % mode)
    return d, d
