""" Aptus Mandelbrot set viewer and renderer.
"""

__version__ = '2.0'

import os.path

def data_file(fname):
    """ Return the path to a data file of ours.
    """
    return os.path.join(os.path.split(__file__)[0], fname)
