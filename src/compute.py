""" Mandelbrot computation.
"""

from aptus import __version__
from aptus.importer import importer
from aptus.options import AptusState
from aptus.tinyjson import dumps
from aptus.progress import NullProgressReporter

# Import our extension engine.
AptEngine = importer('AptEngine')

numpy = importer('numpy')

import copy, math

class AptusCompute:
    """ The Mandelbrot compute class.  It wraps the AptEngine to provide pythonic
        convenience.
    """
    def __init__(self):
        # geometry
        self.center = -0.5, 0.0
        self.diam = 3.0, 3.0
        self.size = 600, 600
        self.angle = 0.0
        # computation
        self.iter_limit = 999
        self.bailout = 0
        self.continuous = False
        self.supersample = 1
        # coloring
        self.palette = None
        self.palette_phase = 0
        self.palette_scale = 1.0
        # other
        self.julia = False
        self.juliaxy = 0.0, 0.0
        self.outfile = 'Aptus.png'
        
    def create_mandel(self):
        self.eng = AptEngine()
        
        # ssize is the dimensions of the sample array, in samples across and down.
        self.ssize = self.size[0]*self.supersample, self.size[1]*self.supersample
        
        # pixsize is the size of a single sample, in real units.
        self.pixsize = max(self.diam[0] / self.ssize[0], self.diam[1] / self.ssize[1])
        
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.pixsize
        dy = math.sin(rad) * self.pixsize

        # The upper-left corner is computed from the center, minus the radii,
        # plus half a pixel, so that we're sampling the center of the pixel.
        self.eng.xydxdy = (dx, dy, dy, -dx)
        halfsizew = self.ssize[0]/2.0 - 0.5
        halfsizeh = self.ssize[1]/2.0 - 0.5
        self.eng.xy0 = (
            self.center[0] - halfsizew * self.eng.xydxdy[0] - halfsizeh * self.eng.xydxdy[2],
            self.center[1] - halfsizew * self.eng.xydxdy[1] - halfsizeh * self.eng.xydxdy[3]
            )
 
        self.eng.iter_limit = self.iter_limit
        self.eng.trace_boundary = 1
        self.progress = NullProgressReporter()
        self.counts = None

        # If bailout was never specified, then default differently based on
        # continuous or discrete coloring.
        if self.bailout:
            self.eng.bailout = self.bailout
        elif self.continuous:
            self.eng.bailout = 100.0
        else:
            self.eng.bailout = 2.0
        if self.continuous:
            self.eng.cont_levels = self.eng.blend_colors = 256
        self.eng.julia = int(self.julia)
        if self.julia:
            self.eng.juliaxy = self.juliaxy
            self.eng.trace_boundary = 0

    def copy_coloring(self, other):
        """ Copy the coloring attributes from other to self, returning True if
            any of them actually changed.
        """
        return self.copy_attributes(other, ['palette', 'palette_phase', 'palette_scale'])

    def copy_computation(self, other):
        """ Copy the computation attributes from other to self, returning True if
            any of them actually changed.
        """
        return self.copy_attributes(other, ['iter_limit', 'bailout', 'continuous', 'supersample'])
    
    def copy_attributes(self, other, attrs):
        """ Copy a list of attributes from other to self, returning True if
            any of them actually changed.
        """
        changed = False
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                otherval = copy.deepcopy(getattr(other, attr))
                setattr(self, attr, otherval)
                changed = True
        return changed
    
    def color_mandel(self):
        pix = numpy.zeros((self.counts.shape[0], self.counts.shape[1], 3), dtype=numpy.uint8)
        self.eng.apply_palette(self.counts, self.palette.color_bytes(), self.palette_phase, self.palette_scale, self.palette.incolor, pix)
        return pix
    
    def coords_from_pixel(self, x, y):
        """ Get the coords of a pixel in the grid. Note that x and y can be
            fractional.
        """
        # The .5 adjustment is because the grid is aligned to the center of the
        # pixels, but we need to return the upper-left of the pixel so that other
        # math comes out right.
        x = float(x) - 0.5
        y = float(y) - 0.5
        return (
            self.eng.xy0[0] + self.eng.xydxdy[0]*x + self.eng.xydxdy[2]*y,
            self.eng.xy0[1] + self.eng.xydxdy[1]*x + self.eng.xydxdy[3]*y
            )

    def pixel_from_coords(self, mx, my):
        """ Get the pixel coords containing the real coords.
        """
        d0, d1, d2, d3 = self.eng.xydxdy
        xy00, xy01 = self.eng.xy0
        # Thanks, Maxima!
        px = (d2*(my-xy01)+d3*xy00-d3*mx)/(d1*d2-d0*d3)
        py = -(d0*(my-xy01)+d1*xy00-d1*mx)/(d1*d2-d0*d3)
        return px, py

    def compute_pixels(self):
        if self.counts is not None:
            return

        print "x, y %r step %r, angle %r, iter_limit %r, size %r" % (
            self.eng.xy0, self.pixsize, self.angle, self.eng.iter_limit, self.ssize
            )

        self.eng.clear_stats()
        self.progress.begin()
        self.counts = numpy.zeros((self.ssize[1], self.ssize[0]), dtype=numpy.uint32)
        self.eng.mandelbrot_array(self.counts, self.progress.progress)
        self.progress.end()
        print self.eng.get_stats()

    def write_image(self, im, fpath):
        """ Write the image `im` to the path `fpath`.
        """
        # PNG info mojo from: http://blog.modp.com/2007/08/python-pil-and-png-metadata-take-2.html
        from PIL import PngImagePlugin
        aptst = AptusState(self)
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "Aptus %s" % __version__)
        info.add_text("Aptus State", aptst.write_string())
        info.add_text("Aptus Stats", dumps(self.eng.get_stats()))
        im.save(fpath, 'PNG', pnginfo=info)
