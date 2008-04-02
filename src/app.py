""" Fundamental application building blocks for Aptus.
"""

from aptus import __version__
from aptus.importer import importer
from aptus.options import AptusState
from aptus.timeutil import duration, future
from aptus.tinyjson import dumps

# Import our extension engine.
AptEngine = importer('AptEngine')

numpy = importer('numpy')

import math, time

class NullProgressReporter:
    def begin(self):
        pass
    
    def progress(self, frac_done):
        pass
    
    def end(self):
        pass
    
class AptusApp:
    """ A mixin class for any Aptus application.
    """
    def __init__(self):
        self.center = -0.5, 0.0
        self.diam = 3.0, 3.0
        self.size = 600, 600
        self.angle = 0.0
        self.iter_limit = 999
        self.bailout = 0
        self.palette = None
        self.palette_phase = 0
        self.palette_scale = 1.0
        self.supersample = 1
        self.outfile = 'Aptus.png'
        self.continuous = False
        
    def create_mandel(self):
        size = self.size[0]*self.supersample, self.size[1]*self.supersample
        m = AptusMandelbrot(self.center, self.diam, size, self.angle, self.iter_limit)
        # If bailout was never specified, then default differently based on
        # continuous or discrete coloring.
        if self.bailout:
            m.bailout = self.bailout
        elif self.continuous:
            m.bailout = 100.0
        else:
            m.bailout = 2.0
        if self.continuous:
            m.cont_levels = m.blend_colors = 256
        return m
    
    def color_mandel(self, m):
        return m.color_pixels(self.palette, self.palette_phase, self.palette_scale)
    
    def write_image(self, im, fpath, mandel=None):
        """ Write the image `im` to the path `fpath`.  If `mandel` is not None,
            it is the AptusMandelbrot from `create_mandel` that computed the
            image.
        """
        # PNG info mojo from: http://blog.modp.com/2007/08/python-pil-and-png-metadata-take-2.html
        from PIL import PngImagePlugin
        aptst = AptusState(self)
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "Aptus %s" % __version__)
        info.add_text("Aptus State", aptst.write_string())
        if mandel:
            info.add_text("Aptus Stats", dumps(mandel.get_stats()))
        im.save(fpath, 'PNG', pnginfo=info)
    
class AptusMandelbrot(AptEngine):
    """ A Python wrapper around the C AptEngine class.
    """
    def __init__(self, center, diam, size, angle, iter_limit):
        self.size = size
        self.angle = angle
        
        self.pixsize = max(diam[0] / size[0], diam[1] / size[1])
        diam = self.pixsize * size[0], self.pixsize * size[1]
        
        dx = math.cos(math.radians(self.angle)) * self.pixsize
        dy = math.sin(math.radians(self.angle)) * self.pixsize

        # The upper-left corner is computed from the center, minus the radii,
        # plus half a pixel, so that we're sampling the center of the pixel.
        self.xydxdy = (dx, dy, dy, -dx)
        halfsizew = size[0]/2.0 - 0.5
        halfsizeh = size[1]/2.0 - 0.5
        self.xy0 = (
            center[0] - halfsizew * self.xydxdy[0] - halfsizeh * self.xydxdy[2],
            center[1] - halfsizew * self.xydxdy[1] - halfsizeh * self.xydxdy[3]
            )
 
        self.iter_limit = iter_limit
        self.progress = NullProgressReporter()
        self.counts = None
        self.trace_boundary = 1
        
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
            self.xy0[0] + self.xydxdy[0]*x + self.xydxdy[2]*y,
            self.xy0[1] + self.xydxdy[1]*x + self.xydxdy[3]*y
            )

    def compute_pixels(self):
        if self.counts is not None:
            return
        print "x, y %r step %r, angle %r, iter_limit %r, size %r" % (self.xy0, self.pixsize, self.angle, self.iter_limit, self.size)

        self.clear_stats()
        self.progress.begin()
        self.counts = numpy.zeros((self.size[1], self.size[0]), dtype=numpy.uint32)
        self.mandelbrot_array(self.counts, self.progress.progress)
        self.progress.end()
        print self.get_stats()

    def color_pixels(self, palette, phase, scale=1.0):
        pix = numpy.zeros((self.counts.shape[0], self.counts.shape[1], 3), dtype=numpy.uint8)
        self.apply_palette(self.counts, palette.color_bytes(), phase, scale, palette.incolor, pix)
        return pix

class ConsoleProgressReporter:
    def begin(self):
        self.start = time.time()
        self.latest = self.start

    def progress(self, frac_done, info=''):
        now = time.time()
        if now - self.latest > 10:
            so_far = int(now - self.start)
            to_go = int(so_far / frac_done * (1-frac_done))
            if info:
                info = '  ' + info
            print "%5.2f%%: %11s done, %11s to go, eta %10s%s" % (
                frac_done*100, duration(so_far), duration(to_go), future(to_go), info
                )
            self.latest = now
    
    def end(self):
        total = time.time() - self.start
        print "Total: %s (%.2fs)" % (duration(total), total)
