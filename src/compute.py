""" Mandelbrot computation.
"""

from aptus import __version__
from aptus.importer import importer
from aptus.options import AptusState
from aptus.palettes import all_palettes
from aptus.tinyjson import dumps
from aptus.progress import NullProgressReporter
from aptus import settings

# Import our extension engine.
AptEngine = importer('AptEngine')

numpy = importer('numpy')

import copy, math

class AptusCompute:
    """ The Mandelbrot compute class.  It wraps the AptEngine to provide pythonic
        convenience.
        
        There are two coordinate systems at work here: the ri plane is the
        fractal plane, real and imaginary floats.  The xy plane are screen coordinates,
        in pixels, usually integers.
    """
    def __init__(self):
        # geometry
        self.center = settings.mandelbrot_center
        self.diam = settings.mandelbrot_diam, settings.mandelbrot_diam
        self.size = settings.explorer_size
        self.angle = 0.0
        self._geometry_attributes = ['center', 'diam', 'size', 'angle']
        
        # computation
        self.iter_limit = 999
        self.continuous = False
        self.supersample = 1
        self.mode = 'mandelbrot'
        self.rijulia = 0.0, 0.0
        self._computation_attributes = ['iter_limit', 'continuous', 'supersample', 'mode', 'rijulia']
        
        # coloring
        self.palette = all_palettes[0]
        self.palette_phase = 0
        self.palette_scale = 1.0
        self._coloring_attributes = ['palette', 'palette_phase', 'palette_scale']
        
        # other
        self.outfile = 'Aptus.png'
        self.quiet = False

        # The C extension for doing the heavy lifting.
        self.eng = AptEngine()
        
        # counts is a numpy array of 32bit ints: the iteration counts at each pixel.
        self.counts = None
        # status is a numpy array of 8bit ints that tracks the boundary trace
        # status of each pixel: 0 for not computed, 1 for computed but not traced,
        # 2 for traced.
        self.status = None
        # An array for the output pixels.
        self.pix = None
        
        self.pixels_computed = False
        self._clear_old_geometry()
        
    def _record_old_geometry(self):
        """ Call this before any of the geometry settings change, to possibly
            optimize the next computation.
        """
        self.old_ssize = self.ssize
        self.old_pixsize = self.pixsize
        self.old_ri0 = self.eng.ri0
        self.old_angle = self.angle
        for a in self._computation_attributes:
            setattr(self, 'old_'+a, getattr(self, a))
        
    def _clear_old_geometry(self):
        self.old_ssize = (0,0)
        self.old_pixsize = 0
        self.old_ri0 = (0,0)
        self.old_angle = 0
        for a in self._computation_attributes:
            setattr(self, 'old_'+a, 0)
    
    def computation_changed(self):
        for a in self._computation_attributes:
            if getattr(self, 'old_'+a) != getattr(self, a):
                return True
        return False

    def create_mandel(self):
        # ssize is the dimensions of the sample array, in samples across and down.
        self.ssize = self.size[0]*self.supersample, self.size[1]*self.supersample
        
        # pixsize is the size of a single sample, in real units.
        self.pixsize = max(self.diam[0] / self.ssize[0], self.diam[1] / self.ssize[1])
        
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.pixsize
        dy = math.sin(rad) * self.pixsize

        # The upper-left corner is computed from the center, minus the radii,
        # plus half a pixel, so that we're sampling the center of the pixel.
        self.eng.ridxdy = (dx, dy, dy, -dx)
        halfsizew = self.ssize[0]/2.0 - 0.5
        halfsizeh = self.ssize[1]/2.0 - 0.5
        self.eng.ri0 = (
            self.center[0] - halfsizew * self.eng.ridxdy[0] - halfsizeh * self.eng.ridxdy[2],
            self.center[1] - halfsizew * self.eng.ridxdy[1] - halfsizeh * self.eng.ridxdy[3]
            )
 
        self.eng.iter_limit = self.iter_limit
        self.eng.trace_boundary = 1
        self.progress = NullProgressReporter()
    
        # Set bailout differently based on continuous or discrete coloring.
        if self.continuous:
            self.eng.bailout = 100.0
        else:
            self.eng.bailout = 2.0
        
        # Continuous is really two different controls in the engine.
        self.eng.cont_levels = self.eng.blend_colors = 256 if self.continuous else 1
        
        # Different modes require different settings.
        if self.mode == 'mandelbrot':
            self.eng.julia = 0
            self.eng.rijulia = (0, 0)
            self.eng.trace_boundary = 1
            self.eng.check_cycles = 1
        elif self.mode == 'julia':
            self.eng.julia = 1
            self.eng.rijulia = tuple(self.rijulia)
            self.eng.trace_boundary = 0
            self.eng.check_cycles = 0
        else:
            raise Exception("Unknown mode: %r" % (self.mode,))
        
        # Create new workspaces for the compute engine.
        old_counts = self.counts
        self.counts = numpy.zeros((self.ssize[1], self.ssize[0]), dtype=numpy.uint32)
        self.status = numpy.zeros((self.ssize[1], self.ssize[0]), dtype=numpy.uint8)

        # Figure out if we can keep any of our old counts or not.
        if (old_counts is not None and
            self.pixsize == self.old_pixsize and
            self.angle == self.old_angle and
            not self.computation_changed()):
            # All the params are compatible, see how much we shifted.
            dx, dy = self.pixel_from_coords(*self.old_ri0)
            dx = int(round(dx))
            dy = int(round(dy))
            
            # Figure out what rectangle is still valid, keep in mind the old
            # and new rectangles could be different sizes.
            nc = min(self.counts.shape[1] - abs(dx), old_counts.shape[1])
            nr = min(self.counts.shape[0] - abs(dy), old_counts.shape[0])
            
            if nc > 0 and nr > 0:
                # Some rows and columns are shared between old and new.
                if dx >= 0:
                    oldx, newx = 0, dx
                else:
                    oldx, newx = -dx, 0
                if dy >= 0:
                    oldy, newy = 0, dy
                else:
                    oldy, newy = -dy, 0
                
                # Copy the common rectangles.  Old_counts gets copied to counts,
                # and status gets the common rectangle filled with 2's.
                self.counts[newy:newy+nr,newx:newx+nc] = old_counts[oldy:oldy+nr,oldx:oldx+nc]
                self.status[newy:newy+nr,newx:newx+nc] = 2  # 2 == Fully computed and filled
                
        self.pixels_computed = False
        self._clear_old_geometry()
    
    def clear_results(self):
        """ Discard any results held.
        """
        self.counts = None

    def copy_all(self, other):
        """ Copy the important attributes from other to self.
        """
        self.copy_geometry(other)
        self.copy_coloring(other)
        self.copy_computation(other)

    def copy_geometry(self, other):
        """ Copy the geometry attributes from other to self, returning True if
            any of them actually changed.
        """
        return self._copy_attributes(other, self._geometry_attributes)

    def copy_coloring(self, other):
        """ Copy the coloring attributes from other to self, returning True if
            any of them actually changed.
        """
        return self._copy_attributes(other, self._coloring_attributes)

    def copy_computation(self, other):
        """ Copy the computation attributes from other to self, returning True if
            any of them actually changed.
        """
        return self._copy_attributes(other, self._computation_attributes)
    
    def _copy_attributes(self, other, attrs):
        """ Copy a list of attributes from other to self, returning True if
            any of them actually changed.
        """
        changed = False
        for attr in attrs:
            # Detect if changed, then copy the attribute regardless.  This makes
            # the .palette copy every time, which guarantees proper drawing at
            # the expense of a lot of palette copying.
            if getattr(self, attr) != getattr(other, attr):
                changed = True
            otherval = copy.deepcopy(getattr(other, attr))
            setattr(self, attr, otherval)
        return changed
    
    def color_mandel(self):
        if (self.pix is None) or (self.pix.shape != self.counts.shape):
            self.pix = numpy.zeros((self.counts.shape[0], self.counts.shape[1], 3), dtype=numpy.uint8)
        # Modulo in C is ill-defined if anything is negative, so make sure the
        # phase is positive if we're going to wrap.
        phase = self.palette_phase
        color_bytes = self.palette.color_bytes()
        if self.palette.wrap:
            phase %= len(color_bytes)
        self.eng.apply_palette(
            self.counts, color_bytes, phase, self.palette_scale,
            self.palette.incolor, self.palette.wrap, self.pix
            )
        return self.pix
    
    def compute_pixels(self):
        if self.pixels_computed:
            return

        if not self.quiet:
            print "ri %r step %r, angle %.1f, iter_limit %r, size %r" % (
                self.eng.ri0, self.pixsize, self.angle, self.eng.iter_limit, self.ssize
                )
            print "center %r, diam %r" % (self.center, self.diam)
        self.eng.clear_stats()
        self.progress.begin()
        # Figure out how many pixels have to be computed: make a histogram of
        # the three buckets of values: 0,1,2.
        buckets, _ = numpy.histogram(self.status, 3, (0, 2))
        num_compute = buckets[0]
        self.eng.compute_array(self.counts, self.status, num_compute, self.progress.progress)
        self.progress.end()
        self._record_old_geometry()
        self.pixels_computed = True
        # Once compute_array is done, the status array is all 2's, so there's no
        # point in keeping it around.
        self.status = None
        
    # Information methods
    
    def coords_from_pixel(self, x, y):
        """ Get the coords of a pixel in the grid. Note that x and y can be
            fractional.
        """
        # The .5 adjustment is because the grid is aligned to the center of the
        # pixels, but we need to return the upper-left of the pixel so that other
        # math comes out right.
        x = float(x) - 0.5
        y = float(y) - 0.5
        r = self.eng.ri0[0] + self.eng.ridxdy[0]*x + self.eng.ridxdy[2]*y
        i = self.eng.ri0[1] + self.eng.ridxdy[1]*x + self.eng.ridxdy[3]*y
        return r, i

    def pixel_from_coords(self, r, i):
        """ Get the pixel coords containing the fractal coordinates.
        """
        d0, d1, d2, d3 = self.eng.ridxdy
        ri00, ri01 = self.eng.ri0
        # Thanks, Maxima!
        x = (d2*(i-ri01)+d3*ri00-d3*r)/(d1*d2-d0*d3)
        y = -(d0*(i-ri01)+d1*ri00-d1*r)/(d1*d2-d0*d3)
        return x, y

    # Output-writing methods
    
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
