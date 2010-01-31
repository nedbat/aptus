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

import copy, math, Queue, sys, thread, threading, time


class WorkerPool:
    def __init__(self):
        self.workers = []            # List of threads that will do work.
        self.work = Queue.Queue(0)   # The queue of work items.
        self.num_threads = 2

    def get_ready(self):
        """Call before adding work.  Prepares the pool."""
        if not self.workers:
            for i in range(self.num_threads):
                t = threading.Thread(target=self.worker)
                t.setDaemon(True)
                t.start()
                self.workers.append(t)
        
    def put(self, work_item):
        """Add a work item, which should be (q, compute, coords)."""
        self.work.put(work_item)

    def worker(self):
        """The work function on each of our compute threads."""
        while True:
            result_queue, apt_compute, n_tile, coords = self.work.get()
            apt_compute.compute_some(n_tile, coords)
            result_queue.put(coords)


class BucketCountingProgressReporter:
    """ A progress reporter for parallel tiles.
    """
    def __init__(self, num_buckets, expected_total, reporter):
        self.buckets = [0] * num_buckets
        self.expected_total = expected_total
        self.reporter = reporter

    def begin(self):
        self.reporter.begin()
    
    def progress(self, arg, num_done, info=''):
        """Bucket-counting progress.
        
        `arg` is the number of the tile.  `num_done` is the number of pixels
        computed so far in that tile.
        
        """
        self.buckets[arg] = num_done
        # Compute a fraction, in millionths.
        total = sum(self.buckets)
        frac_done = int(total * 1000000.0 / self.expected_total)
        self.reporter.progress(0, frac_done, "[%2d] %s" % (arg, info))

    def end(self):
        self.reporter.end()


class AptusCompute:
    """ The Mandelbrot compute class.  It wraps the AptEngine to provide pythonic
        convenience.
        
        There are two coordinate systems at work here: the ri plane is the
        fractal plane, real and imaginary floats.  The xy plane are screen coordinates,
        in pixels, usually integers.
    """

    worker_pool = WorkerPool()
    
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
        # 2 for tracing, and 3 for traced.
        self.status = None
        # An array for the output pixels.
        self.pix = None
        # A gray checkerboard
        self.chex = None
        
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
        self.progress = NullProgressReporter()
        self.while_waiting = None

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
                # and status gets the common rectangle filled with 3's.
                self.counts[newy:newy+nr,newx:newx+nc] = old_counts[oldy:oldy+nr,oldx:oldx+nc]
                self.status[newy:newy+nr,newx:newx+nc] = 3  # 3 == Fully computed and filled
        
        # In desperate times, printing the counts and status might help...
        if 0:
            for y in range(self.ssize[1]):
                l = ""
                for x in range(self.ssize[0]):
                    l += "%s%s" % (
                        "_-=@"[self.status[y,x]],
                        "0123456789"[self.counts[y,x]%10]
                        )
                print l

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
        w, h = self.counts.shape
        if (self.chex is None) or (self.chex.shape[:2] != self.counts.shape):
            # Make a checkerboard
            sq = 15
            c = numpy.fromfunction(lambda x,y: ((x//sq) + (y//sq)) % 2, (w,h))
            self.chex = numpy.empty((w,h,3), dtype=numpy.uint8)
            self.chex[c == 0] = (0xAA, 0xAA, 0xAA)
            self.chex[c == 1] = (0x99, 0x99, 0x99)

        self.pix = numpy.copy(self.chex)

        # Modulo in C is ill-defined if anything is negative, so make sure the
        # phase is positive if we're going to wrap.
        phase = self.palette_phase
        color_bytes = self.palette.color_bytes()
        if self.palette.wrap:
            phase %= len(color_bytes)
        self.eng.apply_palette(
            self.counts, self.status, color_bytes, phase, self.palette_scale,
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
    
        self.stats = ComputeStats()

        # Figure out how many pixels have to be computed: make a histogram of
        # the buckets of values: 0,1,2,3.
        buckets, _ = numpy.histogram(self.status, 4, (0, 3))
        num_compute = buckets[0]
        n_side_cuts = 4

        self.bucket_progress = BucketCountingProgressReporter(n_side_cuts**2, num_compute, self.progress)

        self.bucket_progress.begin()
        self.refresh_rate = .5

        #self.eng.debug_callback = self.debug_callback

        if self.worker_pool:
            # Start the threads going.
            self.worker_pool.get_ready()

            # Create work items with the tiles to compute
            result_queue = Queue.Queue(0)
            n_todo = 0
            xcuts = self.cuts(0, self.counts.shape[1], n_side_cuts)
            ycuts = self.cuts(0, self.counts.shape[0], n_side_cuts)
            for i in range(n_side_cuts):
                for j in range(n_side_cuts):
                    coords = (xcuts[j], xcuts[j+1], ycuts[i], ycuts[i+1])
                    self.worker_pool.put((result_queue, self, n_todo, coords))
                    n_todo += 1

            # Wait for the workers to finish, calling our while_waiting function
            # periodically.
            next_time = time.time() + self.refresh_rate
            while n_todo:
                while True:
                    if self.while_waiting and time.time() > next_time:
                        self.while_waiting()
                        next_time = time.time() + self.refresh_rate
                    try:
                        result_queue.get(timeout=self.refresh_rate)
                        n_todo -= 1
                        break
                    except Queue.Empty:
                        pass

        else:
            # Not threading: just compute the whole rectangle right now.
            self.compute_some(0, (0, self.counts.shape[1], 0, self.counts.shape[0]))

        # Clean up
        self.bucket_progress.end()
        self._record_old_geometry()
        self.pixels_computed = True
        # Once compute_array is done, the status array is all 3's, so there's no
        # point in keeping it around.
        self.status = None

    def cuts(self, lo, hi, n):
        """Return a list of n+1 evenly spaced numbers between `lo` and `hi`."""
        return [int(round(lo+float(i)*(hi-lo)/n)) for i in range(n+1)]

    def compute_some(self, n_tile, coords):
        xmin, xmax, ymin, ymax = coords
        stats = self.eng.compute_array(
            self.counts, self.status,
            xmin, xmax, ymin, ymax,
            n_tile, self.bucket_progress.progress
            )
        self.stats += stats

    def debug_callback(self, info):
        print info

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
        info.add_text("Aptus Stats", dumps(self.stats))
        im.save(fpath, 'PNG', pnginfo=info)


class ComputeStats(dict):
    """Collected statistics about the computation."""
    
    # This statmap is also used by gui.StatsPanel
    statmap = [
        { 'label': 'Min iteration', 'key': 'miniter', 'sum': min },
        { 'label': 'Max iteration', 'key': 'maxiter', 'sum': max },
        { 'label': 'Total iterations', 'key': 'totaliter', 'sum': sum },
        { 'label': 'Total cycles', 'key': 'totalcycles', 'sum': sum },
        { 'label': 'Shortest cycle', 'key': 'minitercycle', 'sum': min },
        { 'label': 'Longest cycle', 'key': 'maxitercycle', 'sum': max },
        { 'label': 'Maxed points', 'key': 'maxedpoints', 'sum': sum },
        { 'label': 'Computed points', 'key': 'computedpoints', 'sum': sum },
        { 'label': 'Filled points', 'key': 'filledpoints', 'sum': sum },
        { 'label': 'Boundaries traced', 'key': 'boundaries', 'sum': sum },
        { 'label': 'Boundaries filled', 'key': 'boundariesfilled', 'sum': sum },
        { 'label': 'Longest boundary', 'key': 'longestboundary', 'sum': max },
        { 'label': 'Largest fill', 'key': 'largestfilled', 'sum': max },
        { 'label': 'Min edge iter', 'key': 'miniteredge', 'sum': min },
        ]
        
    def __init__(self):
        for stat in self.statmap:
            self[stat['key']] = None

    def __iadd__(self, other):
        """Accumulate a dict of stats to ourselves."""
        for stat in self.statmap:
            k = stat['key']
            if self[k] is None:
                self[k] = other[k]
            else:
                self[k] = stat['sum']([self[k], other[k]])
        return self
