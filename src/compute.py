""" Mandelbrot computation.
"""

import copy
import json
import math
import multiprocessing
import queue
import random
import threading
import time

import numpy

from aptus import __version__, settings
from aptus.options import AptusState
from aptus.palettes import all_palettes


class WorkerPool:
    def __init__(self):
        self.workers = []            # List of threads that will do work.
        self.work = queue.Queue(0)   # The queue of work items.
        self.num_threads = multiprocessing.cpu_count()

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

        # computation
        self.iter_limit = 999
        self.continuous = False
        self.supersample = 1
        self.mode = 'mandelbrot'
        self.rijulia = 0.0, 0.0

        # coloring
        self.palette = all_palettes[0]
        self.palette_phase = 0
        self.palette_scale = 1.0

        # other
        self.outfile = 'Aptus.png'
        self.quiet = False

        # The C extension for doing the heavy lifting.
        import types
        self.eng = types.SimpleNamespace()

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
        ri0x = self.center[0] - halfsizew * self.eng.ridxdy[0] - halfsizeh * self.eng.ridxdy[2]
        ri0y = self.center[1] - halfsizew * self.eng.ridxdy[1] - halfsizeh * self.eng.ridxdy[3]

        # In order for x-axis symmetry to apply, the x axis has to fall between
        # pixels or through the center of a pixel.
        pix_offset, _ = math.modf(ri0y / self.pixsize)
        ri0y -= pix_offset * self.pixsize

        self.eng.ri0 = ri0x, ri0y

        self.eng.iter_limit = self.iter_limit
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

        self.pixels_computed = False

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
        if 0:
            self.eng.apply_palette(
            self.counts, self.status, color_bytes, phase, self.palette_scale,
            self.palette.incolor, self.palette.wrap, self.pix
            )
        return self.pix

    def compute_pixels(self):
        if self.pixels_computed:
            return

        if not self.quiet:
            print("ri %r step %r, angle %.1f, iter_limit %r, size %r" % (
                self.eng.ri0, self.pixsize, self.angle, self.eng.iter_limit, self.ssize
                ))
            print("center %r, diam %r" % (self.center, self.diam))

        # Figure out how many pixels have to be computed: make a histogram of
        # the buckets of values: 0,1,2,3.
        buckets, _ = numpy.histogram(self.status, 4, (0, 3))
        num_compute = buckets[0]
        x_side_cuts, y_side_cuts = self.slice_tiles()

        self.refresh_rate = .5

        if self.worker_pool:
            # Start the threads going.
            self.worker_pool.get_ready()

            # Create work items with the tiles to compute
            result_queue = queue.Queue(0)
            n_todo = 0
            xcuts = self.cuts(0, self.counts.shape[1], x_side_cuts)
            ycuts = self.cuts(0, self.counts.shape[0], y_side_cuts)
            for i in range(y_side_cuts):
                for j in range(x_side_cuts):
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
                    except queue.Empty:
                        pass

        else:
            # Not threading: just compute the whole rectangle right now.
            self.compute_some(0, (0, self.counts.shape[1], 0, self.counts.shape[0]))

        # Clean up
        self.pixels_computed = True
        # Once compute_array is done, the status array is all 3's, so there's no
        # point in keeping it around.
        self.status = None

    def cuts(self, lo, hi, n):
        """Return a list of n+1 evenly spaced numbers between `lo` and `hi`."""
        return [int(round(lo+float(i)*(hi-lo)/n)) for i in range(n+1)]

    def slice_tiles(self):
        """Decide how to divide the current view into tiles for workers.

        Returns two numbers, the number of tiles in the x and y directions.

        """
        # Slice into roughly 200-pixel tiles.
        x, y = max(self.ssize[0]//200, 1), max(self.ssize[1]//200, 1)

        # If the xaxis is horizontal, and is in the middle third of the image,
        # then slice the window into vertical slices to maximize the benefit of
        # the axis symmetry.
        top = self.eng.ri0[1]
        height = self.ssize[1]*self.pixsize
        if self.angle == 0 and top > 0 and height > top:
            axis_frac = top / height
            if .25 < axis_frac < .75:
                # Use tall slices to get axis symmetry
                y = 1

        return x, y

    def compute_some(self, n_tile, coords):
        print(f"compute_some({n_tile}, {coords})")
        if self.iter_limit > 1000:
            time.sleep(random.randint(2, 10))

    def debug_callback(self, info):
        print(info)

    # Information methods

    def pixel_from_coords(self, r, i):
        """ Get the pixel coords containing the fractal coordinates.
        """
        d0, d1, d2, d3 = self.eng.ridxdy
        ri00, ri01 = self.eng.ri0
        # Thanks, Maxima!
        x = (d2*(i-ri01)+d3*ri00-d3*r)/(d1*d2-d0*d3)
        y = -(d0*(i-ri01)+d1*ri00-d1*r)/(d1*d2-d0*d3)
        return x, y
