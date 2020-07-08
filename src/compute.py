""" Mandelbrot computation.
"""

import multiprocessing
import queue
import random
import threading
import time
import types

import numpy

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
        self.size = (600, 600)

        # computation
        self.iter_limit = 999

        # coloring
        self.palette = all_palettes[0]
        self.palette_phase = 0
        self.palette_scale = 1.0

        # other
        self.outfile = 'Aptus.png'
        self.quiet = False

        # The C extension for doing the heavy lifting.
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

    def create_mandel(self):
        # ssize is the dimensions of the sample array, in samples across and down.
        self.ssize = self.size[0], self.size[1]

        self.while_waiting = None

        # Create new workspaces for the compute engine.
        self.counts = numpy.zeros((self.ssize[1], self.ssize[0]), dtype=numpy.uint32)
        self.status = numpy.zeros((self.ssize[1], self.ssize[0]), dtype=numpy.uint8)

    def color_mandel(self):
        if (self.chex is None) or (self.chex.shape[:2] != self.counts.shape):
            w, h = self.counts.shape
            # Make a checkerboard
            sq = 15
            c = numpy.fromfunction(lambda x,y: ((x//sq) + (y//sq)) % 2, (w,h))
            self.chex = numpy.empty((w,h,3), dtype=numpy.uint8)
            self.chex[c == 0] = (0xAA, 0xAA, 0xAA)
            self.chex[c == 1] = (0x99, 0x99, 0x99)

        self.pix = numpy.copy(self.chex)
        return self.pix

    def compute_pixels(self):
        x_side_cuts, y_side_cuts = 2, 2

        refresh_rate = .5

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
            next_time = time.time() + refresh_rate
            while n_todo:
                while True:
                    if self.while_waiting and time.time() > next_time:
                        self.while_waiting()
                        next_time = time.time() + refresh_rate
                    try:
                        result_queue.get(timeout=refresh_rate)
                        n_todo -= 1
                        break
                    except queue.Empty:
                        pass

        else:
            # Not threading: just compute the whole rectangle right now.
            self.compute_some(0, (0, self.counts.shape[1], 0, self.counts.shape[0]))

        # Clean up
        # Once compute_array is done, the status array is all 3's, so there's no
        # point in keeping it around.
        self.status = None

    def cuts(self, lo, hi, n):
        """Return a list of n+1 evenly spaced numbers between `lo` and `hi`."""
        return [int(round(lo+float(i)*(hi-lo)/n)) for i in range(n+1)]

    def compute_some(self, n_tile, coords):
        print(f"compute_some({n_tile}, {coords})")
        if self.iter_limit > 1000:
            time.sleep(random.randint(2, 10))
