""" Progress reporters for Aptus.
"""

import time

from aptus.timeutil import duration, future


class IntervalProgressReporter:
    """ A progress reporter decorator that only calls its wrapped reporter
        every N seconds.
    """
    def __init__(self, nsec, reporter):
        self.nsec = nsec
        self.reporter = reporter

    def begin(self):
        self.latest = time.time()
        self.reporter.begin()

    def progress(self, arg, num_done, info=''):
        now = time.time()
        if now - self.latest > self.nsec:
            self.reporter.progress(arg, num_done, info)
            self.latest = now

    def end(self):
        self.reporter.end()


# Cheap way to measure and average a number of runs.
nruns = 0
totaltotal = 0

class ConsoleProgressReporter:
    """ A progress reporter that writes lines to the console.

    This `progress` function interprets the `num_done` arg as a fraction, in
    millionths.

    """
    def begin(self):
        self.start = time.time()

    def progress(self, arg, num_done, info=''):
        frac_done = num_done / 1000000.0
        now = time.time()
        so_far = int(now - self.start)
        to_go = int(so_far / frac_done * (1-frac_done))
        if info:
            info = '  ' + info
        print("%5.2f%%: %11s done, %11s to go, eta %10s%s" % (
            frac_done*100, duration(so_far), duration(to_go), future(to_go), info
            ))

    def end(self):
        total = time.time() - self.start
        global totaltotal, nruns
        totaltotal += total
        nruns += 1
        print("Total: %s (%.4fs)" % (duration(total), total))
        #print("Running average: %.6fs over %d runs" % (totaltotal/nruns, nruns))
