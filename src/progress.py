""" Progress reporters for Aptus.
"""

from aptus.timeutil import duration, future
import time

class NullProgressReporter:
    """ Basic interface for reporting rendering progress.
    """
    
    def begin(self):
        """ Called once at the beginning of a render.
        """
        pass
    
    def progress(self, frac_done, info=''):
        """ Called repeatedly to report progress.  `frac_done` is a float between
            zero and one indicating the fraction of work done.  `info` is a
            string giving some information about what just got completed.
        """
        pass
    
    def end(self):
        """ Called once at the end of a render.
        """
        pass
    

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
        
    def progress(self, frac_done, info=''):
        now = time.time()
        if now - self.latest > self.nsec:
            self.reporter.progress(frac_done, info)
            self.latest = now
            
    def end(self):
        self.reporter.end()


class AggregateProgressReporter:
    """ Collect a number of progress reporters into a single unified front.
    """
    def __init__(self):
        self.kids = []
        
    def add(self, reporter):
        self.kids.append(reporter)
    
    def begin(self):
        for kid in self.kids:
            kid.begin()
            
    def progress(self, frac_done, info=''):
        for kid in self.kids:
            kid.progress(frac_done, info)
    
    def end(self):
        for kid in self.kids:
            kid.end()

# Cheap way to measure and average a number of runs.
nruns = 0
totaltotal = 0

class ConsoleProgressReporter:
    """ A progress reporter that writes lines to the console.
    """
    def begin(self):
        self.start = time.time()

    def progress(self, frac_done, info=''):
        now = time.time()
        so_far = int(now - self.start)
        to_go = int(so_far / frac_done * (1-frac_done))
        if info:
            info = '  ' + info
        print "%5.2f%%: %11s done, %11s to go, eta %10s%s" % (
            frac_done*100, duration(so_far), duration(to_go), future(to_go), info
            )
    
    def end(self):
        total = time.time() - self.start
        global totaltotal, nruns
        totaltotal += total
        nruns += 1
        print "Total: %s (%.4fs)" % (duration(total), total)
        #print "Running average: %.6fs over %d runs" % (totaltotal/nruns, nruns)
