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
    
class ConsoleProgressReporter:
    """ A progress reporter that writes lines to the console every ten seconds.
    """
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
