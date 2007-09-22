""" Options handling for Aptus.
"""

import optparse, sys

class AptusOptions:
    def __init__(self):
        self.center = -0.5, 0.0
        self.diam = 3.0, 3.0
        self.size = 600, 600
        self.maxiter = 999
        self.trace = False
        self.palette_phase = 0
        
    def read_args(self, argv):
        parser = optparse.OptionParser()
        parser.add_option("-i", "--maxiter", dest="maxiter", help="set the maximum iteration count")
        parser.add_option("--phase", dest="palette_phase", help="set the palette phase", metavar="PHASE")
        parser.add_option("-s", "--size", dest="size", help="set the pixel size of the image", metavar="WIDxHGT")
        parser.add_option("-t", "--trace", dest="trace", action="store_true", help="use boundary tracing")
        
        options, args = parser.parse_args(argv)

        if len(args) > 0:
            xaos = XaosState()
            xaos.read(sys.argv[1])
            self.center = xaos.center
            self.diam = xaos.diam
            self.maxiter = xaos.maxiter
            self.palette_phase = xaos.palette_phase
            
        if options.maxiter:
            self.maxiter = int(options.maxiter)
        if options.palette_phase:
            self.palette_phase = int(options.palette_phase)
        if options.size:
            self.size = map(int, options.size.split('x'))
        if options.trace:
            self.trace = True
            
class XaosState:
    """ The state of a Xaos rendering.
    """
    def __init__(self):
        self.maxiter = 170
        self.center = -0.75, 0.0
        self.diam = 2.55, 2.55
        self.palette_phase = 0
        
    def read(self, f):
        if isinstance(f, basestring):
            f = open(f)
        for l in f:
            if l.startswith('('):
                argv = l[1:-2].split()
                if hasattr(self, 'handle_'+argv[0]):
                    getattr(self, 'handle_'+argv[0])(*argv)
                    
    def handle_maxiter(self, op, maxiter):
        self.maxiter = int(maxiter)
        
    def handle_view(self, op, cx, cy, rx, ry):
        self.center = self.read_float(cx), self.read_float(cy)
        self.diam = self.read_float(rx), self.read_float(ry)
        
    def handle_shiftpalette(self, op, phase):
        self.palette_phase = int(phase)
        
    def read_float(self, fstr):
        # Xaos writes out floats with extra characters tacked on the end sometimes.
        # Very ad-hoc: try converting to float, and if it fails, chop off trailing
        # chars until it works.
        while True:
            try:
                return float(fstr)
            except:
                fstr = fstr[:-1]
