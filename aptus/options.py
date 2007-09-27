""" Options handling for Aptus.
"""

import optparse, sys
from aptus.palettes import Palette

class AptusOptions:
    def __init__(self):
        self.center = -0.5, 0.0
        self.diam = 3.0, 3.0
        self.size = 600, 600
        self.iter_limit = 999
        self.palette = None
        self.palette_phase = 0
        
    def read_args(self, argv):
        parser = optparse.OptionParser()
        parser.add_option("-i", "--iterlimit", dest="iter_limit", help="set the limit on the iteration count")
        parser.add_option("--phase", dest="palette_phase", help="set the palette phase", metavar="PHASE")
        parser.add_option("-s", "--size", dest="size", help="set the pixel size of the image", metavar="WIDxHGT")
        
        options, args = parser.parse_args(argv)

        if len(args) > 0:
            fname = args[0]
            if fname.endswith('.aptus'):
                aptusstate = AptusState()
                aptusstate.read(fname)
                self.center = aptusstate.center
                self.diam = aptusstate.diam
                self.iter_limit = aptusstate.iter_limit
                self.size = aptusstate.size
                self.palette = aptusstate.palette
                self.palette_phase = aptusstate.palette_phase
                
            if fname.endswith('.xpf'):
                xaos = XaosState()
                xaos.read(fname)
                self.center = xaos.center
                self.diam = xaos.diam
                self.iter_limit = xaos.maxiter
                self.palette_phase = xaos.palette_phase
            
        if options.iter_limit:
            self.iter_limit = int(options.iter_limit)
        if options.palette_phase:
            self.palette_phase = int(options.palette_phase)
        if options.size:
            self.size = map(int, options.size.split('x'))

class AptusState:
    """ A serialization class for the state of an Aptus rendering.
    """
    def write(self, f):
        if isinstance(f, basestring):
            f = open(f, 'wb')
        print >>f, '{"Aptus state":1,'
        self._write_item(f, 'center', list(self.center))
        self._write_item(f, 'diam', list(self.diam))
        self._write_item(f, 'iter_limit', self.iter_limit)
        self._write_item(f, 'size', list(self.size))
        self._write_item(f, 'palette', self.palette.spec)
        self._write_item(f, 'palette_phase', self.palette_phase, last=True)
        print >>f, '}'
    
    def read(self, f):
        if isinstance(f, basestring):
            f = open(f, 'rb')
        # This is dangerous!
        d = eval(f.read())
        self.size = d['size']
        self.center = d['center']
        self.diam = d['diam']
        self.iter_limit = d['iter_limit']
        self.palette = Palette().from_spec(d['palette'])
        self.palette_phase = d['palette_phase']
        
    def _write_item(self, f, k, v, last=False):
        if last:
            trailing = ""
        else:
            trailing = ","
        print >> f, '"%s": %r%s' % (k, v, trailing)

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
