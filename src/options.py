""" Options handling for Aptus.
"""

import optparse, sys
from aptus.palettes import Palette
from aptus.importer import importer

Image = importer('Image')

class AptusOptions:
    def __init__(self, target):
        self.target = target
        
    def read_args(self, argv):
        parser = optparse.OptionParser()
        parser.add_option("-i", "--iterlimit", dest="iter_limit", help="set the limit on the iteration count")
        parser.add_option("--phase", dest="palette_phase", help="set the palette phase", metavar="PHASE")
        parser.add_option("-s", "--size", dest="size", help="set the pixel size of the image", metavar="WIDxHGT")
        parser.add_option("--super", dest="supersample", help="set the supersample rate", metavar="S")
        
        options, args = parser.parse_args(argv)

        if len(args) > 0:
            self.opts_from_file(args[0])

        if options.iter_limit:
            self.target.iter_limit = int(options.iter_limit)
        if options.palette_phase:
            self.target.palette_phase = int(options.palette_phase)
        if options.size:
            self.target.size = map(int, options.size.split('x'))
        if options.supersample:
            self.target.supersample = int(options.supersample)

    def opts_from_file(self, fname):
        if fname.endswith('.aptus'):
            aptst = AptusState(self.target)
            aptst.read(fname)
            
        elif fname.endswith('.xpf'):
            xaos = XaosState()
            xaos.read(fname)
            self.target.center = xaos.center
            self.target.diam = xaos.diam
            self.target.iter_limit = xaos.maxiter
            self.target.palette_phase = xaos.palette_phase
        
        elif fname.endswith('.png'):
            im = Image.open(fname)
            if "Aptus State" in im.info:
                aptst = AptusState(self.target)
                aptst.read_string(im.info["Aptus State"])
            else:
                raise Exception("PNG file has no Aptus state information: %s" % fname)
        else:
            raise Exception("Don't know how to read options from %s" % fname)
    
class AptusState:
    """ A serialization class for the state of an Aptus rendering.
    """
    def __init__(self, target):
        self.target = target
        
    def write(self, f):
        if isinstance(f, basestring):
            f = open(f, 'wb')
        f.write(self.write_string())
        
    def write_string(self):
        lines = []
        lines.append(self._write_item('Aptus state', 1))
        lines.append(self._write_item('center', list(self.target.center)))
        lines.append(self._write_item('diam', list(self.target.diam)))
        lines.append(self._write_item('iter_limit', self.target.iter_limit))
        lines.append(self._write_item('size', list(self.target.size)))
        lines.append(self._write_item('palette', self.target.palette.spec))
        lines.append(self._write_item('palette_phase', self.target.palette_phase))
        lines.append(self._write_item('supersample', self.target.supersample))
        return "{" + ",\n".join(lines) + "\n}\n"
    
    def read(self, f):
        if isinstance(f, basestring):
            f = open(f, 'rb')
        return self.read_string(f.read())
    
    def read_string(self, s):
        # This is dangerous!
        d = eval(s)
        self.target.size = d['size']
        self.target.center = d['center']
        self.target.diam = d['diam']
        self.target.iter_limit = d['iter_limit']
        self.target.palette = Palette().from_spec(d['palette'])
        self.target.palette_phase = d['palette_phase']
        self.target.supersample = d['supersample']
        
    def _write_item(self, k, v):
        return '"%s": %r' % (k, v)

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
