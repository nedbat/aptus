""" Options handling for Aptus.
"""

import optparse, sys
from aptus.palettes import Palette
from aptus.importer import importer
from aptus.tinyjson import JsonReader, JsonWriter

Image = importer('Image')

description = """\
Aptus renders Mandelbrot set images. Two flavors are available:
aptusgui.py for interactive exploration, and aptuscmd.py for
high-quality rendering.
""".replace('\n', ' ')

class AptusOptions:
    def __init__(self, target):
        self.target = target
        
    def read_args(self, argv):
        parser = optparse.OptionParser(
            usage="%prog [options] [parameterfile]",
            description=description
        )
        parser.add_option("-a", "--angle", dest="rotation", help="set the angle of rotation")
        parser.add_option("-b", "--bailout", dest="bailout", help="set the radius of the escape circle")
        parser.add_option("-c", "--continuous", dest="continuous", help="use continuous coloring", action="store_true")
        parser.add_option("-i", "--iterlimit", dest="iter_limit", help="set the limit on the iteration count")
        parser.add_option("-o", "--output", dest="outfile", help="set the output filename (aptuscmd.py only)")
        parser.add_option("--phase", dest="palette_phase", help="set the palette phase", metavar="PHASE")
        parser.add_option("--pscale", dest="palette_scale", help="set the palette scale", metavar="SCALE")
        parser.add_option("-s", "--size", dest="size", help="set the pixel size of the image", metavar="WIDxHGT")
        parser.add_option("--super", dest="supersample", help="set the supersample rate (aptuscmd.py only)", metavar="S")
        
        options, args = parser.parse_args(argv)

        if len(args) > 0:
            self.opts_from_file(args[0])

        if options.rotation:
            self.target.rotation = float(options.rotation)
        if options.bailout:
            self.target.bailout = float(options.bailout)
        if options.continuous:
            self.target.continuous = options.continuous
        if options.iter_limit:
            self.target.iter_limit = int(options.iter_limit)
        if options.outfile:
            self.target.outfile = options.outfile
        if options.palette_phase:
            self.target.palette_phase = int(options.palette_phase)
        if options.palette_scale:
            self.target.palette_scale = float(options.palette_scale)
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
            self.target.rotation = xaos.rotation
            self.target.iter_limit = xaos.maxiter
            self.target.palette = xaos.palette
            self.target.palette_phase = xaos.palette_phase
        
        elif fname.endswith('.png'):
            im = Image.open(fname)
            if "Aptus State" in im.info:
                aptst = AptusState(self.target)
                aptst.read_string(im.info["Aptus State"])
            else:
                raise Exception("PNG file has no Aptus state information: %s" % fname)
        
        elif fname.endswith('.xet'):
            xet = XetState()
            xet.read(fname)
            self.target.center = xet.center
            self.target.diam = xet.diam
            self.target.iter_limit = xet.iter_limit
            self.target.size = xet.size

        else:
            raise Exception("Don't know how to read options from %s" % fname)
    
class AptusState:
    """ A serialization class for the state of an Aptus rendering.
        The result is a JSON representation.
    """
    def __init__(self, target):
        self.target = target
        
    def write(self, f):
        if isinstance(f, basestring):
            f = open(f, 'wb')
        f.write(self.write_string())
    
    simple_attrs = "center diam rotation iter_limit palette_phase palette_scale supersample continuous".split()
    
    def write_string(self):
        d = {'Aptus State':1}

        for sa in self.simple_attrs:
            d[sa] = getattr(self.target, sa)
        d['size'] = list(self.target.size)
        d['palette'] = self.target.palette.spec()
        return JsonWriter().dumps_dict(d, comma=',\n', colon=': ', first_keys=['Aptus State'])
    
    def read(self, f):
        if isinstance(f, basestring):
            f = open(f, 'r')
        return self.read_string(f.read())
    
    def read_string(self, s):
        d = JsonReader().loads(s)
        for sa in self.simple_attrs:
            if sa in d:
                setattr(self.target, sa, d[sa])
        self.target.palette = Palette().from_spec(d['palette'])
        self.target.size = d['size']
        

class XaosState:
    """ The state of a Xaos rendering.
    """
    def __init__(self):
        self.maxiter = 170
        self.center = -0.75, 0.0
        self.diam = 2.55, 2.55
        self.rotation = 0.0
        self.palette_phase = 0
        self.palette = Palette().xaos()
        
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
    
    def handle_angle(self, op, angle):
        self.rotation = self.read_float(angle)
        
    def read_float(self, fstr):
        # Xaos writes out floats with extra characters tacked on the end sometimes.
        # Very ad-hoc: try converting to float, and if it fails, chop off trailing
        # chars until it works.
        while True:
            try:
                return float(fstr)
            except:
                fstr = fstr[:-1]

class XetState:
    """ The state of a .xet file from http://hbar.servebeer.com/art/mandelbrot/index-1.html
    """
    def read(self, f):
        if isinstance(f, basestring):
            f = open(f)
        # skip the binary crap.
        f.read(30)
        xet = {}
        for l in f:
            try:
                name, val = l.split(':')
            except:
                continue
            xet[name.strip()] = val.strip()

        x = float(xet['x'])
        y = float(xet['y'])
        dx = float(xet['dx'])
        depth = int(xet['depth'])
        w = float(xet['width'])
        h = float(xet['height'])
        
        self.center = (x + w*dx/2, y + h*dx/2)
        self.diam = (min(w,h)*dx,)*2
        self.iter_limit = depth
        self.size = (w, h)
