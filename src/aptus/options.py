""" Options handling for Aptus.
"""

import json
import optparse
import re

from PIL import Image

from aptus.palettes import Palette

description = """\
Aptus renders Mandelbrot set images. Three flavors are available:
aptusweb and aptusgui for interactive exploration, and aptuscmd for
high-quality rendering.
""".replace('\n', ' ')


class AptusOptions:
    """ An option parser for Aptus states.
    """

    def __init__(self, target):
        """ Create an AptusOptions parser.  Attributes are set on the target, which
            should be an AptusCompute-like thing.
        """
        self.target = target

    def _create_parser(self):
        parser = optparse.OptionParser(
            usage="%prog [options] [parameterfile]",
            description=description
        )
        parser.add_option("-a", "--angle", dest="angle", help="set the angle of rotation")
        parser.add_option("--center", dest="center", help="set the center of the view", metavar="RE,IM")
        parser.add_option("-c", "--continuous", dest="continuous", help="use continuous coloring", action="store_true")
        parser.add_option("--diam", dest="diam", help="set the diameter of the view")
        parser.add_option("-i", "--iterlimit", dest="iter_limit", help="set the limit on the iteration count")
        parser.add_option("-o", "--output", dest="outfile", help="set the output filename (aptuscmd only)")
        parser.add_option("--phase", dest="palette_phase", help="set the palette phase", metavar="PHASE")
        parser.add_option("--pscale", dest="palette_scale", help="set the palette scale", metavar="SCALE")
        parser.add_option("-s", "--size", dest="size", help="set the pixel size of the image", metavar="WIDxHGT")
        parser.add_option("--super", dest="supersample",
                          help="set the supersample rate (aptuscmd only)", metavar="S")
        return parser

    def _pair(self, s, cast):
        """ Convert a string argument to a pair of other casted values.
        """
        vals = list(map(cast, re.split("[,x]", s)))
        if len(vals) == 1:
            vals = vals*2
        return vals

    def _int_pair(self, s):
        """ Convert a string argument to a pair of ints.
        """
        return self._pair(s, int)

    def _float_pair(self, s):
        """ Convert a string argument to a pair of floats.
        """
        return self._pair(s, float)

    def read_args(self, argv):
        """ Read aptus options from the provided argv.
        """
        parser = self._create_parser()
        options, args = parser.parse_args(argv)

        if len(args) > 0:
            self.opts_from_file(args[0])

        if options.angle:
            self.target.angle = float(options.angle)
        if options.center:
            self.target.center = self._float_pair(options.center)
        if options.continuous:
            self.target.continuous = options.continuous
        if options.diam:
            self.target.diam = self._float_pair(options.diam)
        if options.iter_limit:
            self.target.iter_limit = int(options.iter_limit)
        if options.outfile:
            self.target.outfile = options.outfile
        if options.palette_phase:
            self.target.palette_phase = int(options.palette_phase)
        if options.palette_scale:
            self.target.palette_scale = float(options.palette_scale)
        if options.size:
            self.target.size = self._int_pair(options.size)
        if options.supersample:
            self.target.supersample = int(options.supersample)

    def options_help(self):
        """ Return the help text about the command line options.
        """
        parser = self._create_parser()
        return parser.format_help()

    def opts_from_file(self, fname):
        """ Read aptus options from the given filename.  Various forms of input
            file are supported.
        """
        if fname.endswith('.aptus'):
            aptst = AptusState(self.target)
            aptst.read(fname)

        elif fname.endswith('.xpf'):
            xaos = XaosState()
            xaos.read(fname)
            self.target.center = xaos.center
            self.target.diam = xaos.diam
            self.target.angle = xaos.angle
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

        else:
            raise Exception("Don't know how to read options from %s" % fname)


class AptusStateError(Exception):
    pass


class AptusState:
    """ A serialization class for the state of an Aptus rendering.
        The result is a JSON representation.
    """
    def __init__(self, target):
        self.target = target

    def write(self, f):
        if isinstance(f, str):
            f = open(f, "w")
        f.write(self.write_string())

    simple_attrs = "center diam angle iter_limit palette_phase palette_scale supersample continuous mode".split()
    julia_attrs = "rijulia".split()

    def write_attrs(self, d, attrs):
        for a in attrs:
            d[a] = getattr(self.target, a)

    def write_string(self):
        d = {'Aptus State':1}

        self.write_attrs(d, self.simple_attrs)
        d['size'] = list(self.target.size)
        d['palette'] = self.target.palette.spec()

        if self.target.mode == 'julia':
            self.write_attrs(d, self.julia_attrs)

        return json.dumps(d)

    def read_attrs(self, d, attrs):
        for a in attrs:
            if a in d:
                setattr(self.target, a, d[a])

    def read(self, f):
        if isinstance(f, str):
            f = open(f, 'r')
        return self.read_string(f.read())

    def read_string(self, s):
        d = json.loads(s)
        self.read_attrs(d, self.simple_attrs)
        self.target.palette = Palette().from_spec(d['palette'])
        self.target.size = d['size']
        self.read_attrs(d, self.julia_attrs)


class XaosState:
    """ The state of a Xaos rendering.
    """
    def __init__(self):
        self.maxiter = 170
        self.center = -0.75, 0.0
        self.diam = 2.55, 2.55
        self.angle = 0.0
        self.palette_phase = 0
        self.palette = Palette().xaos()

    def read(self, f):
        if isinstance(f, str):
            f = open(f)
        for l in f:
            if l.startswith('('):
                argv = l[1:-2].split()
                if hasattr(self, 'handle_'+argv[0]):
                    getattr(self, 'handle_'+argv[0])(*argv)

    def handle_maxiter(self, op_unused, maxiter):
        self.maxiter = int(maxiter)

    def handle_view(self, op_unused, cr, ci, rr, ri):
        # Xaos writes i coordinates inverted.
        self.center = self.read_float(cr), -self.read_float(ci)
        self.diam = self.read_float(rr), self.read_float(ri)

    def handle_shiftpalette(self, op_unused, phase):
        self.palette_phase = int(phase)

    def handle_angle(self, op_unused, angle):
        self.angle = self.read_float(angle)

    def read_float(self, fstr):
        # Xaos writes out floats with extra characters tacked on the end sometimes.
        # Very ad-hoc: try converting to float, and if it fails, chop off trailing
        # chars until it works.
        while True:
            try:
                return float(fstr)
            except ValueError:
                fstr = fstr[:-1]
