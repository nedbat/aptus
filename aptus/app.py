""" Fundamental application building blocks for Aptus.
"""

from aptus import __version__
from aptus.importer import importer
from aptus.options import AptusState
from aptus.timeutil import duration, future

# Import our extension engine.
AptEngine = importer('AptEngine')

numpy = importer('numpy')

import time

class NullProgressReporter:
    def begin(self):
        pass
    
    def progress(self, frac_done):
        pass
    
    def end(self):
        pass
    
class AptusApp:
    """ A mixin class for any Aptus application.
    """
    def __init__(self):
        self.center = -0.5, 0.0
        self.diam = 3.0, 3.0
        self.size = 600, 600
        self.iter_limit = 999
        self.palette = None
        self.palette_phase = 0
        self.supersample = 1

    def create_mandel(self):
        size = self.size[0]*self.supersample, self.size[1]*self.supersample
        return AptusMandelbrot(self.center, self.diam, size, self.iter_limit)
                
    def write_image(self, im, fpath):
        # PNG info mojo from: http://blog.modp.com/2007/08/python-pil-and-png-metadata-take-2.html
        from PIL import PngImagePlugin
        aptst = AptusState()
        self.write_state(aptst)
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "Aptus %s" % __version__)
        info.add_text("Aptus State", aptst.write_string())
        im.save(fpath, 'PNG', pnginfo=info)
    
    def write_state(self, aptst):
        """ Write our state to an AptusState instance.
        """
        aptst.center = self.center
        aptst.diam = self.diam
        aptst.iter_limit = self.iter_limit
        aptst.size = self.size
        aptst.palette = self.palette
        aptst.palette_phase = self.palette_phase
        aptst.supersample = self.supersample
        
class AptusMandelbrot(AptEngine):
    def __init__(self, center, diam, size, iter_limit):
        self.size = size
        
        pixsize = max(diam[0] / size[0], diam[1] / size[1])
        diam = pixsize * size[0], pixsize * size[1]
        
        self.xy0 = (center[0] - diam[0]/2, center[1] - diam[1]/2)
        self.xyd = (pixsize, pixsize)
        #print "Coords: (%r,%r,%r,%r)" % (self.xcenter, self.ycenter, xdiam, ydiam)
 
        self.iter_limit = iter_limit
        self.progress = NullProgressReporter()
        self.counts = None
        
    def coords_from_pixel(self, x, y):
        return self.xy0[0]+self.xyd[0]*x, self.xy0[1]+self.xyd[1]*y

    def compute_pixels(self):
        if self.counts is not None:
            return
        print "x, y %r step %r, iter_limit %r, size %r" % (self.xy0, self.xyd, self.iter_limit, self.size)

        self.clear_stats()
        self.progress.begin()
        self.counts = numpy.zeros((self.size[1], self.size[0]), dtype=numpy.uint32)
        self.mandelbrot_array(self.counts, self.progress.progress)
        self.progress.end()
        print self.get_stats()

    def color_pixels(self, palette, phase):
        palarray = numpy.array(palette.colors, dtype=numpy.uint8)
        pix = palarray[(self.counts+phase) % palarray.shape[0]]
        pix[self.counts == 0] = palette.incolor
        return pix

class ConsoleProgressReporter:
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
            print "%.2f%%: %10s done, %10s to go, eta %10s%s" % (
                frac_done*100, duration(so_far), duration(to_go), future(to_go), info
                )
            self.latest = now
    
    def end(self):
        total = time.time() - self.start
        print "Total: %s (%.2fs)" % (duration(total), total)
