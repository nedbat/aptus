#!/usr/bin/env python

from aptus.app import ConsoleProgressReporter, AptusApp, AptusMandelbrot
from aptus.importer import importer
from aptus.options import AptusOptions
from aptus.palettes import all_palettes

Image = importer('Image')

import sys

class AptusCmdApp(AptusApp):
    def main(self, args):
        """ The main for the Aptus command-line tool.
        """
        opts = AptusOptions()
        opts.read_args(args)
        
        self.center = opts.center
        self.diam = opts.diam
        self.size = opts.size
        self.iter_limit = opts.iter_limit
        self.palette = opts.palette or all_palettes[0]
        self.palette_phase = opts.palette_phase
        self.supersample = opts.supersample
        
        m = self.create_mandel()
        
        m.progress = ConsoleProgressReporter()
        m.compute_pixels()
        pix = m.color_pixels(self.palette, opts.palette_phase)
        im = Image.fromarray(pix)
        if self.supersample > 1:
            im = im.resize(self.size, Image.ANTIALIAS)
        self.write_image(im, "foo.png")

def main(args):
    app = AptusCmdApp()
    app.main(args)
    
if __name__ == '__main__':
    main(sys.argv[1:])
