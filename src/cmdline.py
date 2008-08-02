#!/usr/bin/env python

from aptus.compute import AptusCompute
from aptus.progress import IntervalProgressReporter, ConsoleProgressReporter
from aptus.importer import importer
from aptus.options import AptusOptions

Image = importer('Image')

import sys

class AptusCmdApp():
    def main(self, args):
        """ The main for the Aptus command-line tool.
        """
        compute = AptusCompute()
        opts = AptusOptions(compute)
        opts.read_args(args)
        compute.create_mandel()
        
        compute.progress = IntervalProgressReporter(60, ConsoleProgressReporter())
        compute.compute_pixels()
        pix = compute.color_mandel()
        im = Image.fromarray(pix)
        if m.supersample > 1:
            print "Resampling image..."
            im = im.resize(m.size, Image.ANTIALIAS)
        compute.write_image(im, m.outfile)

def main(args):
    AptusCmdApp().main(args)
    
if __name__ == '__main__':
    main(sys.argv[1:])
