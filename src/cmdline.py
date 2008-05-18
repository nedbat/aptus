#!/usr/bin/env python

from aptus.app import AptusApp
from aptus.progress import ConsoleProgressReporter
from aptus.importer import importer
from aptus.options import AptusOptions

Image = importer('Image')

import sys

class AptusCmdApp():
    def main(self, args):
        """ The main for the Aptus command-line tool.
        """
        m = AptusApp()
        opts = AptusOptions(m)
        opts.read_args(args)
        m.create_mandel()
        
        m.progress = ConsoleProgressReporter()
        m.compute_pixels()
        pix = m.color_mandel()
        im = Image.fromarray(pix)
        if m.supersample > 1:
            print "Resampling image..."
            im = im.resize(m.size, Image.ANTIALIAS)
        m.write_image(im, m.outfile)

def main(args):
    AptusCmdApp().main(args)
    
if __name__ == '__main__':
    main(sys.argv[1:])
