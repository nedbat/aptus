#!/usr/bin/env python

from aptus.app import ConsoleProgressReporter, AptusApp
from aptus.importer import importer
from aptus.options import AptusOptions
from aptus.palettes import all_palettes

Image = importer('Image')

import sys

class AptusCmdApp(AptusApp):
    def main(self, args):
        """ The main for the Aptus command-line tool.
        """
        opts = AptusOptions(self)
        opts.read_args(args)
        
        m = self.create_mandel()
        
        m.progress = ConsoleProgressReporter()
        m.compute_pixels()
        pix = self.color_mandel(m)
        im = Image.fromarray(pix)
        if self.supersample > 1:
            print "Resampling image..."
            im = im.resize(self.size, Image.ANTIALIAS)
        self.write_image(im, self.outfile, mandel=m)

def main(args):
    app = AptusCmdApp()
    app.main(args)
    
if __name__ == '__main__':
    main(sys.argv[1:])
