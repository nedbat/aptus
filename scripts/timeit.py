from aptus.compute import AptusCompute
from aptus.progress import NullProgressReporter

import sys, time

def timeit(args):
    m = AptusCompute()
    m.progress = NullProgressReporter()
    m.size = 600, 600

    nruns = 100
    grandtotal = 0
    
    if not args:
        case = 'a'
    else:
        case = args[0]
        
    if case == 'a':
        m.center = -0.6, 0.0
        m.diam = 3.0, 3.0
    elif case == 'b':
        m.center = -1.8605327670201655, -1.2705648690517021e-005
        m.diam = 2.92062690996144e-010, 2.92062690996144e-010
        m.iter_limit = 99999
        nruns = 20
    elif case == 'c':
        m.center = -1.0030917862909408, -0.28088298837940889
        m.diam = 1.3986199517069311e-008, 1.2034636731605979e-008
        m.iter_limit = 99999
        nruns = 5
    else:
        print "huh?"
        return
    
    for i in range(nruns):
        m.clear_results()
        m.create_mandel()
        start = time.time()
        m.compute_pixels()
        total = time.time() - start
        print "%.4f" % total
        grandtotal += total
        
    print "Average %.5f over %d runs" % (grandtotal/nruns, nruns)
    

if __name__ == '__main__':
    timeit(sys.argv[1:])
