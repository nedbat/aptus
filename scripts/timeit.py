from aptus.compute import AptusCompute
from aptus.progress import NullProgressReporter

import sys, time

def timeit(args):
    m = AptusCompute()
    m.progress = NullProgressReporter()

    nruns = 100
    grandtotal = 0
    
    if not args:
        case = 'a'
    else:
        case = args[0]
        
    if case == 'a':
        m.center = -0.6, 0.0
        m.diam = 3.0, 3.0
        m.size = 600, 600
    elif case == 'b':
        m.center = -1.8605327670201655, -1.2705648690517021e-005
        m.diam = 2.92062690996144e-010, 2.92062690996144e-010
        m.size = 600, 600
        m.iter_limit = 99999
        nruns = 20
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
