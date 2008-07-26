from aptus.compute import AptusCompute
from aptus.progress import NullProgressReporter

import time

def timeit():
    m = AptusCompute()
    m.progress = NullProgressReporter()

    nruns = 100
    grandtotal = 0
    
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
    timeit()
