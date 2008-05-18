from math import log, exp
cmd = "python scripts\aptuscmd.py -b %(bailout)f -o cilia_%(num)04d.png cilia.aptus"

frames = 40
b0 = 1.999774061
b1 = 1.999774062
e1 = 50
e0 = 100

b0log = log(log(b0))
b1log = log(log(b1))
e1log = log(log(e1))
e0log = log(log(e0))

step0 = b1log-b0log
step1 = e0log-e1log

print "Steps", step0, step1

class LinearFn:
    def __init__(self, x0, y0, x1, y1):
        self.m = (y1-y0)/(x1-x0)
        self.yi = y0 - (x0*self.m)
        print "Linear(%.9f, %.9f, %.9f, %.9f)\nm = %f, yi = %f" % (x0, y0, x1, y1, self.m, self.yi)
        
    def __call__(self, x):
        return self.yi + self.m * x

stepfn = LinearFn(0, step0, frames-1, step1)

print stepfn(0)
print stepfn(39)

l = b0log
for i in range(frames):
    print "%04d: %f %.9f" % (i, l, exp(exp(l)))
    l += stepfn(i)