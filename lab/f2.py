# An experiment in using pairs of floats to get better precision.

class f2:
    """ Two floats, to get split precision.
    """
    def __init__(self, a, b):
        self.a = float(a) + float(b)
        self.b = float(b) - (self.a - float(a))
        
    def __repr__(self):
        return "<%r+%r>" % (self.a, self.b)
    
    def __mul__(self, o):
        if isinstance(o, f2):
            return f2(self.a*o.a, self.a*o.b + self.b*o.a + self.b*o.b)
        else:
            return f2(self.a * o, self.b * o)
    
    def __add__(self, o):
        if isinstance(o, f2):
            return f2(self.a+o.a, self.b+o.b)
        else:
            return f2(self.a + o, self.b)
        
    def __sub__(self, o):
        if isinstance(o, f2):
            return f2(self.a-o.a, self.b-o.b)
        else:
            return f2(self.a - o, self.b)
        
    def __float__(self):
        return self.a + self.b

class c2:
    """ A simple complex number.
    """
    def __init__(self, r, i):
        self.r, self.i = r, i
        
    def __repr__(self):
        return "(%r + %ri)" % (self.r, self.i)
    
    def __mul__(self, o):
        return c2(self.r*o.r - self.i*o.i, self.r*o.i + self.i*o.r)
    
    def __add__(self, o):
        return c2(self.r+o.r, self.i+o.i)
    
    def __sub__(self, o):
        return c2(self.r-o.r, self.i-o.i)
    
import sys

if 0:
    f = f2(.345678, 1.1e-20)
    g = float(f)
    p = .25
    
    for i in range(100000):
        print f, g
        f = f*f+p
        g = g*g+p

if __name__ == '__main__':
    dx = 5.5951715923569399e-018
    x = -0.70654266100607843
    y = -0.36491281470843084
    
    for xi in range(100):
        p = c2(f2(x,xi*dx), f2(y,0))
        z = c2(f2(0,0),f2(0,0))
        
        for i in range(10000):
            z = z*z+p
            if float(z.r) > 2.0:
                print p, i
                break
