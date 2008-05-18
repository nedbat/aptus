# Points

""" A simple pair of numbers that can be manipulated arithmetically.

>>> p = Point(10,20)
>>> p
<Point 10, 20>
>>> Point(p)
<Point 10, 20>
>>> Point()
<Point 0, 0>

Points act like 2-tuples:

>>> len(p)
2
>>> a, b = p
>>> print a, b
10 20

Except that you can change their x and y components:

>>> p = Point(10, 20)
>>> p.x = 11
>>> p
<Point 11, 20>

Points can be added (or subtracted, multipled, or divided):

>>> Point(10,20) + Point(1,2)
<Point 11, 22>
>>> Point(10,20) + (17, 23)
<Point 27, 43>
>>> p = Point(10,20)
>>> p += (1,2)
>>> p
<Point 11, 22>
>>> Point() + 1
<Point 1, 1>

Error cases:

>>> Point(1)
Traceback (most recent call last):
TypeError: Don't know how to make a Point from 1
>>> Point() + "hey"
Traceback (most recent call last):
TypeError: Don't know how to add <Point 0, 0> and 'hey'

"""

class Point(object):
    def __init__(self, *args):
        self.x, self.y = 0, 0
        if len(args) == 2:
            self.x, self.y = args
        elif len(args) == 1:
            if isinstance(args[0], Point):
                self.x, self.y = args[0].x, args[0].y
            elif len(args[0]) == 2:
                self.x, self.y = args[0]
            else:
                raise TypeError("Don't know how to make a Point from %r" % args)
        elif len(args) == 0:
            pass
        else:
            raise TypeError("Don't know how to make a Point from %r" % args)
    
    def __repr__(self):
        return "<Point %r, %r>" % (self.x, self.y)
    
    # Methods that make this act like a 2-tuple.
    
    def __iter__(self):
        yield self.x
        yield self.y
        
    def __len__(self):
        return 2
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise IndexError
    
    # Methods that make this work like an arithmetic object.
    
    def get_pair(self, other, op):
        if isinstance(other, (Point, tuple, list)):
            return other
        elif isinstance(other, (int, float)):
            return other, other
        else:
            raise TypeError("Don't know how to %s %r and %r" % (op, self, other))
        
    def __add__(self, other):
        ox, oy = self.get_pair(other, "add")
        return Point(self.x+ox, self.y+oy)
        
    def __iadd__(self, other):
        ox, oy = self.get_pair(other, "add")
        self.x += ox
        self.y += oy
        return self
        
    def __sub__(self, other):
        ox, oy = self.get_pair(other, "subtract")
        return Point(self.x-ox, self.y-oy)

    def __isub__(self, other):
        ox, oy = self.get_pair(other, "subtract")
        self.x -= ox
        self.y -= oy
        return self
        
    def __mul__(self, other):
        ox, oy = self.get_pair(other, "multiply")
        return Point(self.x*ox, self.y*oy)
        
    def __imul__(self, other):
        ox, oy = self.get_pair(other, "multiply")
        self.x *= ox
        self.y *= oy
        return self
        
    def __div__(self, other):
        ox, oy = self.get_pair(other, "divide")
        return Point(self.x/ox, self.y/oy)
        
    def __idiv__(self, other):
        ox, oy = self.get_pair(other, "divide")
        self.x /= ox
        self.y /= oy
        return self
        
if __name__ == '__main__':
    import doctest, sys
    doctest.testmod(verbose=('-v' in sys.argv))
