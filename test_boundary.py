""" Unit tests for the boundary tracer.
"""

from boundary import trace_boundary
import unittest

class BoundaryTest(unittest.TestCase):

    def prepare_picture(self, picture):
        lines = picture.split()
        # Make sure it's really a rectangle.
        assert [len(l) for l in lines] == [len(lines[0])]*len(lines)
        return lines, len(lines[0]), len(lines)
    
    
    def count_fn(self, picture):
        lines, _, _ = self.prepare_picture(picture)
        def fn(x,y):
            return ord(lines[y][x])
        return fn
    
    def assert_correct(self, counts, lines, xo, yo):
        y, x = counts.shape
        out = []
        for yi in range(y):
            l = ''
            for xi in range(x):
                l += chr(counts[yi,xi] or 32)
            out.append(l)
        
        self.assertEqual(xo, x)
        self.assertEqual(yo, y)
        self.assertEqual(out, lines)
    
    def try_it(self, picture):
        lines, x, y = self.prepare_picture(picture)
        cfn = self.count_fn(picture)
        counts = trace_boundary(cfn, x, y)
        self.assert_correct(counts, lines, x, y)
        
    def testTestCode(self):
        cfn = self.count_fn("""
                       abcdefghi
                       xyz012345
                       """)
        self.assertEqual(cfn(0,0), 97)
        self.assertEqual(cfn(1,0), 98)
        self.assertEqual(cfn(0,1), 120)

    def test1(self):
        self.try_it("""
            XXXXXX
            XXXXXX
            XXXXXX
            """)
        
    def test2(self):
        self.try_it("""
            XXXXXA
            XXXXXB
            XXYZXC
            XXYZXD
            JKLMNE
            """)
    
    def testMandelbrot(self):
        self.try_it("""
            ~~~~~~~~~~~~~}}}}}}}}}}}}}}}}}}}}||||||||{{{zyvrwuW{|||||}}}}}}~~~~~~~~~~~~
            ~~~~~~~~~~}}}}}}}}}}}}}}}}}}}}|||||||||{{{zyxwoaqwxz{{{|||||}}}}}}~~~~~~~~~
            ~~~~~~~~}}}}}}}}}}}}}}}}}}}|||||||||{{zzzyxvn....Knwyz{{{{||||}}}}}}~~~~~~~
            ~~~~~~}}}}}}}}}}}}}}}}}}||||||||{{zyxuxxxwvuq.....svwwyzzzyr{||}}}}}}}~~~~~
            ~~~~}}}}}}}}}}}}}}}}}|||||{{{{{zzzxt>..qf.............pttfqeqz{|}}}}}}}}~~~
            ~~~}}}}}}}}}}}}}}|||{{{{{{{{{zzzywotn.....................atyz{||}}}}}}}}~~
            ~~}}}}}}}}}||||{{zwvyyyyyyyyyyyxvsP........................swvz{||}}}}}}}}~
            ~}}}}|||||||{{{{zyxvpN[ur]spvwwvi...........................qxz{|||}}}}}}}}
            ~}||||||||{{{{{zyytun.........qq............................avz{|||}}}}}}}}
            ~||||||{zzzzyyxtroqb...........a............................xz{{|||}}}}}}}}
            ~@G::#.6#.(..............................................pvxyz{{||||}}}}}}}
            ~||||||{zzzzyyxtroqb...........a............................xz{{|||}}}}}}}}
            ~}||||||||{{{{{zyytun.........qq............................avz{|||}}}}}}}}
            ~}}}}|||||||{{{{zyxvpN[ur]spvwwvi...........................qxz{|||}}}}}}}}
            ~~}}}}}}}}}||||{{zwvyyyyyyyyyyyxvsP........................swvz{||}}}}}}}}~
            ~~~}}}}}}}}}}}}}}|||{{{{{{{{{zzzywotn.....................atyz{||}}}}}}}}~~
            ~~~~}}}}}}}}}}}}}}}}}|||||{{{{{zzzxt>..qf.............pttfqeqz{|}}}}}}}}~~~
            ~~~~~~}}}}}}}}}}}}}}}}}}||||||||{{zyxuxxxwvuq.....svwwyzzzyr{||}}}}}}}~~~~~
            ~~~~~~~~}}}}}}}}}}}}}}}}}}}|||||||||{{zzzyxvn....Knwyz{{{{||||}}}}}}~~~~~~~
            ~~~~~~~~~~}}}}}}}}}}}}}}}}}}}}|||||||||{{{zyxwoaqwxz{{{|||||}}}}}}~~~~~~~~~
            ~~~~~~~~~~~~~}}}}}}}}}}}}}}}}}}}}||||||||{{{zyvrwuW{|||||}}}}}}~~~~~~~~~~~~
            ~~~~~~~~~~~~~~~~~}}}}}}}}}}}}}}}}}}}}}|||||{zmt{{{||||}}}}}~~~~~~~~~~~~~~~~
            """)

if __name__ == '__main__':
    unittest.main()
