import unittest
from aptus.options import *

class OptionsTestTarget:
    def __init__(self):
        self.angle = 0
        self.continuous = False
        self.iter_limit = 10
        self.outfile = None
        self.size = [100, 100]
        
class OptionsTest(unittest.TestCase):
    
    def try_read_args(self, cmdline):
        argv = cmdline.split()
        target = OptionsTestTarget()
        AptusOptions(target).read_args(argv)
        return target

    def testNoArgs(self):
        target = self.try_read_args("")
        self.assertEqual(target.angle, 0)
        self.assertEqual(target.continuous, False)

    def testSize(self):
        target = self.try_read_args("-s 300x200")
        self.assertEqual(target.size, [300, 200])
        target = self.try_read_args("--size 300x200")
        self.assertEqual(target.size, [300, 200])
        target = self.try_read_args("--size=300x200")
        self.assertEqual(target.size, [300, 200])
        target = self.try_read_args("-s 300,200")
        self.assertEqual(target.size, [300, 200])
        target = self.try_read_args("--size 300,200")
        self.assertEqual(target.size, [300, 200])
        target = self.try_read_args("--size=300,200")
        self.assertEqual(target.size, [300, 200])

    def testMisc(self):
        target = self.try_read_args("-c")
        self.assertEqual(target.continuous, True)
