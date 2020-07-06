#!/usr/bin/env python
import aptus, aptus.cmdline, sys

try:
    aptus.cmdline.main(sys.argv[1:])
except aptus.AptusException as ae:
    print("Oh no! %s" % ae)
