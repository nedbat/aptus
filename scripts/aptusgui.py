#!/usr/bin/env python
import aptus, aptus.gui, sys

try:
    aptus.gui.main(sys.argv[1:])
except aptus.AptusException as ae:
    print("Oh no! %s" % ae)
