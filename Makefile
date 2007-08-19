# Makefile for utility work on mand.py

EXTENSION = mandext.pyd

build: $(EXTENSION)

$(EXTENSION): mandext.c setup.py
	python setup.py build -cmingw32
	cp build/lib.win32-2.4/mandext.pyd .

clean:
	-rm -rf build
	-rm -rf dist
	-rm -f MANIFEST
	-rm -f mandext.pyd
	-rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	-rm -f *.pyo */*.pyo */*/*.pyo */*/*/*.pyo
	-rm -f *.bak */*.bak */*/*.bak */*/*/*.bak
