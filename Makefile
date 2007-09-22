# Makefile for utility work on mand.py

EXTENSION = mandext.pyd

install:
	python setup.py build install

build: $(EXTENSION)

$(EXTENSION): mandext.c setup.py
	python setup.py build 
	cp build/lib.win32-2.4/mandext.pyd .

clean:
	-rm -rf build
	-rm -rf dist
	-rm -f MANIFEST
	-rm -f mandext.pyd
	-rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	-rm -f *.pyo */*.pyo */*/*.pyo */*/*/*.pyo
	-rm -f *.bak */*.bak */*/*.bak */*/*/*.bak
