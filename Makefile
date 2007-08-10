# Makefile for utility work on coverage.py

build:
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

# Junk below here

tests:
	python test_coverage.py

WEBHOME = c:/ned/web/stellated/pages/code/modules

publish: kit
	cp coverage.py $(WEBHOME)
	cp test_coverage.py $(WEBHOME)
	cp coverage_coverage.py $(WEBHOME)
	cp coverage.px $(WEBHOME)
	cp dist/coverage*.tar.gz $(WEBHOME)
	
kit:
	python setup.py sdist --formats=gztar
