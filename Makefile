# Makefile for utility work on Aptus

install: build
	python setup.py install

build: 
	python setup.py build 

clean:
	-rm -rf build
	-rm -rf dist
	-rm -f MANIFEST
	-rm -f doc/*.png
	-rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	-rm -f *.pyo */*.pyo */*/*.pyo */*/*/*.pyo
	-rm -f *.bak */*.bak */*/*.bak */*/*/*.bak

kit: build
	python setup.py sdist --formats=gztar
	python setup.py bdist_wininst --bitmap etc/wininst.bmp

icon:
	python scripts/aptuscmd.py etc/icon.aptus -s 47x47
	python scripts/aptuscmd.py etc/icon.aptus -s 31x31
	python scripts/aptuscmd.py etc/icon.aptus -s 15x15

lint:
	python -x /Python25/Scripts/pylint.bat --rcfile=.pylintrc src
	
WEBHOME = c:/ned/web/stellated/pages/code/aptus
LOCALHOME = c:/www/code/aptus

%.png: %.aptus
	python scripts/aptuscmd.py $< --super=3 -o $*.png -s 1000x740
	python scripts/aptuscmd.py $< --super=3 -o $*_med.png -s 500x370
	python scripts/aptuscmd.py $< --super=5 -o $*_thumb.png -s 250x185

SAMPLE_PNGS := $(patsubst %.aptus,%.png,$(wildcard doc/*.aptus))

samples: $(SAMPLE_PNGS) build

publish_samples: samples
	cp -v doc/*.png $(WEBHOME)

publish_kit: kit
	cp -v dist/*.* $(WEBHOME)

publish_doc:
	cp -v doc/*.px $(WEBHOME)

publish: publish_kit publish_doc publish_samples

local_kit: kit
	cp -v dist/*.* $(LOCALHOME)
