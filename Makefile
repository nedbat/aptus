# Makefile for utility work on Aptus

RESFILE = src/gui/resources.py

install: build
	python setup.py install

build:
	python setup.py build

rez: $(RESFILE)

$(RESFILE): etc/crosshair.gif
	python /Python25/Scripts/img2py -n Crosshair etc/crosshair.gif $(RESFILE)

clean:
	-rm -rf build dist
	-rm -rf *.egg-info */*.egg-info */*/*.egg-info
	-rm -f MANIFEST
	-rm -f doc/*.png
	-rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	-rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	-rm -f *.pyo */*.pyo */*/*.pyo */*/*/*.pyo
	-rm -f *.bak */*.bak */*/*.bak */*/*/*.bak
	-rm -f *.so */*.so */*/*.so */*/*/*.so

kit: build
	python setup.py sdist --formats=gztar
	python setup.py bdist_wininst --bitmap etc/wininst.bmp

icon:
	python scripts/aptuscmd.py etc/icon.aptus -s 47x47
	python scripts/aptuscmd.py etc/icon.aptus -s 31x31
	python scripts/aptuscmd.py etc/icon.aptus -s 15x15

lint: clean
	python -x /Python25/Scripts/pylint.bat --rcfile=.pylintrc src
	python checkeol.py

test: install
	nosetests

asm:
	gcc.exe -mno-cygwin -mdll -O -Wall -Ic:\\Python25\\lib\\site-packages\\numpy\\core\\include -Ic:\\Python25\\include -Ic:\\Python25\\PC -c ext/engine.c -O3 -g -Wa,-alh > engine.lst

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

pypi:
	python setup.py register

DOWNLOAD_PY = https://raw.githubusercontent.com/nedbat/coveragepy/master/ci/download_gha_artifacts.py
download_kits:
	wget -qO - $(DOWNLOAD_PY) | python - nedbat/aptus
	python -m twine check dist/*

SCSS = src/aptus/web/static/style.scss
CSS = src/aptus/web/static/style.css

sass:
	pysassc --style=compact $(SCSS) $(CSS)

livesass:
	echo src/aptus/web/static/style.scss | entr -n pysassc --style=compact $(SCSS) $(CSS)
