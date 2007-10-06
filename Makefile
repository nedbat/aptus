# Makefile for utility work on Aptus

install: build
	python setup.py install

build: 
	python setup.py build 

clean:
	-rm -rf build
	-rm -rf dist
	-rm -f MANIFEST
	-rm -f mandext.pyd
	-rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	-rm -f *.pyo */*.pyo */*/*.pyo */*/*/*.pyo
	-rm -f *.bak */*.bak */*/*.bak */*/*/*.bak

kit: build
	python setup.py sdist --formats=gztar
	python setup.py bdist_wininst --bitmap etc/wininst.bmp

doc:
	python /Python25/Scripts/rst2html.py --template=rst_template.txt --link-stylesheet README.txt README.html
