# Makefile for utility work on Aptus

install: build
	python setup.py install

build: 
	python setup.py build 

clean:
	-rm -rf build
	-rm -rf dist
	-rm -f MANIFEST
	-rm -f doc/sample.png
	-rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	-rm -f *.pyo */*.pyo */*/*.pyo */*/*/*.pyo
	-rm -f *.bak */*.bak */*/*.bak */*/*/*.bak

kit: build
	python setup.py sdist --formats=gztar
	python setup.py bdist_wininst --bitmap etc/wininst.bmp

doc:
	python /Python25/Scripts/rst2html.py --template=etc/rst_template.txt --link-stylesheet README.txt README.html

icon:
	python scripts/aptuscmd.py etc/icon.aptus -s 47x47
	python scripts/aptuscmd.py etc/icon.aptus -s 31x31
	python scripts/aptuscmd.py etc/icon.aptus -s 15x15

WEBHOME = c:/ned/web/stellated/pages/code/aptus

publish: kit
	python scripts/aptuscmd.py doc/sample.aptus --super=3 -o doc/sample.png
	cp -v doc/*.* $(WEBHOME)
	cp -v dist/*.* $(WEBHOME)
