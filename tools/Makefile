PYTHON:= python3.6
.PHONY: all clean

all: virtualenv

virtualenv:
	test -d venv || virtualenv -p $(PYTHON) venv
	. venv/bin/activate; pip install -U pip
	. venv/bin/activate; cd ../; pip install -e .
	. venv/bin/activate; cd ../; pip install -e .[test]
	touch venv/bin/activate

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
