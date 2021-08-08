PYTHON:= python3.7
CUDA_VERSION:= 11.0
PYTORCH_VERSION:= 1.7.1
DOT:= .
.PHONY: all clean

all: virtualenv.done

virtualenv.done:
	test -d venv || $(PYTHON) -m venv venv
	touch virtualenv.done

pytorch.done:
ifeq ($(CUDA_VERSION),)
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cpu/stable.html
	touch pytorch.done
else
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
	touch pytorch.done
endif

parallel_wavegan.done: virtualenv.done pytorch.done
	. venv/bin/activate; cd ../; pip install -e .
	. venv/bin/activate; cd ../; pip install -e .[test]
	touch parallel_wavegan.done

apex.done: virtualenv.done pytorch.done
	git clone https://github.com/NVIDIA/apex.git
	. venv/bin/activate; cd apex; \
		pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	touch apex.done

clean:
	rm -fr venv apex *.done
	find -iname "*.pyc" -delete
