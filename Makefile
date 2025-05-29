VENV = .venv
PY   = $(VENV)/bin/python
PIP  = $(VENV)/bin/pip

.PHONY: run setup clean

setup:
	pyenv exec python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

run:
	$(PY) main.py

clean:
	rm -rf $(VENV)
