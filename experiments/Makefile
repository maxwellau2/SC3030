# Reproduce every experiment, proof of concept and figure from scratch.
#
#   make            setup + data + validate + experiments + poc + figures   (~8 min)
#   make help       list targets
#
# Needs: uv (or python3.12 with venv/pip), git, curl, a C compiler (cc).
# The PDF/PNG chart export (kaleido) also needs Google Chrome or Chromium installed.

PY        := .venv/bin/python -I
UV        := $(shell command -v uv 2>/dev/null)

DBO_REPO  := https://github.com/eash3010/dbo-simulation.git
DBO_REV   := 29f945908ea61b53df0dfb8360b4e36af5af96a5
TRACE     := dbo-simulation/traces/direct.zip
TRACE_URL := https://github.com/eash3010/dbo-simulation/releases/download/v1/direct.zip
TRACE_SHA := d0087f0b98cd453be6ba0d14c61449136f47803feb5c36c0fb9325207dc95190

.PHONY: all help setup data validate experiments e0 e1 e3 e4 e5 \
        poc poc-protocols poc-tommy poc-cuttlefish figures clean distclean

all: validate experiments poc figures

help:
	@echo "make setup          create .venv with pinned packages (requirements.txt)"
	@echo "make data           clone DBO authors' simulator @$(DBO_REV) + download their Azure trace (313 MB)"
	@echo "make validate       check our vectorised port == authors' original code (~25 s)"
	@echo "make experiments    E0-E5 -> results/*.csv (~3 min); or one of: e0 e1 e3 e4 e5"
	@echo "make poc            CloudEx/Onyx/DBO, Beyond-Lamport and Cuttlefish PoCs (~3 min)"
	@echo "make figures        matplotlib figures/ + Plotly charts (figures/plotly, ../slides/charts)"
	@echo "make clean          remove caches;  make distclean  also remove .venv and dbo-simulation/"

# ---------------------------------------------------------------- environment
setup: .venv/.ok
.venv/.ok: requirements.txt
ifdef UV
	uv venv --python 3.12 --allow-existing .venv
	uv pip install --python .venv/bin/python -r requirements.txt
else
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
endif
	touch $@

# ---------------------------------------------------------------- data (authors' code + trace)
data: $(TRACE)

dbo-simulation/.git:
	git clone $(DBO_REPO) dbo-simulation
	git -C dbo-simulation checkout -q $(DBO_REV)

$(TRACE): | dbo-simulation/.git
	curl -L --fail -o $@.part $(TRACE_URL)
	@python3 -c "import hashlib,sys; h=hashlib.sha256(open('$@.part','rb').read()).hexdigest(); \
	  sys.exit(0 if h=='$(TRACE_SHA)' else 'trace checksum mismatch: '+h)"
	mv $@.part $@

# ---------------------------------------------------------------- runs
validate: setup data
	$(PY) fairlab/validate.py

experiments: setup data
	$(PY) fairlab/run_experiments.py

e0 e1 e3 e4 e5: setup data
	$(PY) fairlab/run_experiments.py $(shell echo $@ | tr a-z A-Z)

poc: poc-protocols poc-tommy poc-cuttlefish

poc-protocols: setup data
	$(PY) poc/run_poc.py

poc-tommy: setup data
	$(PY) poc/tommy.py

poc-cuttlefish: setup
	$(PY) poc/cuttlefish/demo.py

figures: setup
	$(PY) fairlab/plots.py
	$(PY) poc/plots_poc.py
	$(PY) fairlab/charts.py

# ---------------------------------------------------------------- cleanup
clean:
	rm -rf cache fairlab/__pycache__ poc/__pycache__ poc/cuttlefish/__pycache__ poc/cuttlefish/native_race

distclean: clean
	rm -rf .venv dbo-simulation
