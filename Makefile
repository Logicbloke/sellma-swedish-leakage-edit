# Full reproduction on CPU (~2 h on 8 cores). Each step writes into results/.
PY = uv run python

all: data pilot direction sweep evaluate plots

data:      ; $(PY) src/make_data.py
pilot:     ; $(PY) src/pilot.py
direction: ; $(PY) src/direction.py
sweep:     ; $(PY) src/sweep.py | tee results/dev_sweep.log
evaluate:  ; $(PY) src/evaluate.py | tee results/test.log
plots:     ; $(PY) src/plots.py

.PHONY: all data pilot direction sweep evaluate plots

pdf:       ; $(PY) report/build_pdf.py
.PHONY: pdf
