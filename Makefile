PYTHON ?= python

.PHONY: all reanalysis sensitivity cost_effectiveness figures clean

all: reanalysis sensitivity cost_effectiveness figures

reanalysis:
	$(PYTHON) code/reanalysis.py

sensitivity:
	$(PYTHON) code/sensitivity_analyses.py

cost_effectiveness:
	$(PYTHON) code/cost_effectiveness.py

figures:
	$(PYTHON) code/generate_figures.py

clean:
	rm -f output/reanalysis_results.json
	rm -f output/sensitivity_results.json
	rm -f output/cost_effectiveness_results.json
	rm -f figures/figure1_treatment_effects.png figures/figure1_treatment_effects.pdf
	rm -f figures/efigure2_cost_effectiveness.png figures/efigure2_cost_effectiveness.pdf
