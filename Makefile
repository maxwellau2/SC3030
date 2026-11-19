# Everything reproducible lives in experiments/. This forwards any target there:
#   make            run all experiments, PoCs and figures
#   make help       list targets
.DEFAULT_GOAL := all

Makefile: ;

%:
	@$(MAKE) --no-print-directory -C experiments $@

all:
	@$(MAKE) --no-print-directory -C experiments all
